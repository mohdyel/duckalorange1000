# =========================
import os, re, json, math, gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Modeling
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
# Planet Spectrum Pipeline — XGBoost Baseline + Quality Dashboard (memory-safe, Arrow-fix)
# Author: ChatGPT
# -----------------------------------------------------------------------------
# Practical pipeline to map planet folders → 283-length spectrum (wl_1..wl_283),
# with strict RAM controls and a robust Parquet reader that only converts
# primitive numeric columns (avoids Arrow→Pandas ExtensionArray errors).
# -----------------------------------------------------------------------------

# =========================
# 🔧 CONFIG — EDIT HERE
# =========================
BASE_DIR      = "/kaggle/input/ariel-data-challenge-2025/train"         # root: planet folders (e.g., 1010375142/ ...)
TRAIN_CSV     = "/kaggle/input/ariel-data-challenge-2025/train.csv"     # path: planet_id + wl_1..wl_283
OUT_DIR       = "/kaggle/working/"           # outputs

SEED          = 42

# Hard cap on how many labeled planets to process (None for full run)
MAX_PLANETS   = 1000

# ---- Speed & caching toggles ----
ENABLE_CACHE = True
CACHE_DIR    = os.path.join(OUT_DIR, "feat_cache")
SPEED_PRESET = "fast"    # "fast", "balanced", or "max" (original)

# Sequence representation (defaults; adjusted by preset below)
SEQ_LENGTH    = 2048
MAX_ROWS_PER_RG_SAMPLE = 80_000

# Feature engineering (defaults; may be trimmed by preset)
FFT_BANDS     = 16
ACF_LAGS      = [1,2,4,8,16,32,64,128]
WINDOW_SIZES  = [16, 64, 256, 1024]

# Channel clamp per stream to reduce feature width
MAX_CHANNELS_PER_STREAM = 8

# Force float32 everywhere for arrays
FORCE_FLOAT32 = True

# Training
N_FOLDS       = 5
USE_XGB       = True
TARGET_CHUNK  = 20

XGB_PARAMS    = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=SEED,
    tree_method="hist"
)

# Execution toggles
BUILD_DATASET = True
TRAIN_MODEL   = True
MAKE_PLOTS    = True
EXPORT_PRED   = True

# =========================
# Imports
# =========================


# XGBoost
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
import joblib

# I/O & plots
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# Parquet reading (pyarrow preferred)
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PA_AVAILABLE = True
except Exception:
    PA_AVAILABLE = False
    pa = None
    pq = None

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ---- Apply SPEED_PRESET ----
if SPEED_PRESET == "fast":
    MAX_ROWS_PER_RG_SAMPLE = 20_000
    MAX_CHANNELS_PER_STREAM = 4
    WINDOW_SIZES  = [32, 256]
    FFT_BANDS     = 8
    ACF_LAGS      = [1, 4, 16]
elif SPEED_PRESET == "balanced":
    MAX_ROWS_PER_RG_SAMPLE = 60_000
    MAX_CHANNELS_PER_STREAM = 8
    WINDOW_SIZES  = [16, 64, 256, 1024]
    FFT_BANDS     = 12
    ACF_LAGS      = [1, 2, 4, 8, 16, 32]
# else: "max" keeps the defaults above

# =========================
# Utility helpers
# =========================
def list_planet_dirs(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    dirs = [d for d in dirs if re.fullmatch(r"\d+", d)]
    return sorted(dirs)

def _arrow_numeric_col_indices(schema: "pa.Schema") -> List[int]:
    idxs = []
    for i, field in enumerate(schema):
        t = field.type
        if pa.types.is_floating(t) or pa.types.is_integer(t):
            idxs.append(i)
    return idxs

def read_parquet_safely(path: str, sample_rows: int = MAX_ROWS_PER_RG_SAMPLE) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    if not PA_AVAILABLE:
        df = pd.read_parquet(path)
        num = df.select_dtypes(include=[np.number]).copy()
        if FORCE_FLOAT32 and not num.empty:
            for c in num.columns:
                if np.issubdtype(num[c].dtype, np.floating) and num[c].dtype != np.float32:
                    num[c] = num[c].astype(np.float32, copy=False)
                elif np.issubdtype(num[c].dtype, np.integer):
                    num[c] = num[c].astype(np.float32, copy=False)
        return num

    pf = pq.ParquetFile(path, memory_map=True)
    dfs = []
    rows_so_far = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        keep_idxs = _arrow_numeric_col_indices(tbl.schema)
        if not keep_idxs:
            continue
        tbl_num = tbl.select(keep_idxs)
        df_rg = tbl_num.to_pandas(ignore_metadata=True, use_threads=True)

        if FORCE_FLOAT32 and not df_rg.empty:
            for c in df_rg.columns:
                if pd.api.types.is_float_dtype(df_rg[c]) and df_rg[c].dtype != np.float32:
                    df_rg[c] = df_rg[c].astype(np.float32, copy=False)
                elif pd.api.types.is_integer_dtype(df_rg[c]):
                    df_rg[c] = df_rg[c].astype(np.float32, copy=False)

        dfs.append(df_rg)
        rows_so_far += len(df_rg)
        if rows_so_far >= sample_rows:
            break

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return df

def common_numeric_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
    c1 = set([c for c in df1.columns if pd.api.types.is_numeric_dtype(df1[c])])
    c2 = set([c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])])
    return sorted(list(c1 & c2))

# =========================
# Calibration
# =========================
def apply_dark(signal: pd.DataFrame, dark: pd.DataFrame) -> pd.DataFrame:
    sig = signal.copy()
    if dark is None or dark.empty:
        return sig
    cols = common_numeric_columns(sig, dark)
    if len(cols) >= 1:
        for c in cols:
            sig[c] = sig[c] - float(np.nanmedian(dark[c].values))
    else:
        sig = sig - float(np.nanmedian(dark.select_dtypes(include=[np.number]).values))
    return sig

def apply_dead(signal: pd.DataFrame, dead: pd.DataFrame) -> pd.DataFrame:
    sig = signal.copy()
    if sig is None or sig.empty or dead is None or dead.empty:
        return sig
    cols = common_numeric_columns(sig, dead)
    if len(cols) >= 1:
        for c in cols:
            if c not in dead.columns:
                continue
            m_aligned = dead[c].reindex(sig.index)
            if pd.api.types.is_bool_dtype(m_aligned):
                mask_bool = m_aligned.fillna(False).to_numpy(dtype=bool)
            else:
                m_num = pd.to_numeric(m_aligned, errors='coerce')
                if m_num.notna().any():
                    mask_bool = (m_num >= 0.5).fillna(False).to_numpy(dtype=bool)
                else:
                    mask_bool = m_aligned.isna().fillna(False).to_numpy(dtype=bool)
            if len(mask_bool) == len(sig):
                sig.loc[mask_bool, c] = np.nan
    else:
        idx_cols = [col for col in dead.columns if 'index' in col.lower() or 'pixel' in col.lower()]
        if idx_cols:
            bad_idx = dead[idx_cols[0]].dropna()
            bad_idx = bad_idx[bad_idx.astype(int).isin(sig.index)].astype(int)
            sig.loc[bad_idx.values, :] = np.nan
    return sig

def apply_flat(signal: pd.DataFrame, flat: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    sig = signal.copy()
    if flat is None or flat.empty:
        return sig
    cols = common_numeric_columns(sig, flat)
    if len(cols) >= 1:
        for c in cols:
            denom = flat[c].values
            scale = np.nanmedian(denom)
            if not np.isfinite(scale) or abs(scale) < eps: scale = 1.0
            sig[c] = sig[c] / float(scale)
    else:
        denom = np.nanmedian(flat.select_dtypes(include=[np.number]).values)
        if not np.isfinite(denom) or abs(denom) < eps: denom = 1.0
        sig = sig / float(denom)
    return sig

def apply_linear_corr(signal: pd.DataFrame, linear_corr: pd.DataFrame) -> pd.DataFrame:
    sig = signal.copy()
    if linear_corr is None or linear_corr.empty:
        return sig
    lc = linear_corr
    if {'slope','intercept'}.issubset(set(lc.columns)):
        a = float(np.nanmedian(lc['slope']))
        b = float(np.nanmedian(lc['intercept']))
        return sig * a + b
    for c in sig.columns:
        c_s = f"{c}_slope"
        c_i = f"{c}_intercept"
        if c_s in lc.columns and c_i in lc.columns:
            a = float(np.nanmedian(lc[c_s])); b = float(np.nanmedian(lc[c_i]))
            sig[c] = sig[c] * a + b
    return sig

def apply_read(signal: pd.DataFrame, read: pd.DataFrame) -> pd.DataFrame:
    sig = signal.copy()
    if read is None or read.empty:
        return sig
    cols = common_numeric_columns(sig, read)
    if len(cols) >= 1:
        for c in cols:
            sig[c] = sig[c] - float(np.nanmedian(read[c]))
    else:
        sig = sig - float(np.nanmedian(read.select_dtypes(include=[np.number]).values))
    return sig

# =========================
# Loading a planet and calibration
# =========================
def load_planet_data(planet_dir: str) -> Dict[str, pd.DataFrame]:
    def maybe(path):
        return read_parquet_safely(path) if os.path.exists(path) else None

    p = Path(planet_dir)
    A_sig = p / "AIRS-CH0_signal_0.parquet"
    F_sig = p / "FGS1_signal_0.parquet"

    A_dark = p / "AIRS-CH0_calibration_0/dark.parquet"
    A_dead = p / "AIRS-CH0_calibration_0/dead.parquet"
    A_flat = p / "AIRS-CH0_calibration_0/flat.parquet"
    A_lcorr= p / "AIRS-CH0_calibration_0/linear_corr.parquet"
    A_read = p / "AIRS-CH0_calibration_0/read.parquet"

    F_dark = p / "FGS1_calibration_0/dark.parquet"
    F_dead = p / "FGS1_calibration_0/dead.parquet"
    F_flat = p / "FGS1_calibration_0/flat.parquet"
    F_lcorr= p / "FGS1_calibration_0/linear_corr.parquet"
    F_read = p / "FGS1_calibration_0/read.parquet"

    A = maybe(str(A_sig)); F = maybe(str(F_sig))
    Ad = maybe(str(A_dark)); Ade = maybe(str(A_dead)); Af = maybe(str(A_flat)); Alc = maybe(str(A_lcorr)); Ar = maybe(str(A_read))
    Fd = maybe(str(F_dark)); Fde = maybe(str(F_dead)); Ff = maybe(str(F_flat)); Flc = maybe(str(F_lcorr)); Fr = maybe(str(F_read))

    def pipe(sig, d, de, f, lc, r):
        if sig is None or sig.empty: return sig
        sig = apply_dark(sig, d)
        sig = apply_dead(sig, de)
        sig = apply_flat(sig, f)
        sig = apply_linear_corr(sig, lc)
        sig = apply_read(sig, r)
        return sig

    A_corr = pipe(A, Ad, Ade, Af, Alc, Ar)
    F_corr = pipe(F, Fd, Fde, Ff, Flc, Fr)

    del Ad, Ade, Af, Alc, Ar, Fd, Fde, Ff, Flc, Fr; gc.collect()
    return {"AIRS": A_corr, "FGS": F_corr}

# =========================
# Sequence builder & features (memory-light)
# =========================
def to_numeric_channels(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    num = df
    if MAX_CHANNELS_PER_STREAM is not None and len(num.columns) > MAX_CHANNELS_PER_STREAM:
        num = num.iloc[:, :MAX_CHANNELS_PER_STREAM].copy()
    num.columns = [f"{prefix}_{c}" for c in num.columns]
    if FORCE_FLOAT32:
        for c in num.columns:
            if num[c].dtype != np.float32:
                num[c] = num[c].astype(np.float32, copy=False)
    return num

def downsample_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == 0:
        return np.zeros((target_len, arr.shape[1] if arr.ndim==2 else 1), dtype=np.float32)
    T = arr.shape[0]
    if T == target_len:
        return arr.astype(np.float32, copy=False)
    x_old = np.linspace(0, 1, T, dtype=np.float32)
    x_new = np.linspace(0, 1, target_len, dtype=np.float32)
    out = np.zeros((target_len, arr.shape[1]), dtype=np.float32)
    for c in range(arr.shape[1]):
        col = arr[:, c]
        mask = np.isfinite(col)
        if mask.sum() < 2:
            out[:, c] = 0.0
            continue
        out[:, c] = np.interp(x_new, x_old[mask], col[mask]).astype(np.float32, copy=False)
    return out

def build_sequence(A_corr: pd.DataFrame, F_corr: pd.DataFrame) -> np.ndarray:
    A_num = to_numeric_channels(A_corr, "A")
    F_num = to_numeric_channels(F_corr, "F")
    if A_num.empty and F_num.empty:
        return np.zeros((SEQ_LENGTH, 1), dtype=np.float32)

    max_len = max(len(A_num) if not A_num.empty else 0, len(F_num) if not F_num.empty else 0)
    if not A_num.empty and len(A_num) < max_len:
        A_num = A_num.reindex(range(max_len))
    if not F_num.empty and len(F_num) < max_len:
        F_num = F_num.reindex(range(max_len))

    X = pd.concat([A_num, F_num], axis=1) if not (A_num.empty and F_num.empty) else pd.DataFrame()
    del A_num, F_num; gc.collect()

    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    arr = X.to_numpy(dtype=np.float32, copy=False)
    del X; gc.collect()

    arr = downsample_to_length(arr, SEQ_LENGTH).astype(np.float32, copy=False)
    return arr

def _rolling_mean_1d(x: np.ndarray, w: int) -> float:
    if len(x) < 1: return 0.0
    k = max(1, w)
    ker = np.ones(k, dtype=np.float32) / k
    mv = np.convolve(x, ker, mode='valid')
    return float(mv.mean()) if mv.size else float(np.nanmean(x))

def _rolling_std_1d(x: np.ndarray, w: int) -> float:
    if len(x) < 1: return 0.0
    return float(np.nanstd(x))

def _rolling_quantile_mean(x: np.ndarray, w: int, q: float) -> float:
    if len(x) < 1: return 0.0
    k = max(1, w)
    step = max(1, k // 2)
    qs = []
    for i in range(0, max(0, len(x) - k + 1), step):
        qs.append(np.nanquantile(x[i:i+k], q))
        if len(qs) >= max(1, len(x) // k): break
    return float(np.mean(qs)) if qs else float(np.nanquantile(x, q))

def rolling_features(x: np.ndarray, windows: List[int]) -> Dict[str, np.ndarray]:
    T, C = x.shape
    feats = {}
    for w in windows:
        means, stds, p10s, p90s = [], [], [], []
        for c in range(C):
            v = x[:, c]
            means.append(_rolling_mean_1d(v, w))
            stds.append(_rolling_std_1d(v, w))
            p10s.append(_rolling_quantile_mean(v, w, 0.10))
            p90s.append(_rolling_quantile_mean(v, w, 0.90))
        feats[f"roll{w}_mean"] = np.array(means, dtype=np.float32)
        feats[f"roll{w}_std"]  = np.array(stds,  dtype=np.float32)
        feats[f"roll{w}_p10"]  = np.array(p10s, dtype=np.float32)
        feats[f"roll{w}_p90"]  = np.array(p90s, dtype=np.float32)
    return feats

def fft_band_features(x: np.ndarray, n_bands: int) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    xz = x - x.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(xz, axis=0)
    mag = np.abs(fft)
    F = mag.shape[0]
    edges = np.linspace(0, F, n_bands+1, dtype=int)
    bands = []
    for i in range(n_bands):
        sl = mag[edges[i]:edges[i+1], :]
        bands.append(sl.mean(axis=0))
    return np.concatenate(bands, axis=0).astype(np.float32, copy=False)

def acf_features(x: np.ndarray, lags: List[int]) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    T, C = x.shape
    xz = x - x.mean(axis=0, keepdims=True)
    xz = xz / (xz.std(axis=0, keepdims=True) + 1e-8)
    feats = []
    for c in range(C):
        v = xz[:, c]
        for L in lags:
            if L >= T:
                feats.append(0.0)
            else:
                feats.append(float(np.dot(v[:-L], v[L:]) / (T - L)))
    return np.array(feats, dtype=np.float32)

def cross_channel_corr(x: np.ndarray, max_lag: int = 16) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    T, C = x.shape
    if C < 2: return np.zeros(0, dtype=np.float32)
    feats = []
    xz = x - x.mean(axis=0, keepdims=True)
    xz = xz / (xz.std(axis=0, keepdims=True) + 1e-8)
    for i in range(C):
        for j in range(i+1, C):
            a, b = xz[:, i], xz[:, j]
            feats.append(float(np.dot(a, b)/T))
            for L in [1,2,4,8,16]:
                if L < T:
                    feats.append(float(np.dot(a[:-L], b[L:])/(T-L)))
                else:
                    feats.append(0.0)
    return np.array(feats, dtype=np.float32)

def build_feature_vector(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    g_mean = x.mean(axis=0).astype(np.float32, copy=False)
    g_std  = x.std(axis=0).astype(np.float32, copy=False)
    g_min  = x.min(axis=0).astype(np.float32, copy=False)
    g_max  = x.max(axis=0).astype(np.float32, copy=False)
    g_q10  = np.quantile(x, 0.10, axis=0).astype(np.float32, copy=False)
    g_q90  = np.quantile(x, 0.90, axis=0).astype(np.float32, copy=False)

    rf = rolling_features(x, WINDOW_SIZES)
    rf_vec = np.concatenate([v for _, v in sorted(rf.items())], axis=0).astype(np.float32, copy=False) if rf else np.zeros(0, dtype=np.float32)
    fft_vec   = fft_band_features(x, FFT_BANDS)
    acf_vec   = acf_features(x, ACF_LAGS)
    xcorr_vec = cross_channel_corr(x)

    out = np.concatenate([g_mean, g_std, g_min, g_max, g_q10, g_q90, rf_vec, fft_vec, acf_vec, xcorr_vec], axis=0)
    return out.astype(np.float32, copy=False)

# =========================
# Caching helpers (FIXED atomic save)
# =========================
def _feat_cache_path(pid: str) -> str:
    return os.path.join(CACHE_DIR, f"{pid}.npy")

def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    """Write to a temp file via file-handle so np.save doesn't append .npy, then atomic replace."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:              # FIX: write via handle -> no .npy auto-append
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)                   # atomic move

def _load_or_build_features_for_pid(pid: str, ydf: pd.DataFrame, wl_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, int]:
    cpath = _feat_cache_path(pid)
    if ENABLE_CACHE and os.path.exists(cpath):
        feat_vec = np.load(cpath, allow_pickle=False)
    else:
        pdir = os.path.join(BASE_DIR, pid)
        data = load_planet_data(pdir)
        seq = build_sequence(data.get("AIRS"), data.get("FGS"))
        del data; gc.collect()
        feat_vec = build_feature_vector(seq)
        del seq; gc.collect()
        if ENABLE_CACHE:
            _atomic_save_npy(cpath, feat_vec)   # FIX: atomic save (no suffix bug)
    row = ydf[ydf['planet_id'] == int(pid)]
    target = row[wl_cols].values.astype(np.float32, copy=False)[0] if not row.empty else None
    return feat_vec, target, int(pid)

# =========================
# Dataset assembly (THREADED)
# =========================
def build_dataset(base_dir: str, train_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    ydf = pd.read_csv(train_csv)
    wl_cols = [c for c in ydf.columns if re.fullmatch(r"wl_\d+", c)]
    labeled_ids = set(map(int, ydf['planet_id'].unique()))
    all_planet_dirs = list_planet_dirs(base_dir)
    planets = [pid for pid in all_planet_dirs if int(pid) in labeled_ids]
    if MAX_PLANETS is not None:
        planets = planets[:int(MAX_PLANETS)]

    if len(planets) == 0:
        print("[build_dataset] No labeled planet folders found under", base_dir)
        return pd.DataFrame(), pd.DataFrame(columns=wl_cols), pd.Series([], name="planet_id", dtype=int)

    print(f"[build_dataset] Will process {len(planets)} labeled planets present on disk.")
    print("[build_dataset] Planet IDs:", ", ".join(planets))

    feats, targets, groups = [], [], []
    total = len(planets)
    done = 0

    N_WORKERS = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_load_or_build_features_for_pid, pid, ydf, wl_cols): pid for pid in planets}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                feat_vec, target, gid = fut.result()
                if target is None:
                    continue
                feats.append(feat_vec)
                targets.append(target)
                groups.append(gid)
            except Exception as e:
                print(f"[build_dataset] WARN: planet {pid} failed: {e}")
            finally:
                done += 1
                if (done == 1) or (done == total) or (done % max(1, total//10) == 0):
                    print(f"[build_dataset] {done}/{total} ({int(done/total*100)}%)")

    if not feats:
        # Graceful empty return (avoids shape mismatch when wl_cols is non-empty)
        return pd.DataFrame(), pd.DataFrame(columns=wl_cols), pd.Series([], name="planet_id", dtype=int)

    X = np.vstack(feats).astype(np.float32, copy=False)
    Y = np.vstack(targets).astype(np.float32, copy=False)
    groups = np.array(groups, dtype=np.int64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32, copy=False)

    pd.DataFrame(Xs).to_parquet(os.path.join(OUT_DIR, "features.parquet"), index=False)
    meta = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "wl_cols": wl_cols,
        "planets": groups.tolist(),
        "feature_dim": int(Xs.shape[1]),
        "speed_preset": SPEED_PRESET
    }
    with open(os.path.join(OUT_DIR, "features_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    return pd.DataFrame(Xs), pd.DataFrame(Y, columns=wl_cols), pd.Series(groups, name="planet_id")

# =========================
# Training — XGBoost baseline (chunked)
# =========================
def train_xgb_chunked(X: pd.DataFrame, Y: pd.DataFrame, groups: pd.Series) -> Dict[str, Any]:
    assert USE_XGB and XGBRegressor is not None, "XGBoost not available. Install xgboost."
    wl_cols = list(Y.columns)
    n_targets = len(wl_cols)
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Decide GPU vs CPU once
    xgb_params_cpu = dict(XGB_PARAMS); xgb_params_cpu["tree_method"] = "hist"; xgb_params_cpu.pop("predictor", None)
    xgb_params_gpu = dict(xgb_params_cpu); xgb_params_gpu["tree_method"] = "gpu_hist"; xgb_params_gpu["predictor"] = "gpu_predictor"
    chosen_params, chosen_device = xgb_params_gpu, "GPU"
    try:
        _probe = XGBRegressor(**chosen_params); _probe.fit(X.iloc[:1], Y.iloc[:1, 0]); del _probe
    except Exception:
        chosen_params, chosen_device = xgb_params_cpu, "CPU"
    print(f"[XGBoost] Using device: {chosen_device}")

    fold_metrics = []
    preds_oof = np.zeros_like(Y.values, dtype=np.float32)
    models = []
    total_models = n_targets * N_FOLDS
    trained_models = 0

    for fold, (tr, va) in enumerate(gkf.split(X, Y, groups)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        Y_tr, Y_va = Y.iloc[tr], Y.iloc[va]
        fold_models = []
        for start in range(0, n_targets, TARGET_CHUNK):
            end = min(start + TARGET_CHUNK, n_targets)
            chunk_models = []
            for t in range(start, end):
                y_tr = Y_tr.iloc[:, t]
                model = XGBRegressor(**chosen_params)
                model.fit(X_tr, y_tr)
                chunk_models.append(model)
                preds_oof[va, t] = model.predict(X_va).astype(np.float32, copy=False)
                trained_models += 1
                if trained_models % 10 == 0 or trained_models == total_models:
                    pct = 100.0 * trained_models / total_models
                    print(f"[Training] Progress: {trained_models}/{total_models} models ({pct:.1f}%)")
            fold_models.append(chunk_models)
        models.append(fold_models)

        mae  = mean_absolute_error(Y_va.values, preds_oof[va])
        rmse = math.sqrt(mean_squared_error(Y_va.values, preds_oof[va]))
        cs = []
        Yv = Y_va.values; Pv = preds_oof[va]
        for i in range(Yv.shape[0]):
            a = Yv[i]; b = Pv[i]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            cs.append(float(np.dot(a, b) / denom))
        fold_metrics.append({"fold": fold, "MAE": mae, "RMSE": rmse, "cosine": float(np.mean(cs))})
        print(f"Fold {fold}: MAE={mae:.6f} RMSE={rmse:.6f} Cosine={np.mean(cs):.6f}")
        del X_tr, X_va, Y_tr, Y_va; gc.collect()

    oof_mae  = mean_absolute_error(Y.values, preds_oof)
    oof_rmse = math.sqrt(mean_squared_error(Y.values, preds_oof))
    pw_mae   = np.mean(np.abs(Y.values - preds_oof), axis=0)
    pw_r2    = [r2_score(Y.iloc[:, j], preds_oof[:, j]) for j in range(n_targets)]

    def total_variation(v): return np.sum(np.abs(np.diff(v)))
    tv_true = np.mean([total_variation(y) for y in Y.values])
    tv_pred = np.mean([total_variation(p) for p in preds_oof])

    metrics = {
        "folds": fold_metrics,
        "OOF_MAE": float(oof_mae),
        "OOF_RMSE": float(oof_rmse),
        "PerWavelength_MAE": {wl_cols[i]: float(pw_mae[i]) for i in range(n_targets)},
        "PerWavelength_R2":  {wl_cols[i]: float(pw_r2[i])  for i in range(n_targets)},
        "TV_true": float(tv_true),
        "TV_pred": float(tv_pred)
    }

    pd.DataFrame(preds_oof, columns=wl_cols).to_parquet(os.path.join(OUT_DIR, "oof_preds.parquet"), index=False)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    return {"models": models, "preds_oof": preds_oof, "metrics": metrics, "wl_cols": wl_cols}

# =========================
# Plotting — quality dashboard
# =========================
def plot_per_wavelength_mae(metrics: Dict[str, Any]):
    wl = list(metrics["PerWavelength_MAE"].keys())
    vals = [metrics["PerWavelength_MAE"][k] for k in wl]
    plt.figure(figsize=(10,4))
    plt.plot(range(len(wl)), vals)
    plt.title("Per-Wavelength MAE")
    plt.xlabel("Wavelength index (wl_1..wl_283)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "per_wavelength_mae.png"), dpi=150)
    plt.show()

def plot_residual_heatmap(Y: pd.DataFrame, preds: np.ndarray, sample_rows: int = 100):
    idx = np.random.RandomState(SEED).choice(len(Y), size=min(sample_rows, len(Y)), replace=False)
    res = (Y.values[idx] - preds[idx])
    plt.figure(figsize=(8,6))
    plt.imshow(res, aspect='auto', interpolation='nearest')
    plt.title("Residual Heatmap (sampled planets × wavelengths)")
    plt.xlabel("Wavelength index")
    plt.ylabel("Sampled planets")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "residual_heatmap.png"), dpi=150)
    plt.show()

# =========================
# Export train-like CSV
# =========================
def export_train_like(planets: List[int], preds: np.ndarray, wl_cols: List[str], path: str):
    df = pd.DataFrame(preds, columns=wl_cols)
    df.insert(0, 'planet_id', planets[:len(preds)])
    df.to_csv(path, index=False)

# =========================
# Inference for a single planet (after training)
# =========================
def predict_for_planet(planet_dir: str, scaler: StandardScaler, models_bundle: Dict[str, Any]) -> np.ndarray:
    data = load_planet_data(planet_dir)
    seq = build_sequence(data.get("AIRS"), data.get("FGS"))
    del data; gc.collect()
    feat = build_feature_vector(seq).reshape(1, -1)
    del seq; gc.collect()

    Xs = scaler.transform(feat)
    wl_cols = models_bundle["wl_cols"]

    fold0 = models_bundle["models"][0]
    preds = []
    for chunk in fold0:
        for model in chunk:
            preds.append(float(model.predict(Xs)[0]))
    return np.array(preds[:len(wl_cols)], dtype=np.float32)

def _json_default(o):
    import numpy as _np
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return o

# =========================
# 🚀 Main
# =========================
if os.path.exists(BASE_DIR) and os.path.exists(TRAIN_CSV):
    if BUILD_DATASET:
        print("[1/4] Building dataset…")
        X, Y, groups = build_dataset(BASE_DIR, TRAIN_CSV)
        print(f"Features: {X.shape}, Targets: {Y.shape}, Planets: {len(groups)}")
    else:
        print("[1/4] Skipping dataset build; loading cached features.parquet …")
        X = pd.read_parquet(os.path.join(OUT_DIR, "features.parquet"))
        ydf = pd.read_csv(TRAIN_CSV)
        wl_cols = [c for c in ydf.columns if re.fullmatch(r"wl_\d+", c)]
        Y = ydf[wl_cols].astype(np.float32)
        groups = pd.Series(ydf['planet_id'], name='planet_id', dtype=np.int64)

    models_bundle = None
    if len(X) == 0 or len(Y) == 0:
        print("No features/targets built — training skipped. Check dataset paths and logs above.")
    elif TRAIN_MODEL and USE_XGB and XGBRegressor is not None:
        print("[2/4] Training XGBoost baseline…")
        models_bundle = train_xgb_chunked(X, Y, groups)
        with open(os.path.join(OUT_DIR, "models_info.json"), "w") as f:
            json.dump({"wl_cols": models_bundle["wl_cols"]}, f, indent=2)
        joblib.dump(models_bundle, os.path.join(OUT_DIR, "models_bundle.pkl"))
        if MAKE_PLOTS:
            print("[3/4] Making quality plots…")
            plot_per_wavelength_mae(models_bundle["metrics"])
            plot_residual_heatmap(Y, models_bundle["preds_oof"], sample_rows=100)

        if EXPORT_PRED:
            print("[4/4] Exporting train-like predictions (OOF)…")
            export_train_like(groups.tolist(), models_bundle["preds_oof"], models_bundle["wl_cols"], os.path.join(OUT_DIR, "train_like_oof.csv"))
            print("Saved:", os.path.join(OUT_DIR, "train_like_oof.csv"))
    else:
        print("Training skipped or prerequisites missing (check BASE_DIR/TRAIN_CSV/XGBoost).")
else:
    print("Paths not found. Please set BASE_DIR and TRAIN_CSV correctly and rerun.")
