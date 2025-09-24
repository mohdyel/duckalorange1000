# ==========================================================
# Standalone Inference — μ (wl_1.. or mu_.. ) + σ (sigma_..)
# Needs only: models_bundle.pkl + features_meta.json
# Auto-matches training feature profile ("fast"/"balanced"/"max")
# ==========================================================
import os, gc, json, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# -------- CONFIG (edit these) --------
DATA_ROOT   = "/kaggle/input/ariel-data-challenge-2025"
TEST_ROOT   = f"{DATA_ROOT}/test"
BUNDLE_PKL  = "/kaggle/input/duckalorange10001/duckalorange1000/models_bundle.pkl"
META_JSON   = "/kaggle/input/duckalorange10001/duckalorange1000/features_meta.json"
SUB_CSV     = "submission.csv"
# ------------------------------------

# ---- MUST match the training run that produced the artifacts ----
SEED                    = 42
SEQ_LENGTH              = 2048
MAX_ROWS_PER_RG_SAMPLE  = 80_000
FFT_BANDS               = 16
ACF_LAGS                = [1,2,4,8,16,32,64,128]
WINDOW_SIZES            = [16, 64, 256, 1024]
MAX_CHANNELS_PER_STREAM = 8
FORCE_FLOAT32           = True

# ---------------- Parquet reading ----------------
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PA_AVAILABLE = True
except Exception:
    PA_AVAILABLE = False
    pa = None
    pq = None

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

    pf = pq.ParquetFile(path)
    dfs, rows = [], 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        keep = _arrow_numeric_col_indices(tbl.schema)
        if not keep:
            continue
        df_rg = tbl.select(keep).to_pandas(ignore_metadata=True)
        if FORCE_FLOAT32 and not df_rg.empty:
            for c in df_rg.columns:
                if pd.api.types.is_float_dtype(df_rg[c]) and df_rg[c].dtype != np.float32:
                    df_rg[c] = df_rg[c].astype(np.float32, copy=False)
                elif pd.api.types.is_integer_dtype(df_rg[c]):
                    df_rg[c] = df_rg[c].astype(np.float32, copy=False)
        dfs.append(df_rg)
        rows += len(df_rg)
        if rows >= sample_rows:
            break
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ---------------- Calibration helpers ----------------
def common_numeric_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
    c1 = set([c for c in df1.columns if pd.api.types.is_numeric_dtype(df1[c])])
    c2 = set([c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])])
    return sorted(list(c1 & c2))

def apply_dark(signal: pd.DataFrame, dark: pd.DataFrame) -> pd.DataFrame:
    sig = signal.copy()
    if dark is None or dark.empty: return sig
    cols = common_numeric_columns(sig, dark)
    if cols:
        for c in cols: sig[c] = sig[c] - float(np.nanmedian(dark[c].values))
    else:
        sig = sig - float(np.nanmedian(dark.select_dtypes(include=[np.number]).values))
    return sig

def apply_dead(signal: pd.DataFrame, dead: pd.DataFrame) -> pd.DataFrame:
    sig = signal.copy()
    if sig is None or sig.empty or dead is None or dead.empty: return sig
    cols = common_numeric_columns(sig, dead)
    if cols:
        for c in cols:
            if c not in dead.columns: continue
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
    if flat is None or flat.empty: return sig
    cols = common_numeric_columns(sig, flat)
    if cols:
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
    if linear_corr is None or linear_corr.empty: return sig
    lc = linear_corr
    if {'slope','intercept'}.issubset(set(lc.columns)):
        a = float(np.nanmedian(lc['slope'])); b = float(np.nanmedian(lc['intercept']))
        return sig * a + b
    for c in sig.columns:
        c_s = f"{c}_slope"; c_i = f"{c}_intercept"
        if c_s in lc.columns and c_i in lc.columns:
            a = float(np.nanmedian(lc[c_s])); b = float(np.nanmedian(lc[c_i]))
            sig[c] = sig[c] * a + b
    return sig

def apply_read(signal: pd.DataFrame, read: pd.DataFrame) -> pd.DataFrame:
    sig = signal.copy()
    if read is None or read.empty: return sig
    cols = common_numeric_columns(sig, read)
    if cols:
        for c in cols: sig[c] = sig[c] - float(np.nanmedian(read[c]))
    else:
        sig = sig - float(np.nanmedian(read.select_dtypes(include=[np.number]).values))
    return sig

# ---------------- Load + calibrate a planet ----------------
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

# ---------------- Feature-profile helpers ----------------
def _apply_speed_preset(preset: str):
    """Match the training feature profile (as used in the trainer)."""
    global MAX_CHANNELS_PER_STREAM, FFT_BANDS, ACF_LAGS, WINDOW_SIZES
    if preset == "fast":
        MAX_CHANNELS_PER_STREAM = 4
        FFT_BANDS = 8
        ACF_LAGS = [1, 4, 16]
        WINDOW_SIZES = [32, 256]
    elif preset == "balanced":
        MAX_CHANNELS_PER_STREAM = 8
        FFT_BANDS = 12
        ACF_LAGS = [1, 2, 4, 8, 16, 32]
        WINDOW_SIZES = [16, 64, 256, 1024]
    elif preset == "max":
        pass
    else:
        pass

# ---------------- Sequence + features ----------------
def to_numeric_channels(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
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

def rolling_features(x: np.ndarray, windows: List[int]) -> Dict[str, np.ndarray]:
    T, C = x.shape
    feats = {}
    for w in windows:
        means, stds, p10s, p90s = [], [], [], []
        for c in range(C):
            v = x[:, c]
            means.append(float(np.convolve(v, np.ones(max(1,w), dtype=np.float32)/max(1,w), mode='valid').mean()) if len(v)>=w else float(np.nanmean(v)))
            stds.append(float(np.nanstd(v)))
            # quantile-of-means approximation:
            k = max(1, w); step = max(1, k // 2); qs10 = []; qs90 = []
            for i in range(0, max(0, len(v) - k + 1), step):
                qs10.append(np.nanquantile(v[i:i+k], 0.10))
                qs90.append(np.nanquantile(v[i:i+k], 0.90))
                if len(qs10) >= max(1, len(v)//k): break
            p10 = float(np.mean(qs10)) if qs10 else float(np.nanquantile(v, 0.10))
            p90 = float(np.mean(qs90)) if qs90 else float(np.nanquantile(v, 0.90))
            p10s.append(p10); p90s.append(p90)
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

def cross_channel_corr(x: np.ndarray, lags=(0,1,2,4,8,16)) -> np.ndarray:
    T, C = x.shape
    if C < 2: return np.zeros(0, dtype=np.float32)
    xz = x - x.mean(axis=0, keepdims=True)
    xz = xz / (x.std(axis=0, keepdims=True) + 1e-8)
    feats = []
    for i in range(C):
        for j in range(i+1, C):
            a, b = xz[:, i], xz[:, j]
            feats.append(float(np.dot(a, b)/T))
            for L in lags[1:]:
                if L < T:
                    feats.append(float(np.dot(a[:-L], b[L:])/(T-L)))
                else:
                    feats.append(0.0)
    return np.array(feats, dtype=np.float32)

def build_feature_vector(x: np.ndarray) -> np.ndarray:
    g_mean = x.mean(0).astype(np.float32, copy=False)
    g_std  = x.std(0).astype(np.float32, copy=False)
    g_min  = x.min(0).astype(np.float32, copy=False)
    g_max  = x.max(0).astype(np.float32, copy=False)
    g_q10  = np.quantile(x, 0.10, axis=0).astype(np.float32, copy=False)
    g_q90  = np.quantile(x, 0.90, axis=0).astype(np.float32, copy=False)
    rf  = rolling_features(x, WINDOW_SIZES)
    rf_vec = np.concatenate([v for _, v in sorted(rf.items())], axis=0).astype(np.float32, copy=False) if rf else np.zeros(0, np.float32)
    fft = fft_band_features(x, FFT_BANDS)
    acf = acf_features(x, ACF_LAGS)
    xcr = cross_channel_corr(x)
    out = np.concatenate([g_mean, g_std, g_min, g_max, g_q10, g_q90, rf_vec, fft, acf, xcr], 0)
    return out.astype(np.float32, copy=False)

# ---------------- Load artifacts + scaler ----------------
models_bundle = joblib.load(BUNDLE_PKL)
with open(META_JSON, "r") as f:
    meta = json.load(f)

wl_cols      = models_bundle.get("wl_cols", meta.get("wl_cols"))
planets_list = meta.get("planets", None)

scaler = StandardScaler()
scaler.mean_  = np.array(meta["scaler_mean"],  dtype=np.float32)
scaler.scale_ = np.array(meta["scaler_scale"], dtype=np.float32)
target_dim    = int(scaler.mean_.shape[0])

# ---------------- Build standardized features (auto-match profile) ----------------
def _try_build_with_profile(preset: str, planet_dir: str) -> np.ndarray:
    _apply_speed_preset(preset)
    data = load_planet_data(planet_dir)
    seq  = build_sequence(data.get("AIRS"), data.get("FGS"))
    feat = build_feature_vector(seq).reshape(1, -1)
    del data, seq; gc.collect()
    return feat

# ---------------- Fold prediction helpers ----------------
def predict_with_fold_models(Xs, models_bundle, fold_idx):
    preds = []
    for chunk in models_bundle["models"][fold_idx]:
        for model in chunk:
            preds.append(float(model.predict(Xs)[0]))
    return np.array(preds[:len(wl_cols)], dtype=np.float32)

def gather_fold_preds(Xs, models_bundle) -> np.ndarray:
    fold_preds = []
    for f in range(len(models_bundle["models"])):
        fold_preds.append(predict_with_fold_models(Xs, models_bundle, f))  # (F, 283)
    return np.stack(fold_preds, axis=0)  # (F, 283)

# ---------------- σ helpers ----------------
def smooth_ma(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1: return x.astype(np.float32, copy=False)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.convolve(xp, np.ones(k, dtype=np.float32) / k, mode="valid")
    return out.astype(np.float32, copy=False)

# ---------------- Submission layout helpers ----------------
def resolve_submission_layout(sample_df: pd.DataFrame, n_mu_expected: int) -> Tuple[List[str], List[str]]:
    cols = list(sample_df.columns)
    mu_cols = [c for c in cols if c.startswith("mu_") or c.startswith("wl_")]
    sigma_cols = [c for c in cols if c.startswith("sigma_")]
    if mu_cols and len(sigma_cols) == len(mu_cols):
        return mu_cols, sigma_cols
    # Fallback: split equally
    if len(cols) % 2 == 0:
        half = len(cols) // 2
        return cols[:half], cols[half:]
    # Final fallback
    return cols[:n_mu_expected], cols[n_mu_expected:n_mu_expected*2]

# ---------------- TEST SWEEP → submission.csv ----------------
sub_template = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv", index_col="planet_id")
planet_ids = sub_template.index.tolist()

# Safer defaults (do NOT set a giant global sigma floor)
alpha     = float(meta.get("sigma_scale", 1.0))
sigma_min = float(meta.get("sigma_min", 1e-15))   # evaluator min; per-channel floors are applied below
sigma_max = float(meta.get("sigma_max", 0.1))     # generous but not destructive
SIGMA_ALPHA = float(meta.get("sigma_alpha", 1.5)) # gentle global relaxer if fold spread is small

# Instrument-aware floors used by the evaluator’s “true noise” baseline
FSG_SIGMA_TRUE  = 1e-6   # for wl_1 (FGS1)
AIRS_SIGMA_TRUE = 1e-5   # for wl_2..wl_283 (AIRS)

mu_cols, sigma_cols = resolve_submission_layout(sub_template, n_mu_expected=len(wl_cols))

rows = []
for pid in planet_ids:
    planet_dir = f"{TEST_ROOT}/{int(pid)}"

    # auto-match training feature profile
    feat = None
    preset_candidates = []
    if meta.get("speed_preset"):
        preset_candidates.append(meta["speed_preset"])
    preset_candidates += ["fast", "balanced", "max"]

    for preset in preset_candidates:
        if preset is None:
            continue
        try:
            feat_try = _try_build_with_profile(preset, planet_dir)
            if feat_try.shape[1] == target_dim:
                feat = feat_try
                break
        except Exception:
            continue

    if feat is None:
        feat = _try_build_with_profile("max", planet_dir)
        raise RuntimeError(
            f"[{pid}] Could not match feature_dim={target_dim}. "
            f"Tried presets {preset_candidates}. Got {feat.shape[1]} with 'max'. "
            "Ensure inference constants match training."
        )

    Xs = scaler.transform(feat)
    all_preds = gather_fold_preds(Xs, models_bundle)        # (F, 283)
    mu_preds  = all_preds.mean(axis=0).astype(np.float32)   # (283,)
    raw_sigma = all_preds.std(axis=0).astype(np.float32)    # (283,)

    # Start from fold-spread; clamp, smooth
    sigma = smooth_ma(np.clip(alpha * raw_sigma, sigma_min, sigma_max), k=5)

    # ---- Metric-aware safety patch (avoid 0.00) ----
    mu_preds = np.maximum(mu_preds, 0.0)                    # evaluator forbids negatives
    sigma *= SIGMA_ALPHA                                    # gentle global relaxer (e.g., 1.5)

    # Per-channel floors aligned with evaluator's "true noise"
    if sigma.shape[0] >= 1:
        sigma[0]  = max(sigma[0],  FSG_SIGMA_TRUE)          # wl_1 (FGS)
    if sigma.shape[0] >= 2:
        sigma[1:] = np.maximum(sigma[1:], AIRS_SIGMA_TRUE)  # wl_2.. (AIRS)

    # Final guard: evaluator absolute bounds
    sigma = np.clip(sigma, 1e-15, 0.1)

    # Align to template length if needed (trim/pad — should normally match)
    m_target = len(mu_cols)
    if len(mu_preds) != m_target:
        if len(mu_preds) > m_target:
            mu_preds = mu_preds[:m_target]
            sigma   = sigma[:m_target]
        else:
            mu_preds = np.pad(mu_preds, (0, m_target - len(mu_preds)), mode="edge")
            sigma   = np.pad(sigma,   (0, m_target - len(sigma)),   mode="edge")

    row = pd.Series(index=sub_template.columns, dtype=np.float32)
    row.loc[mu_cols]    = mu_preds
    row.loc[sigma_cols] = sigma
    rows.append((int(pid), row))

# Build final submission DataFrame in the same index/column order
submission_df = pd.DataFrame({pid: r for pid, r in rows}).T
submission_df.index.name = "planet_id"
submission_df = submission_df.reindex(index=sub_template.index, columns=sub_template.columns)

# Quick sanity log (optional, one line)
print("μ[min,med,max] σ[min,med,max]:",
      float(submission_df.iloc[:, :submission_df.shape[1]//2].min().min()),
      float(np.median(submission_df.iloc[:, :submission_df.shape[1]//2].values)),
      float(submission_df.iloc[:, :submission_df.shape[1]//2].max().max()),
      float(submission_df.iloc[:, submission_df.shape[1]//2:].min().min()),
      float(np.median(submission_df.iloc[:, submission_df.shape[1]//2:].values)),
      float(submission_df.iloc[:, submission_df.shape[1]//2:].max().max())
)

submission_df.to_csv(SUB_CSV)
print(f"✅ Wrote {SUB_CSV} with {submission_df.shape[0]} planets × {submission_df.shape[1]} columns.")