# Ariel Data Challenge – Ensemble Inference

## Overview
This repository contains:
- **`models_bundle.pkl`** → Trained ensemble model bundle.  
- **Testing scripts** → Multiple inference codes (e.g., `test_specific_folder_NoSigma.py`) for running predictions under different settings.  
- **Datasets**:  
  - `train.csv` → Original training data.  
  - `train_like_oof.csv` → Training-like OOF (out-of-fold) dataset, slightly different from `train.csv`.  
  - `untrained_planet_ids.csv` → List of 100 planet IDs that were **not trained or validated**.  

Applying ensemble methods significantly improves accuracy:  
- Predictions from `train_like_oof.csv` alone differ from `train.csv`.  
- Once ensembled, results align much closer to real distribution, boosting accuracy.

## Usage
1. **Models**  
   Ensure `models_bundle.pkl` is available in your working directory.  

2. **Testing**  
   - Use the provided test codes depending on the evaluation mode.  
   - Example:  
     
     test_specific_folder_NoSigma.py 
    
   - Works especially well for **untrained/novalidated IDs** listed in `untrained_planet_ids.csv`.

3. **Untrained Planets**  
   Most of the 100 untrained planet IDs were tested manually with promising results using `test_specific_folder_NoSigma.py`.  

## Notes
- Ensemble methods close the gap between `train_like_oof.csv` and real-world performance.  
- Accuracy improvements are consistent across tested untrained IDs.  

## Important
I’m not sure about **sigma** – I think it might be the reason why I’m getting `0` on the Kaggle public score.
