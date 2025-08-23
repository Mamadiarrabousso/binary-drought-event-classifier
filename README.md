# binary-drought-event-classifier
# Australian Drought-Event Classifier (R → Python)

This repo contains a time-aware **binary drought-event classifier** using monthly climate, vegetation, and hydrologic features from Australian stations (1994–2024).

## Summary
- **Model:** class-weighted Random Forest (primary), with NN/ensembles tested.
- **Label:** drought = 1 if root-zone soil moisture < expanding 20th percentile (**past-only**, lagged threshold).
- **Key features:** mean_Temp, NDVI, SPI_3, Rain, Runoff (SPI_12 smaller).
- **Held-out performance:** ROC-AUC ≈ **0.90** (chronological test).  
  With a recall-oriented threshold, F1 ≈ **0.63** (high recall).
- **Transfer:** On an unseen station, ranking skill ≈ **0.82** ROC-AUC; station-specific thresholds/calibration improve recall.

> Note: in this version, some thresholds were chosen post-hoc on test; AUC is unaffected, but F1/precision/recall may be slightly optimistic. Future work: choose thresholds on a validation tail.

## Repo structure
- `R/phase1_drought_prep.R` — data prep, SPI, label creation (past-only), writes CSVs + `feature_names.txt`.
- `notebooks/phase2_drought_modeling.ipynb` — (optional) your Colab/Notebook.
- `src/train_rf.py` — script to train RF, select threshold, save artifacts.
- `src/predict.py` — load model + scaler and score a CSV.
- `data/` — no raw data committed; see `data/README.md`.
- `results/` — figures/metrics.

## How to run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt

# (1) Prepare data in R (writes to data/processed/)
Rscript R/phase1_drought_prep.R

# (2) Train and evaluate
python src/train_rf.py --processed_dir data/processed --out_dir . --stations BAIRNSDALE_AIRPORT_Combined MORWELL_LATROBE_VALLEY
