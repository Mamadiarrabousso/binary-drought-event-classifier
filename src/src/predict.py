"""
Minimal predictor: load model/scaler/threshold and score a processed station CSV.
"""

import argparse, json
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

def main(input_csv, out_csv):
  rf     = joblib.load("models/model_rf.pkl")
  scaler = joblib.load("models/scaler.pkl")
  cfg    = json.load(open("model_config.json"))
  feats  = cfg["features"]; th = float(cfg["threshold"])

  df = pd.read_csv(input_csv, parse_dates=["Date"])
  good = df[feats].notnull().all(axis=1)
  scored = df.loc[good].copy()

  Xs = scaler.transform(scored[feats].to_numpy())
  proba = rf.predict_proba(Xs)[:,1]
  pred  = (proba >= th).astype(int)

  scored["pred_proba"] = proba
  scored["pred_label"] = pred

  out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
  scored.to_csv(out, index=False)
  print(f"Wrote: {out} ({len(scored)} rows)")

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--input_csv", required=True)
  ap.add_argument("--out_csv", default="results/predictions.csv")
  a = ap.parse_args()
  main(a.input_csv, a.out_csv)
