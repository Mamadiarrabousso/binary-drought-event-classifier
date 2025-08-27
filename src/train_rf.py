"""
Author: Mame Diarra Bousso
Minimal Phase 2: Train class-weighted Random Forest with chronological split.
- Use last 10% of TRAIN as validation to pick a threshold (PR max-F1).
- Evaluate once on held-out test tail.
- Save model, scaler, config, and two figures.
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import joblib

def load_station(processed_dir, name):
  safe = "".join([c if (c.isalnum() or c=="_") else "_" for c in name])
  fp = Path(processed_dir) / f"processed_station_{safe}.csv"
  if not fp.exists(): raise FileNotFoundError(fp)
  return pd.read_csv(fp, parse_dates=["Date"])

def main(processed_dir, out_dir, stations):
  processed_dir = Path(processed_dir)
  out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "models").mkdir(exist_ok=True)
  Path("results/figures").mkdir(parents=True, exist_ok=True)

  # features
  feat_fp = processed_dir / "feature_names.txt"
  feature_names = [x.strip() for x in open(feat_fp).read().splitlines() if x.strip()]

  # combine two source stations
  dfs = [load_station(processed_dir, s) for s in stations]
  df = pd.concat(dfs, ignore_index=True).sort_values("Date").reset_index(drop=True)

  # drop rows with NA in features or target
  df = df.dropna(subset=feature_names + ["Drought_Label"]).copy()
  X = df[feature_names].to_numpy()
  y = df["Drought_Label"].astype(int).to_numpy()

  # chronological split: 80/20
  split = int(0.8 * len(X))
  X_train, X_test = X[:split], X[split:]
  y_train, y_test = y[:split], y[split:]

  # last 10% of TRAIN for threshold selection
  v = max(1, int(0.1 * len(X_train)))
  X_tr, X_val = X_train[:-v], X_train[-v:]
  y_tr, y_val = y_train[:-v], y_train[-v:]

  # scale (kept for consistency)
  scaler = StandardScaler()
  X_tr_s  = scaler.fit_transform(X_tr)
  X_val_s = scaler.transform(X_val)
  X_test_s= scaler.transform(X_test)

  # Random Forest (stable settings from your runs)
  rf = RandomForestClassifier(
      n_estimators=200,
      max_depth=50,
      min_samples_split=5,
      min_samples_leaf=4,
      max_features="sqrt",
      class_weight="balanced_subsample",
      bootstrap=True,
      random_state=42,
  )
  rf.fit(X_tr_s, y_tr)

  # threshold from validation PR curve (max F1)
  val_proba = rf.predict_proba(X_val_s)[:, 1]
  p, r, thr = precision_recall_curve(y_val, val_proba)
  f1s = 2*(p*r)/(p+r+1e-12)
  best = int(np.argmax(f1s))
  # scikit returns thresholds length = len(p)-1; guard index
  th = float(thr[best-1] if best > 0 and best <= len(thr) else 0.5)

  # evaluate on TEST tail
  test_proba = rf.predict_proba(X_test_s)[:, 1]
  test_pred  = (test_proba >= th).astype(int)

  acc = accuracy_score(y_test, test_pred)
  f1  = f1_score(y_test, test_pred)
  auc = roc_auc_score(y_test, test_proba)
  print("=== TEST (held-out) ===")
  print(f"Accuracy: {acc:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")
  print(classification_report(y_test, test_pred, digits=3))

  # save figures
  plt.figure()
  plt.plot(r, p)
  plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Validation PR curve")
  plt.tight_layout()
  plt.savefig("results/figures/phase2_prcurve.png", dpi=150)
  plt.close()

  imp = pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_}) \
          .sort_values("Importance", ascending=False)
  plt.figure(figsize=(7,4))
  plt.barh(imp["Feature"], imp["Importance"])
  plt.title("RF feature importance"); plt.tight_layout()
  plt.savefig("results/figures/phase2_feature_importance.png", dpi=150)
  plt.close()

  
