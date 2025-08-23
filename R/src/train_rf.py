"""
Train a Random Forest drought-event classifier on processed station CSVs.

Usage:
  python src/train_rf.py \
    --processed_dir data/processed \
    --out_dir . \
    --stations BAIRNSDALE_AIRPORT_Combined MORWELL_LATROBE_VALLEY
"""
import argparse, glob, json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, f1_score, classification_report)
import joblib
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

def load_processed(processed_dir, stations):
    files = glob.glob(os.path.join(processed_dir, "processed_station_*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["Date"])
        if stations and df["Station"].iloc[0] not in stations:
            continue
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No processed station files found for given stations.")
    data = pd.concat(dfs, ignore_index=True).sort_values(["Station","Date"]).reset_index(drop=True)
    return data

def main(args):
    # Load
    data = load_processed(args.processed_dir, args.stations)
    with open("feature_names.txt") as f:
        feature_names = [ln.strip() for ln in f if ln.strip()]

    # Keep only rows with complete features + label
    used = data[feature_names + ["Drought_Label","Date","Station"]].dropna().copy()
    X = used[feature_names].values
    y = used["Drought_Label"].astype(int).values
    dates = used["Date"].values

    # Chronological split (overall). Optionally: per-station split; keep simple here.
    split_idx = int(0.8 * len(used))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}  Test period: {dates_test.min()} → {dates_test.max()}")

    # Scale (RF doesn't need it, but it won't hurt; helps if you try SVC/NN later)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Time-series CV hyperparameter search (no leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [None, 12, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt","log2"],
        "class_weight": ["balanced","balanced_subsample"]
    }
    base = RandomForestClassifier(random_state=SEED)
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=40,
        cv=tscv, scoring="average_precision", n_jobs=-1, random_state=SEED, verbose=1
    )
    search.fit(X_train_s, y_train)
    best_rf = search.best_estimator_
    print("Best params:", search.best_params_)

    # Threshold from validation tail of TRAIN
    cut = int(len(X_train_s) * 0.8)
    X_tr, y_tr = X_train_s[:cut], y_train[:cut]
    X_val, y_val = X_train_s[cut:], y_train[cut:]

    best_rf.fit(X_tr, y_tr)
    val_p = best_rf.predict_proba(X_val)[:,1]
    prec, rec, thr = precision_recall_curve(y_val, val_p)
    f1s = 2 * prec[1:] * rec[1:] / (prec[1:] + rec[1:] + 1e-12)
    t_star = float(thr[np.argmax(f1s)])
    print(f"Chosen threshold on VAL: {t_star:.3f}")

    # Final test
    test_p = best_rf.predict_proba(X_test_s)[:,1]
    test_pred = (test_p >= t_star).astype(int)
    roc = roc_auc_score(y_test, test_p)
    pr  = average_precision_score(y_test, test_p)
    f1  = f1_score(y_test, test_pred)
    print(f"TEST ROC-AUC: {roc:.3f} | PR-AUC: {pr:.3f} | F1: {f1:.3f}")
    print(classification_report(y_test, test_pred, digits=3))

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(best_rf, os.path.join("models","best_drought_model.pkl"))
    joblib.dump(scaler, os.path.join("models","feature_scaler.pkl"))
    with open("model_config.json","w") as f:
        json.dump({
            "optimal_threshold": t_star,
            "features": feature_names,
            "best_params": search.best_params_,
            "metrics": {"roc_auc": float(roc), "pr_auc": float(pr), "f1": float(f1)}
        }, f, indent=2)

    # Feature importance plot
    fi = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    ax = fi.plot(kind="barh", figsize=(8,6))
    ax.invert_yaxis(); ax.set_title("Random Forest Feature Importance"); plt.tight_layout()
    plt.savefig(os.path.join("results","feature_importance.png"), dpi=150)
    fi.to_csv(os.path.join("results","feature_importance.csv"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--out_dir", type=str, default=".")
    p.add_argument("--stations", nargs="*", default=[],
                   help="Optional: station names to include (match the 'Station' column).")
    args = p.parse_args()
    main(args)

