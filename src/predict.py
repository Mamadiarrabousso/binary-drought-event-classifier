"""
Score a processed station CSV using the trained model.

Usage:
  python src/predict.py --csv data/processed/processed_station_<NAME>.csv --out preds_<NAME>.csv
"""
import argparse, json, joblib, pandas as pd, numpy as np

def main(args):
    model = joblib.load("models/best_drought_model.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    with open("model_config.json") as f:
        cfg = json.load(f)
    feats = cfg["features"]; t_star = float(cfg["optimal_threshold"])

    df = pd.read_csv(args.csv, parse_dates=["Date"])
    X = df[feats].dropna().values
    idx = df[feats].dropna().index

    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)[:,1]
    yhat = (p >= t_star).astype(int)

    out = df.loc[idx, ["Date","Station"]].copy()
    out["proba_drought"] = p
    out["pred_drought"]  = yhat
    out.to_csv(args.out, index=False)
    print(f"Saved predictions → {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Processed station CSV to score.")
    ap.add_argument("--out", default="predictions.csv")
    main(ap.parse_args())

