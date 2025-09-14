#!/usr/bin/env python3
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.utils import check_random_state
import umap
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out_full")
    ap.add_argument("--n-neighbors", type=int, default=50)
    ap.add_argument("--min-dist", type=float, default=0.05)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fused = np.load(os.path.join(args.out_dir, "fused.npy"))
    meta = pd.read_parquet(os.path.join(args.out_dir, "meta.parquet"))

    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                        metric=args.metric, random_state=check_random_state(args.seed))
    xy = reducer.fit_transform(fused).astype("float32")
    joblib.dump(reducer, os.path.join(args.out_dir, "umap_model.joblib"))

    points = pd.DataFrame({
        "id": meta["id"].astype("int64", errors="ignore"),
        "name": meta["name"].astype("string"),
        "x": xy[:,0],
        "y": xy[:,1],
        "tags": meta["tags"].astype("object"),
        "status": meta["status"].fillna("Unknown").astype("string") if "status" in meta.columns else "Unknown",
        "stage": meta["stage"].fillna("Unknown").astype("string") if "stage" in meta.columns else "Unknown",
        "one_liner": meta["one_liner"].fillna("").astype("string") if "one_liner" in meta.columns else "",
        "long_description": meta["long_description"].fillna("").astype("string") if "long_description" in meta.columns else "",
        "batch": meta["batch"].fillna("").astype("string") if "batch" in meta.columns else "",
        "logo_url": meta["small_logo_thumb_url"].fillna("").astype("string") if "small_logo_thumb_url" in meta.columns else "",
        "team_size": meta["team_size"].astype("float64") if "team_size" in meta.columns else None
    })
    points.to_parquet(os.path.join(args.out_dir, "points.parquet"))
    points.to_json(os.path.join(args.out_dir, "points.json"), orient="records")

    print("Wrote:", os.path.join(args.out_dir, "points.parquet"), "and points.json")

if __name__ == "__main__":
    main()