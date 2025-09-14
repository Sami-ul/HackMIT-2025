#!/usr/bin/env python3
import numpy as np, pandas as pd, sys
out = sys.argv[1] if len(sys.argv) > 1 else "out_full"
E = np.load(f"{out}/text_embeddings.npy")
F = np.load(f"{out}/fused.npy")
T = np.load(f"{out}/tag_multi_hot.npy")
M = pd.read_parquet(f"{out}/meta.parquet")
print("text_embeddings:", E.shape)
mean_L2 = float(((F * F).sum(axis=1) ** 0.5).mean())
print("fused:", F.shape, "mean_L2=", mean_L2)
print("tag_multi_hot:", T.shape, "nonzero=", int(T.sum()))
print("meta rows:", M.shape, "cols:", list(M.columns))