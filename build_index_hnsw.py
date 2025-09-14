#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd, hnswlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out_full")
    ap.add_argument("--index_path", default="index_hnsw.bin")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efC", type=int, default=200)
    ap.add_argument("--efS", type=int, default=96)
    ap.add_argument("--query_row", type=int, default=None)
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    vecs = np.load(os.path.join(args.out_dir, "fused.npy")).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms

    d = vecs.shape[1]
    index = hnswlib.Index(space='cosine', dim=d)
    index.init_index(max_elements=vecs.shape[0], ef_construction=args.efC, M=args.M)
    index.add_items(vecs, ids=np.arange(vecs.shape[0]))
    index.set_ef(args.efS)
    index.save_index(os.path.join(args.out_dir, args.index_path))
    print("Index saved to", os.path.join(args.out_dir, args.index_path), "ntotal=", vecs.shape[0])

    if args.query_row is not None:
        labels, distances = index.knn_query(vecs[args.query_row:args.query_row+1], k=args.k)
        meta = pd.read_parquet(os.path.join(args.out_dir, "meta.parquet"))
        print("Neighbors for", meta.iloc[int(args.query_row)]["name"], "=>")
        for rank,(i,dist) in enumerate(zip(labels[0], distances[0])):
            print(f"  {rank:>2} | sim={1.0 - float(dist): .3f} | {int(i)} | {meta.iloc[int(i)]['name']}")

if __name__ == "__main__":
    main()
