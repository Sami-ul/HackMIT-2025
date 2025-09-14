
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed YC-style startup records.

Supports two backends:
  - OpenAI text-embedding-3-large (paid, highest quality)
  - HuggingFace/SentenceTransformers (free, strong): default intfloat/e5-large-v2

It embeds: name + one_liner + long_description (no tags).
It ALSO builds a separate multi-hot tag vector you can weight/concatenate later.

Outputs (in --out_dir):
  - meta.parquet  (id, name, one_liner, long_description, tags, etc.)
  - tag_vocab.json
  - text_embeddings.npy  (float32, shape [N, D])
  - tag_multi_hot.npy    (uint8, shape [N, T])
  - fused.npy            (float32, shape [N, D + T] or [N, D + pca_dim] if --pca-dim)
  - (optional) pca.joblib if --pca-dim provided

Usage:
  python embed_pipeline.py --input ycombinator_startups.json --out_dir ./out --backend hf
  # For OpenAI (requires OPENAI_API_KEY in env):
  python embed_pipeline.py --input ycombinator_startups.json --out_dir ./out --backend openai --model text-embedding-3-large
"""

import argparse
import json
import math
import os
from typing import List, Dict, Any, Tuple

import numpy as np

# Lazy imports for optional deps
def _lazy_import_pandas():
    import pandas as pd
    return pd

def _lazy_import_pyarrow():
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq

def _lazy_import_joblib():
    import joblib
    return joblib

def _lazy_import_openai():
    try:
        from openai import OpenAI
        return OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not found. `pip install openai`") from e

def _lazy_import_st():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers not found. `pip install sentence-transformers`") from e

def _lazy_import_sklearn():
    try:
        from sklearn.decomposition import PCA
        return PCA
    except Exception as e:
        raise RuntimeError("scikit-learn not found. `pip install scikit-learn`") from e


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def read_json_records(path: str) -> List[Dict[str, Any]]:
    """
    Accepts either a JSON list file or a newline-delimited JSON (NDJSON) file.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


def build_text(record: Dict[str, Any], max_chars: int = 4000) -> str:
    name = (record.get("name") or "").strip()
    one = (record.get("one_liner") or "").strip()
    desc = (record.get("long_description") or "").strip()
    blob = f"{name}\n{one}\n{desc}"
    if max_chars:
        blob = blob[:max_chars]
    return blob


def build_tag_vocab(records: List[Dict[str, Any]]) -> List[str]:
    vocab = set()
    for r in records:
        for t in (r.get("tags") or []):
            if t and isinstance(t, str):
                vocab.add(t.strip())
    return sorted(vocab)


def multi_hot_for_record(record: Dict[str, Any], vocab: List[str]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(vocab)}
    vec = np.zeros((len(vocab),), dtype=np.uint8)
    for t in (record.get("tags") or []):
        j = idx.get(t.strip())
        if j is not None:
            vec[j] = 1
    return vec


def embed_texts_hf(texts: List[str], model_name: str) -> np.ndarray:
    ST = _lazy_import_st()
    model = ST(model_name, trust_remote_code=True)
    # SentenceTransformer can normalize for us, but we do it explicitly for clarity
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=False)
    emb = np.asarray(emb, dtype=np.float32)
    emb = l2_normalize(emb)
    return emb


def embed_texts_openai(texts: List[str], model_name: str) -> np.ndarray:
    OpenAI = _lazy_import_openai()
    client = OpenAI()
    out = []
    B = 256  # API allows large batches; adjust if you hit rate limits
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=model_name, input=batch)
        batch_vecs = [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]
        out.append(np.vstack(batch_vecs))
    emb = np.vstack(out)
    emb = l2_normalize(emb)
    return emb


def save_parquet_meta(out_dir: str, records: List[Dict[str, Any]]) -> None:
    pa, pq = _lazy_import_pyarrow()
    pd = _lazy_import_pandas()

    rows = []
    for r in records:
        rows.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "one_liner": r.get("one_liner"),
            "long_description": r.get("long_description"),
            "tags": r.get("tags"),
            "batch": r.get("batch"),
            "status": r.get("status"),
            "stage": r.get("stage"),
            "team_size": r.get("team_size"),
            "regions": r.get("regions"),
            "website": r.get("website"),
            "url": r.get("url"),
        })
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), os.path.join(out_dir, "meta.parquet"))


def save_numpy(path: str, arr: np.ndarray) -> None:
    np.save(path, arr)


def fuse_vectors(text_emb: np.ndarray,
                 tag_multi_hot: np.ndarray,
                 tag_weight: float,
                 pca_dim: int = None,
                 pca_out_path: str = None) -> np.ndarray:
    # Weight tags by scaling to desired L2 magnitude relative to text
    # Compute current magnitudes
    if tag_multi_hot.size == 0 or tag_multi_hot.shape[1] == 0:
        fused = text_emb
    else:
        # Scale tag vector so its average norm ~= tag_weight * avg text norm
        text_norm = np.mean(np.linalg.norm(text_emb, axis=1))
        tag_norm = np.mean(np.linalg.norm(tag_multi_hot, axis=1) + 1e-12)
        scale = (tag_weight * text_norm) / (tag_norm if tag_norm > 0 else 1.0)
        tag_scaled = tag_multi_hot.astype(np.float32) * scale
        fused = np.concatenate([text_emb, tag_scaled], axis=1)

    # Optional PCA (on fused) then re-normalize
    if pca_dim is not None:
        PCA = _lazy_import_sklearn()
        pca = PCA(n_components=pca_dim, random_state=42)
        fused = pca.fit_transform(fused)
        fused = fused.astype(np.float32)
        fused = l2_normalize(fused)
        if pca_out_path:
            joblib = _lazy_import_joblib()
            joblib.dump(pca, pca_out_path)
    else:
        fused = l2_normalize(fused)

    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to YC-style JSON (list or NDJSON)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backend", choices=["hf", "openai"], default="hf")
    ap.add_argument("--model", default="intfloat/e5-large-v2",
                    help="HF model name (for --backend hf) or OpenAI embedding model name (for --backend openai)")
    ap.add_argument("--tag-weight", type=float, default=0.15, help="Relative magnitude of tag vector vs text")
    ap.add_argument("--pca-dim", type=int, default=None, help="Optional PCA dim for fused vectors")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = read_json_records(args.input)

    # Build text corpus
    texts = [build_text(r) for r in records]

    # Build tag vocab and multi-hot
    vocab = build_tag_vocab(records)
    tag_mat = np.vstack([multi_hot_for_record(r, vocab) for r in records]) if vocab else np.zeros((len(records), 0), dtype=np.uint8)

    # Embeddings
    if args.backend == "hf":
        emb = embed_texts_hf(texts, args.model)
    else:
        emb = embed_texts_openai(texts, args.model)

    # Save components
    save_parquet_meta(args.out_dir, records)

    with open(os.path.join(args.out_dir, "tag_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(args.out_dir, "text_embeddings.npy"), emb)
    np.save(os.path.join(args.out_dir, "tag_multi_hot.npy"), tag_mat)

    # Fuse & save
    pca_path = os.path.join(args.out_dir, "pca.joblib") if args.pca_dim else None
    fused = fuse_vectors(emb, tag_mat, tag_weight=args.tag_weight, pca_dim=args.pca_dim, pca_out_path=pca_path)
    np.save(os.path.join(args.out_dir, "fused.npy"), fused)

    print(f"Done. N={len(records)}, D_text={emb.shape[1]}, T_tags={tag_mat.shape[1]}, D_fused={fused.shape[1]}")
    print(f"Artifacts in: {args.out_dir}")

if __name__ == "__main__":
    main()
