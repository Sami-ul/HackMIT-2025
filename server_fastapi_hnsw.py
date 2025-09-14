# server_fastapi_hnsw.py
import os, json, numpy as np, pandas as pd, hnswlib, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OUT_DIR = os.getenv("OUT_DIR", "out_full")
BACKEND = os.getenv("BACKEND", "hf")             # "hf" or "openai"
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-large-v2")  # or "text-embedding-3-large"
TAG_WEIGHT = float(os.getenv("TAG_WEIGHT", "0.15"))

POINTS_PATH = os.path.join(OUT_DIR, "points.json")
INDEX_PATH  = os.path.join(OUT_DIR, "index_hnsw.bin")
FUSED_PATH  = os.path.join(OUT_DIR, "fused.npy")
META_PATH   = os.path.join(OUT_DIR, "meta.parquet")
VOCAB_PATH  = os.path.join(OUT_DIR, "tag_vocab.json")
TAGMAT_PATH = os.path.join(OUT_DIR, "tag_multi_hot.npy")
PCA_PATH    = os.path.join(OUT_DIR, "pca.joblib")          # may not exist
UMAP_PATH   = os.path.join(OUT_DIR, "umap_model.joblib")   # created by layout_umap.py

app = FastAPI(title="Startup Map API (HNSW)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- lazy singletons ----
_points = _vecs = _meta = _index = _vocab = _reducer = _pca = None
_tag_scale_const = 0.0
_st_model = None
_oa_client = None

def _ensure_loaded():
    global _points, _vecs, _meta, _index, _vocab, _reducer, _pca, _tag_scale_const, _st_model, _oa_client
    if _points is None:
        with open(POINTS_PATH, "r", encoding="utf-8") as f:
            _points = json.load(f)
    if _vecs is None:
        _vecs = np.load(FUSED_PATH).astype("float32")
        _vecs /= (np.linalg.norm(_vecs, axis=1, keepdims=True) + 1e-12)
    if _meta is None:
        _meta = pd.read_parquet(META_PATH)
    if _index is None:
        d = _vecs.shape[1]
        idx = hnswlib.Index(space='cosine', dim=d)
        idx.load_index(INDEX_PATH)
        idx.set_ef(96)
        _index = idx
    if _vocab is None:
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            _vocab = json.load(f)
    if _reducer is None and os.path.exists(UMAP_PATH):
        _reducer = joblib.load(UMAP_PATH)
    if _pca is None and os.path.exists(PCA_PATH):
        _pca = joblib.load(PCA_PATH)
    # precompute tag scaling constant to match training fuse logic
    global _tag_scale_const
    if _tag_scale_const == 0.0 and os.path.exists(TAGMAT_PATH):
        tag_mat = np.load(TAGMAT_PATH).astype("float32")
        avg_tag_norm = float(np.mean(np.linalg.norm(tag_mat, axis=1)))
        _tag_scale_const = (TAG_WEIGHT * 1.0) / (avg_tag_norm + 1e-12)  # text norm ~1
    # embedding backends
    if BACKEND == "hf" and _st_model is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        _st_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    if BACKEND == "openai" and _oa_client is None:
        if OpenAI is None:
            raise RuntimeError("openai not installed")
        _oa_client = OpenAI()

def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def _embed_text(s: str) -> np.ndarray:
    if BACKEND == "hf":
        v = _st_model.encode([s], normalize_embeddings=False)
        v = np.asarray(v, dtype=np.float32)
        return _l2norm(v)[0]
    elif BACKEND == "openai":
        resp = _oa_client.embeddings.create(model=EMBED_MODEL, input=[s])
        v = np.asarray([resp.data[0].embedding], dtype=np.float32)
        return _l2norm(v)[0]
    else:
        raise HTTPException(500, f"Unknown BACKEND={BACKEND}")

def _build_text(name: str, one_liner: str, long_description: str, max_chars=4000) -> str:
    blob = f"{name or ''}\n{one_liner or ''}\n{long_description or ''}"
    return blob[:max_chars]

def _tags_to_vec(tags) -> np.ndarray:
    idx = {t:i for i,t in enumerate(_vocab)}
    vec = np.zeros((len(_vocab),), dtype=np.float32)
    for t in (tags or []):
        j = idx.get(t)
        if j is not None:
            vec[j] = 1.0
    return vec

def _fuse_and_project(text_vec: np.ndarray, tag_vec: np.ndarray) -> np.ndarray:
    # scale tags to match training average magnitude
    tag_scaled = tag_vec * _tag_scale_const
    fused = np.concatenate([text_vec.astype("float32"), tag_scaled.astype("float32")], axis=0)[None, :]
    if _pca is not None:
        fused = _pca.transform(fused).astype("float32")
    fused = _l2norm(fused)[0]
    return fused

class LocateIn(BaseModel):
    name: str
    one_liner: str | None = ""
    long_description: str | None = ""
    tags: list[str] | None = []

@app.get("/points")
def points():
    _ensure_loaded()
    return JSONResponse(_points)

@app.get("/neighbors")
def neighbors(id: int, k: int = 10):
    _ensure_loaded()
    if id < 0 or id >= _vecs.shape[0]:
        raise HTTPException(400, "id out of range")
    labels, distances = _index.knn_query(_vecs[id:id+1], k=k)
    out = [{"row": int(i), "score": float(1.0 - d), "name": str(_meta.iloc[int(i)]['name'])}
           for i, d in zip(labels[0], distances[0])]
    return JSONResponse(out)

@app.post("/locate")
def locate(payload: LocateIn):
    _ensure_loaded()
    text = _build_text(payload.name, payload.one_liner or "", payload.long_description or "")
    tvec = _embed_text(text)
    tag_vec = _tags_to_vec(payload.tags)
    z = _fuse_and_project(tvec, tag_vec)      # fused (and PCA if fitted), L2-normalized
    # 2D position via UMAP.transform (falls back to [0,0] if no reducer)
    if _reducer is not None:
        xy = _reducer.transform(z[None, :]).astype("float32")[0].tolist()
    else:
        xy = [0.0, 0.0]
    # neighbors from ANN
    labels, distances = _index.knn_query(z[None, :], k=8)
    neigh = [{"row": int(i), "score": float(1.0 - d), "name": str(_meta.iloc[int(i)]['name'])}
             for i, d in zip(labels[0], distances[0])]
    return JSONResponse({"x": xy[0], "y": xy[1], "neighbors": neigh})
