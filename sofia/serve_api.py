"""
serve_api.py
FastAPI server exposing /health and /embed for SOFIA.
OpenAPI: /docs  |  ReDoc: /redoc
"""
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os

MODEL = os.getenv("SOFIA_MODEL", "./SOFIA")
model = SentenceTransformer(MODEL)
app = FastAPI(title="SOFIA Embeddings API", version="1.0.0")

class Req(BaseModel):
    texts: list[str]

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "dim": model.get_sentence_embedding_dimension()}

@app.post("/embed")
def embed(r: Req):
    embs = model.encode(r.texts, normalize_embeddings=True).tolist()
    return {"embeddings": embs, "dim": len(embs[0]) if embs else 0}
