# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import json

P = json.load(open("prompts.json"))
m = SentenceTransformer("MaliosDark/sofia-embedding-v1")  # or your HF id

app = FastAPI(title="SOFIA API")

class Req(BaseModel):
    texts: list[str]
    mode: str = "query"

@app.post("/embed")
def embed(r: Req):
    pref = P.get(r.mode, "")
    X = [pref + t for t in r.texts]
    v = m.encode(X, normalize_embeddings=P["normalize"]).tolist()
    return {"dim": len(v[0]) if v else 0, "embeddings": v}
