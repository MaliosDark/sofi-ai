"""
mine_hard_negatives.py
FAISS-based hard negative mining using a strong baseline encoder.
Inputs:  ./data/pairs.jsonl
Outputs: ./data/pairs_hard.jsonl
"""
import argparse, json, os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def read_jsonl(p):
    with open(p) as f:
        for line in f:
            yield json.loads(line)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="./data/pairs.jsonl")
    ap.add_argument("--outfile", default="./data/pairs_hard.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()

    data = list(read_jsonl(args.infile))
    docs = sorted(list(set([r["d"] for r in data])))
    mdl = SentenceTransformer(args.model)
    embs = mdl.encode(docs, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype("float32"))

    with open(args.outfile, "w") as out:
        for r in data:
            q = r["q"]
            qe = mdl.encode([q], normalize_embeddings=True)[0].astype("float32")
            faiss.normalize_L2(qe.reshape(1,-1))
            D, I = index.search(qe.reshape(1,-1), args.k)
            added = 0
            for j in I[0]:
                cand = docs[j]
                if cand != r["d"] and added < 2:
                    out.write(json.dumps({"q": q, "d": cand, "score": 0.0})+"
")
                    added += 1
            out.write(json.dumps(r)+"
")
    print("saved:", args.outfile)
