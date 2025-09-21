"""
distill_kd.py
Creates soft labels (similarities) from a teacher ensemble for KD.
Inputs:  ./data/pairs_hard.jsonl
Outputs: ./data/pairs_kd.jsonl
"""
import argparse, json
from sentence_transformers import SentenceTransformer
import numpy as np

def read_jsonl(p):
    with open(p) as f:
        for line in f:
            yield json.loads(line)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="./data/pairs_hard.jsonl")
    ap.add_argument("--outfile", default="./data/pairs_kd.jsonl")
    ap.add_argument("--teachers", nargs="+", required=True)
    args = ap.parse_args()

    teachers = [SentenceTransformer(t) for t in args.teachers]
    with open(args.outfile, "w") as out:
        for r in read_jsonl(args.infile):
            q, d = r["q"], r["d"]
            qv = np.stack([t.encode([q], normalize_embeddings=True)[0] for t in teachers],0)
            dv = np.stack([t.encode([d], normalize_embeddings=True)[0] for t in teachers],0)
            sim = float(np.mean(np.sum(qv*dv, axis=1)))  # average cosine
            r["kd"] = sim
            out.write(json.dumps(r)+"\n")
    print("saved:", args.outfile)
