"""
prepare_data.py
Builds multi-task training pairs and mines BM25 negatives.
Outputs: ./data/pairs.jsonl
"""
import os, random, argparse, json
from datasets import load_dataset
from rank_bm25 import BM25Okapi

random.seed(42)

def inst_pair(q, d, score):
    return {"q": f"Query: {q}", "d": f"Document: {d}", "score": float(score)}

def load_pairs():
    print("Starting load_pairs...")
    pairs = []
    print("Loading STSB...")
    stsb = load_dataset("sentence-transformers/stsb", split="train")
    print(f"STSB loaded, {len(stsb)} rows")
    for r in stsb:
        pairs.append(inst_pair(r["sentence1"], r["sentence2"], r["score"]/5.0))
    print(f"STSB pairs added, total pairs: {len(pairs)}")
    # Skip nq for now to avoid download issues
    # nq = load_dataset("mteb/nq", "default", split="test")
    # for r in nq:
    #     q = r["query"]
    #     for d in r["positive_passages"][:1]:
    #         pairs.append(inst_pair(q, d["text"], 1.0))
    #     for d in r["negative_passages"][:1]:
    #         pairs.append(inst_pair(q, d["text"], 0.0))
    print("Loading PAWS...")
    paws = load_dataset("paws", "labeled_final", split="train[:5%]")
    print(f"PAWS loaded, {len(paws)} rows")
    for r in paws:
        s = 1.0 if r["label"]==1 else 0.0
        pairs.append(inst_pair(r["sentence1"], r["sentence2"], s))
    print(f"PAWS pairs added, total pairs: {len(pairs)}")
    print("Loading Banking77...")
    b77 = load_dataset("banking77", split="train[:5%]")
    print(f"Banking77 loaded, {len(b77)} rows")
    label2txt = {}
    for r in b77:
        label2txt.setdefault(r["label"], []).append(r["text"])
    for _, arr in label2txt.items():
        for i in range(min(200, max(1, len(arr)-1))):
            a, b = arr[i], arr[(i+1)%len(arr)]
            pairs.append(inst_pair(a, b, 0.9))
    print(f"Banking77 pairs added, total pairs: {len(pairs)}")
    random.shuffle(pairs)
    print(f"Total pairs after shuffle: {len(pairs)}")
    return pairs

def mine_bm25_negatives(pairs, per_pos=2, k=50):
    print("Starting mine_bm25_negatives...")
    corpus = [p["d"] for p in pairs]
    print(f"Corpus size: {len(corpus)}")
    bm25 = BM25Okapi([c.split() for c in corpus])
    print("BM25 index built")
    out = []
    for i, p in enumerate(pairs):
        if i % 100 == 0:
            print(f"Processing pair {i}/{len(pairs)}")
        q = p["q"].replace("Query: ","")
        scores = bm25.get_scores(q.split())
        idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        added = 0
        for j in idxs:
            c = corpus[j]
            if c != p["d"] and added < per_pos:
                out.append({"q": p["q"], "d": f"Document: {c}", "score": 0.0})
                added += 1
        out.append(p)
    print(f"Mining done, total mined pairs: {len(out)}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/pairs.jsonl")
    ap.add_argument("--per_pos", type=int, default=2)
    ap.add_argument("--bm25_k", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pairs = load_pairs()
    mined = mine_bm25_negatives(pairs, args.per_pos, args.bm25_k)
    with open(args.out, "w") as f:
        for r in mined:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print("saved:", args.out, "rows:", len(mined))
