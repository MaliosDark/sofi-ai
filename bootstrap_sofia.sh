#!/usr/bin/env bash
set -euo pipefail

# Color codes for logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ====== Config ======
PROJECT_DIR="${PROJECT_DIR:-sofia}"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PORT="${PORT:-8000}"

# ====== Create layout ======
mkdir -p "$PROJECT_DIR"/{data,docs,docs/site}
cd "$PROJECT_DIR"

# ====== requirements.txt ======
cat > requirements.txt <<'REQ'
torch>=2.2
transformers>=4.44
sentence-transformers>=3.0.0
datasets>=2.20
faiss-cpu>=1.7.4
rank_bm25>=0.2.2
huggingface_hub>=0.24.0
accelerate>=0.34.0
peft>=0.13.0
trl>=0.9.6
mteb>=1.15.0
fastapi>=0.111.0
uvicorn>=0.30.0
pydantic>=2.8.0
scikit-learn>=1.4
numpy>=1.26
pdoc>=14.5.0
graphviz>=0.20.3
REQ

# ====== config.yaml ======
cat > config.yaml <<'YAML'
name: SOFIA
base_model: Qwen/Qwen3-Embedding-8B
output_dir: ./SOFIA
seed: 42
epochs: 2
batch_size: 128
lr: 2.0e-5
warmup_ratio: 0.06
max_len: 512
fp16: true
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
kd:
  teachers:
    - BAAI/bge-m3
    - intfloat/e5-mistral-7b-instruct
  kd_weight: 0.6
losses:
  - cosine
  - triplet
instruction_templates:
  query: "Query: {text}"
  doc:   "Document: {text}"
dims_to_export: [1024, 3072, 4096]
datasets:
  stsb: sentence-transformers/stsb
  nq:   mteb/nq
  quora: paws
  paws: paws
  banking77: banking77
hard_negatives:
  bm25_k: 50
  faiss_k: 50
  per_pos: 2
YAML

# ====== prepare_data.py ======
cat > prepare_data.py <<'PY'
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
    pairs = []
    stsb = load_dataset("sentence-transformers/stsb", split="train")
    for r in stsb:
        pairs.append(inst_pair(r["sentence1"], r["sentence2"], r["score"]/5.0))
    nq = load_dataset("mteb/nq", "default", split="test")
    for r in nq:
        q = r["query"]
        for d in r["positive_passages"][:1]:
            pairs.append(inst_pair(q, d["text"], 1.0))
        for d in r["negative_passages"][:1]:
            pairs.append(inst_pair(q, d["text"], 0.0))
    paws = load_dataset("paws", "labeled_final", split="train[:50%]")
    for r in paws:
        s = 1.0 if r["label"]==1 else 0.0
        pairs.append(inst_pair(r["sentence1"], r["sentence2"], s))
    b77 = load_dataset("banking77", split="train[:50%]")
    label2txt = {}
    for r in b77:
        label2txt.setdefault(r["label"], []).append(r["text"])
    for _, arr in label2txt.items():
        for i in range(min(200, max(1, len(arr)-1))):
            a, b = arr[i], arr[(i+1)%len(arr)]
            pairs.append(inst_pair(a, b, 0.9))
    random.shuffle(pairs)
    return pairs

def mine_bm25_negatives(pairs, per_pos=2, k=50):
    corpus = [p["d"] for p in pairs]
    bm25 = BM25Okapi([c.split() for c in corpus])
    out = []
    for p in pairs:
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
PY

# ====== mine_hard_negatives.py ======
cat > mine_hard_negatives.py <<'PY'
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
PY

# ====== distill_kd.py ======
cat > distill_kd.py <<'PY'
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
            out.write(json.dumps(r)+"
")
    print("saved:", args.outfile)
PY

# ====== train_sofia.py ======
cat > train_sofia.py <<'PY'
"""
train_sofia.py
Trains SOFIA with LoRA adapters + multi-loss (cosine + triplet).
Saves base in ./SOFIA and projection variants in ./SOFIA/proj-<dim>.
"""
import os, argparse, json, math, torch
from torch.utils import data
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models
from peft import LoraConfig, get_peft_model

class JsonlDataset(data.Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        label = float(r.get("kd", r["score"]))
        return (r["q"], r["d"], label)

def collate(batch):
    a = [x[0] for x in batch]; b = [x[1] for x in batch]; y = torch.tensor([x[2] for x in batch], dtype=torch.float32)
    return a, b, y

def projection_head(dim_in, dim_out):
    return models.Dense(in_features=dim_in, out_features=dim_out, bias=True, activation_function=torch.nn.Identity())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--train", default="./data/pairs_kd.jsonl")
    ap.add_argument("--out", default="./SOFIA")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--dims", nargs="+", type=int, default=[1024,3072,4096])
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    base = SentenceTransformer(args.base, trust_remote_code=True)
    if hasattr(base, "auto_model"):
        peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="FEATURE_EXTRACTION")
        base.auto_model = get_peft_model(base.auto_model, peft_cfg)

    # Add a projection head (we train with first head; later export other dims)
    base._modules["1"] = projection_head(base.get_sentence_embedding_dimension(), args.dims[0])

    ds = JsonlDataset(args.train)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)

    cos_loss = losses.CosineSimilarityLoss(base)
    triplet_loss = losses.TripletLoss(model=base, triplet_margin=0.2)

    steps_per_epoch = len(dl)
    warmup_steps = math.ceil(steps_per_epoch * args.epochs * args.warmup)

    base.fit(
        train_objectives=[(dl, cos_loss), (dl, triplet_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        use_amp=args.fp16,
        output_path=args.out
    )

    # Export extra dimensions
    for d in args.dims[1:]:
        m2 = SentenceTransformer(args.out)
        m2._modules["1"] = projection_head(m2.get_sentence_embedding_dimension(), d)
        m2.save(os.path.join(args.out, f"proj-{d}"))

    print("saved:", args.out)
PY

# ====== eval_compare.py ======
cat > eval_compare.py <<'PY'
"""
eval_compare.py
Quick subset MTEB evaluation for SOFIA vs baseline.
Outputs under ./eval_out/{SOFIA,QWEN}
"""
import os, argparse
from mteb import MTEB
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sofia", default="./SOFIA")
    ap.add_argument("--baseline", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--tasks", nargs="+", default=["STS12","STS13","STS14","BIOSSES","SICK-R"])
    ap.add_argument("--outdir", default="./eval_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for name, mid in [("SOFIA", args.sofia), ("QWEN", args.baseline)]:
        mdl = SentenceTransformer(mid)
        MTEB(tasks=args.tasks).run(mdl, output_folder=os.path.join(args.outdir, name))
    print("done:", args.outdir)
PY

# ====== serve_api.py ======
cat > serve_api.py <<'PY'
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
PY

# ====== README.md (live doc) ======
GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
cat > README.md <<'EOF'
# SOFIA — Embedding Pipeline

**Build hash:** \`${GIT_HASH}\`

This repo was fully generated by \`bootstrap_sofia.sh\`. It contains:
- Data prep + negatives (\`prepare_data.py\`, \`mine_hard_negatives.py\`)
- KD distillation (\`distill_kd.py\`)
- Training with LoRA + multi-loss (\`train_sofia.py\`)
- Evaluation vs baseline (\`eval_compare.py\`)
- Serving API (\`serve_api.py\`)
- Config: \`config.yaml\`

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pipeline
python prepare_data.py
python mine_hard_negatives.py
python distill_kd.py --teachers BAAI/bge-m3 intfloat/e5-mistral-7b-instruct
python train_sofia.py --fp16
python eval_compare.py

# API
uvicorn serve_api.py:app --host 0.0.0.0 --port 8000
```

## Live Docs
- OpenAPI: http://localhost:8000/docs
- ReDoc:    http://localhost:8000/redoc
- Developer docs (auto-generated from source): open \`docs/site/index.html\` after running:
  \`pdoc --html --output-dir docs/site .\`

## Model Artifacts
- Main: \`./SOFIA\`
- Projections: \`./SOFIA/proj-1024\`, \`./SOFIA/proj-3072\`, \`./SOFIA/proj-4096\`

## Files → Responsibilities
- \`prepare_data.py\`: builds pairs, adds BM25 negatives
- \`mine_hard_negatives.py\`: mines FAISS hard negatives (encoder-based)
- \`distill_kd.py\`: soft labels from teacher ensemble
- \`train_sofia.py\`: LoRA + cosine + triplet
- \`eval_compare.py\`: MTEB subset for SOFIA vs Qwen baseline
- \`serve_api.py\`: /embed endpoint to consume embeddings
EOF

# ====== pipeline diagram (docs) ======
mkdir -p docs
cat > docs/diagram.dot <<'DOT'
digraph SOFIA {
  rankdir=LR;
  node [shape=box, style=rounded];
  A[label="prepare_data.py
pairs.jsonl"];
  B[label="mine_hard_negatives.py
pairs_hard.jsonl"];
  C[label="distill_kd.py
pairs_kd.jsonl"];
  D[label="train_sofia.py
SOFIA/, proj-*"];
  E[label="eval_compare.py
eval_out/"];
  F[label="serve_api.py
FastAPI /embed"];

  A -> B -> C -> D -> E;
  D -> F;
}
DOT

# ====== Python venv + install ======
$PYTHON -m venv "../$VENV_DIR"
source "../$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

# ====== Run the full pipeline (no uploads) ======
echo -e "${GREEN}Starting data preparation...${NC}"
python prepare_data.py --out ./data/pairs.jsonl
echo -e "${GREEN}Mining hard negatives...${NC}"
python mine_hard_negatives.py --infile ./data/pairs.jsonl --outfile ./data/pairs_hard.jsonl
echo -e "${GREEN}Distilling knowledge...${NC}"
python distill_kd.py --infile ./data/pairs_hard.jsonl --outfile ./data/pairs_kd.jsonl   --teachers BAAI/bge-m3 intfloat/e5-mistral-7b-instruct
echo -e "${GREEN}Training SOFIA model...${NC}"
python train_sofia.py --base Qwen/Qwen3-Embedding-8B --train ./data/pairs_kd.jsonl --out ./SOFIA --epochs 2 --batch 128 --fp16
echo -e "${GREEN}Evaluating and comparing...${NC}"
python eval_compare.py --sofia ./SOFIA --baseline Qwen/Qwen3-Embedding-8B --tasks STS12 STS13 STS14 BIOSSES SICK-R

# ====== Build dev docs from code ======
pdoc --html --output-dir docs/site .

# ====== Render pipeline diagram ======
if command -v dot >/dev/null 2>&1; then
  dot -Tpng docs/diagram.dot -o docs/site/pipeline.png || true
fi

# ====== Final info ======
echo ""
echo -e "${BLUE}====================${NC}"
echo -e "${GREEN}SOFIA READY${NC}"
echo -e "${YELLOW}Model dir: $(pwd)/SOFIA${NC}"
echo -e "${YELLOW}Projections: SOFIA/proj-1024, SOFIA/proj-3072, SOFIA/proj-4096${NC}"
echo -e "${YELLOW}Eval out:   $(pwd)/eval_out${NC}"
echo -e "${YELLOW}Docs (code): $(pwd)/docs/site/index.html${NC}"
echo -e "${BLUE}To serve API:${NC}"
echo -e "  ${GREEN}uvicorn serve_api.py:app --host 0.0.0.0 --port ${PORT}${NC}"
echo -e "${BLUE}OpenAPI:    http://localhost:${PORT}/docs${NC}"
echo -e "${BLUE}====================${NC}"
