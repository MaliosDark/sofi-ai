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
