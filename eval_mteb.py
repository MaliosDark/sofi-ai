# eval_mteb.py
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import sys

mid = sys.argv[1] if len(sys.argv) > 1 else "MaliosDark/sofia-embedding-v1"
subset = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark"]
MTEB(tasks=subset).run(SentenceTransformer(mid), output_folder=f"./mteb_{mid.split('/')[-1]}")
print("Done.")
