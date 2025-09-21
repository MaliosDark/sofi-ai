# search.py
import sys, json, faiss
from sentence_transformers import SentenceTransformer

q = " ".join(sys.argv[1:]) or "machine learning"
m = SentenceTransformer("MaliosDark/sofia-embedding-v1")
docs = json.load(open("corpus.json"))
idx = faiss.read_index("sofia.faiss")
vq = m.encode([q], normalize_embeddings=True).astype("float32")
faiss.normalize_L2(vq)
D, I = idx.search(vq, 5)
for rank, i in enumerate(I[0], 1):
    print(f"{rank}. {D[0][rank-1]:.3f} - {docs[i][0]}")
