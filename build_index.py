# build_index.py
import faiss, json, os, glob
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("MaliosDark/sofia-embedding-v1")
docs = []
for p in glob.glob("corpus/*.txt"):
    docs.append((os.path.basename(p), open(p, encoding="utf-8").read()))

V = m.encode([d[1] for d in docs], normalize_embeddings=True, show_progress_bar=True)
faiss.normalize_L2(V)
idx = faiss.IndexFlatIP(V.shape[1])
idx.add(V.astype("float32"))
faiss.write_index(idx, "sofia.faiss")
json.dump(docs, open("corpus.json", "w"), ensure_ascii=False)
print("OK")
