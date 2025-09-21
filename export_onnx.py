# export_onnx.py
from sentence_transformers import SentenceTransformer
from pathlib import Path

m = SentenceTransformer("MaliosDark/sofia-embedding-v1")
p = Path("onnx")
p.mkdir(exist_ok=True)
m.save_to_hub = None
m._first_module().save(str(p/"0_Transformer"))
print("NOTE: For full ST ONNX, prefer `optimum` pipelines or wrap encode() in your app.")
