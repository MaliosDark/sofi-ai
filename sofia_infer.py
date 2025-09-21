# sofia_infer.py - Prompted inference for SOFIA embeddings
import json, sys
from sentence_transformers import SentenceTransformer

print("🔄 Loading prompts configuration...", file=sys.stderr)
P = json.load(open("prompts.json"))

print("🤖 Loading SOFIA model...", file=sys.stderr)
m = SentenceTransformer("./SOFIA-v2-lora")

def enc(texts, mode="query"):
    pref = P[mode]
    x = [pref + t for t in texts]
    print(f"📝 Encoding {len(texts)} text(s) with mode '{mode}'...", file=sys.stderr)
    return m.encode(x, normalize_embeddings=P["normalize"])

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "query"
    print(f"🎯 Starting inference in mode: {mode}", file=sys.stderr)

    input_text = sys.stdin.read().strip()
    if not input_text:
        print("❌ No input text provided", file=sys.stderr)
        sys.exit(1)

    print(f"📥 Input text: '{input_text[:50]}{'...' if len(input_text) > 50 else ''}'", file=sys.stderr)

    embedding = enc([input_text], mode=mode)[0]
    print("✅ Embedding generated successfully", file=sys.stderr)
    print(embedding.tolist())
