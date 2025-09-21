#!/usr/bin/env python3
"""
compare_embeddings.py
Compare embeddings from SOFIA vs baseline model for the same text.
Shows cosine similarity and first 10 values of each embedding.
"""
import sys
import numpy as np
from sentence_transformers import SentenceTransformer, util

def main():
    if len(sys.argv) < 2:
        print("Usage: echo 'text' | python compare_embeddings.py [mode]")
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else "query"

    # Read input text
    text = sys.stdin.read().strip()
    if not text:
        print("❌ No input text provided")
        sys.exit(1)

    print(f"🔍 Comparing embeddings for: '{text}'")
    print(f"🎯 Mode: {mode}")
    print()

    # Load models
    print("🤖 Loading SOFIA model...")
    sofia_model = SentenceTransformer("./SOFIA-v2-lora")

    print("📊 Loading baseline model (MPNet)...")
    baseline_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Generate embeddings
    print("⚡ Generating SOFIA embedding...")
    sofia_emb = sofia_model.encode([text], normalize_embeddings=True)[0]

    print("⚡ Generating baseline embedding...")
    baseline_emb = baseline_model.encode([text], normalize_embeddings=True)[0]

    # SOFIA has 1024 dims (768 MPNet + 256 dense), baseline has 768
    print(f"📏 Embedding dimensions - SOFIA: {len(sofia_emb)}, Baseline: {len(baseline_emb)}")

    # Compare first 768 dimensions (MPNet backbone)
    sofia_mpnet_part = sofia_emb[:768]
    similarity = util.cos_sim([sofia_mpnet_part], [baseline_emb]).item()
    print(f"📊 Cosine similarity (MPNet backbone): {similarity:.4f}")

    # Also compare full SOFIA vs baseline (if possible)
    if len(sofia_emb) == len(baseline_emb):
        full_similarity = util.cos_sim([sofia_emb], [baseline_emb]).item()
        print(f"📊 Cosine similarity (full): {full_similarity:.4f}")
    else:
        print("📊 Full comparison not possible due to different dimensions")
    # Show first 10 values
    print("\n📈 First 10 embedding values:")
    print(f"SOFIA (full):     {sofia_emb[:10]}")
    print(f"SOFIA (MPNet):    {sofia_mpnet_part[:10]}")
    print(f"Baseline:         {baseline_emb[:10]}")

    # Show differences for MPNet part
    diff = sofia_mpnet_part - baseline_emb
    print("\n🔄 Differences (SOFIA MPNet - Baseline):")
    print(f"First 10: {diff[:10]}")
    print(f"Mean absolute difference: {np.mean(np.abs(diff)):.4f}")
    print(f"Max absolute difference: {np.max(np.abs(diff)):.4f}")

    # Show SOFIA dense head contribution
    dense_head = sofia_emb[768:]
    print(f"\n🏗️  SOFIA dense head (first 10): {dense_head[:10]}")
    print(f"Dense head magnitude: {np.linalg.norm(dense_head):.4f}")
if __name__ == "__main__":
    main()
