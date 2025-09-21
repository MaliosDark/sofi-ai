#!/usr/bin/env python3
"""
Test script for SOFIA Multi-modal capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sofia_multimodal import MultiModalSOFIA
import numpy as np

def test_text_encoding():
    """Test basic text encoding functionality"""
    print("Testing text encoding...")

    try:
        mm_sofia = MultiModalSOFIA()

        # Test text encoding
        texts = ["Hello world", "How are you?", "Machine learning is amazing"]
        embeddings = mm_sofia.model.encode_text(texts)

        print(f"✓ Text encoding successful. Shape: {embeddings.shape}")
        print(f"✓ Embeddings are finite: {torch.isfinite(embeddings).all().item()}")

        return True
    except Exception as e:
        print(f"✗ Text encoding failed: {e}")
        return False

def test_similarity_computation():
    """Test similarity computation between embeddings"""
    print("\nTesting similarity computation...")

    try:
        mm_sofia = MultiModalSOFIA()

        # Create some test embeddings
        texts1 = ["I love programming"]
        texts2 = ["I enjoy coding", "I hate vegetables", "Programming is fun"]

        emb1 = mm_sofia.model.encode_text(texts1)
        emb2 = mm_sofia.model.encode_text(texts2)

        similarities = mm_sofia.model.compute_similarity(emb1, emb2)

        print(f"✓ Similarity computation successful. Shape: {similarities.shape}")
        print(f"Similarity between '{texts1[0]}' and '{texts2[0]}': {similarities[0,0]:.4f}")
        print(f"Similarity between '{texts1[0]}' and '{texts2[1]}': {similarities[0,1]:.4f}")
        print(f"Similarity between '{texts1[0]}' and '{texts2[2]}': {similarities[0,2]:.4f}")
        return True
    except Exception as e:
        print(f"✗ Similarity computation failed: {e}")
        return False

def test_multimodal_interface():
    """Test the high-level multi-modal interface"""
    print("\nTesting multi-modal interface...")

    try:
        mm_sofia = MultiModalSOFIA()

        # Test with text-only queries (simulated multi-modal)
        query = "beautiful landscape"
        candidates = ["mountain view", "city skyline", "ocean sunset", "forest path"]

        # For now, we'll simulate with text-only since we don't have real images
        results = mm_sofia.model.search_similar(query, candidates, top_k=2)

        print("✓ Multi-modal interface working")
        print("Top 2 similar items to '{}':".format(query))
        for idx, score in results:
            print(f"  Item {idx}: similarity = {score:.4f}")
        return True
    except Exception as e:
        print(f"✗ Multi-modal interface failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Testing SOFIA Multi-modal Capabilities")
    print("=" * 50)

    tests = [
        test_text_encoding,
        test_similarity_computation,
        test_multimodal_interface
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All multi-modal tests passed! SOFIA is ready for multi-modal operations.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")

    return passed == total

if __name__ == "__main__":
    import torch
    success = main()
    sys.exit(0 if success else 1)
