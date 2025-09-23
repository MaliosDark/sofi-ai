"""
mine_hard_negatives.py
Advanced FAISS-based hard negative mining with diversity sampling and quality filtering.
Inputs:  ./data/pairs.jsonl
Outputs: ./data/pairs_hard.jsonl
"""
import argparse, json, os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def read_jsonl(p):
    with open(p) as f:
        for line in f:
            yield json.loads(line)

def calculate_diversity_score(embeddings, indices, query_emb):
    """Calculate diversity score for selected negatives"""
    if len(indices) <= 1:
        return 1.0

    selected_embs = embeddings[indices]
    # Calculate pairwise similarities
    similarities = cosine_similarity(selected_embs)
    # Remove self-similarities
    np.fill_diagonal(similarities, 0)
    # Average similarity (lower is more diverse)
    avg_similarity = np.mean(similarities)
    # Convert to diversity score (higher is more diverse)
    diversity_score = 1.0 - avg_similarity
    return diversity_score

def filter_by_semantic_diversity(candidates, embeddings, query_emb, max_candidates=5, diversity_threshold=0.7):
    """Filter candidates by semantic diversity"""
    if len(candidates) <= max_candidates:
        return candidates

    # Calculate similarities to query
    candidate_embs = embeddings[candidates]
    query_similarities = cosine_similarity([query_emb], candidate_embs)[0]

    # Sort by similarity (we want hard negatives, so moderately similar)
    sorted_indices = np.argsort(query_similarities)

    # Select diverse subset
    selected = []
    selected_embs = []

    for idx in sorted_indices:
        if len(selected) >= max_candidates:
            break

        candidate_emb = candidate_embs[idx]

        # Check diversity with already selected
        if selected_embs:
            similarities = cosine_similarity([candidate_emb], selected_embs)[0]
            max_similarity = np.max(similarities)

            if max_similarity < diversity_threshold:
                selected.append(candidates[idx])
                selected_embs.append(candidate_emb)
        else:
            selected.append(candidates[idx])
            selected_embs.append(candidate_emb)

    return selected

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Advanced hard negative mining for SOFIA")
    ap.add_argument("--infile", default="./data/pairs.jsonl")
    ap.add_argument("--outfile", default="./data/pairs_hard.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--bm25_k", type=int, default=100, help="Number of BM25 candidates")
    ap.add_argument("--faiss_k", type=int, default=100, help="Number of FAISS candidates")
    ap.add_argument("--per_pos", type=int, default=3, help="Hard negatives per positive")
    ap.add_argument("--diversity_threshold", type=float, default=0.8, help="Diversity threshold for selection")
    args = ap.parse_args()

    print("Loading data and model...")
    data = list(read_jsonl(args.infile))
    docs = sorted(list(set([r["d"] for r in data])))

    print(f"Found {len(data)} pairs and {len(docs)} unique documents")

    # Load model
    mdl = SentenceTransformer(args.model, trust_remote_code=True)

    # Encode all documents
    print("Encoding documents...")
    embs = mdl.encode(docs, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    faiss.normalize_L2(embs)

    # Build FAISS index
    dimension = embs.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embs.astype("float32"))

    print(f"Built FAISS index with {len(docs)} documents")

    # Process each query
    processed_count = 0
    hard_negatives_added = 0

    with open(args.outfile, "w") as out:
        for r in data:
            q = r["q"]
            positive_doc = r["d"]

            # Encode query
            qe = mdl.encode([q], normalize_embeddings=True)[0].astype("float32")
            faiss.normalize_L2(qe.reshape(1,-1))

            # Find candidates using FAISS
            D, I = index.search(qe.reshape(1,-1), args.faiss_k)

            # Filter out the positive document and select diverse hard negatives
            candidates = []
            for j in I[0]:
                cand = docs[j]
                if cand != positive_doc:
                    candidates.append(j)

            # Apply diversity filtering
            diverse_negatives = filter_by_semantic_diversity(
                candidates, embs, qe,
                max_candidates=args.per_pos,
                diversity_threshold=args.diversity_threshold
            )

            # Write hard negatives
            added = 0
            for idx in diverse_negatives:
                if added >= args.per_pos:
                    break
                cand = docs[idx]
                out.write(json.dumps({"q": q, "d": cand, "score": 0.0}) + "\n")
                added += 1
                hard_negatives_added += 1

            # Write original positive pair
            out.write(json.dumps(r) + "\n")

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} queries, added {hard_negatives_added} hard negatives")

    print(f"Completed! Processed {processed_count} queries")
    print(f"Added {hard_negatives_added} hard negatives")
    print(f"Saved to: {args.outfile}")

    # Print statistics
    total_pairs = processed_count + hard_negatives_added
    print(f"Total pairs in output: {total_pairs}")
    print(".1f"
