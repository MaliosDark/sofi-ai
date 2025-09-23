#!/usr/bin/env python3
"""
Data augmentation script for SOFIA training
Generates synthetic training pairs to improve model generalization
"""

import json
import argparse
import random
from typing import List, Dict, Any
import re

class SOFIADataAugmentor:
    """Data augmentation for SOFIA training data"""

    def __init__(self):
        # Query paraphrases
        self.query_templates = [
            "What is {topic}?",
            "Tell me about {topic}",
            "Explain {topic}",
            "What do you know about {topic}?",
            "Can you describe {topic}?",
            "What are the details about {topic}?",
            "Give me information on {topic}",
            "What is the meaning of {topic}?",
            "How does {topic} work?",
            "What are the characteristics of {topic}?"
        ]

        # Document paraphrases
        self.doc_templates = [
            "{topic} refers to {definition}",
            "{topic} is {definition}",
            "{topic} means {definition}",
            "The term {topic} describes {definition}",
            "{topic} can be defined as {definition}",
            "In general, {topic} is {definition}",
            "{topic} represents {definition}",
            "Basically, {topic} is {definition}"
        ]

        # Synonym mappings for common terms
        self.synonyms = {
            "artificial intelligence": ["AI", "machine intelligence", "computer intelligence"],
            "machine learning": ["ML", "statistical learning", "automated learning"],
            "neural network": ["neural net", "artificial neural network", "ANN"],
            "deep learning": ["deep neural networks", "hierarchical learning"],
            "computer vision": ["image recognition", "visual computing"],
            "natural language processing": ["NLP", "language understanding", "text analysis"],
            "reinforcement learning": ["RL", "reward-based learning"],
            "supervised learning": ["labeled learning", "teacher-student learning"],
            "unsupervised learning": ["unlabeled learning", "self-organized learning"]
        }

    def extract_topic(self, text: str) -> str:
        """Extract main topic from text"""
        # Simple topic extraction - look for key terms
        text_lower = text.lower()

        for topic, variants in self.synonyms.items():
            for variant in [topic] + variants:
                if variant in text_lower:
                    return topic

        # Fallback: extract first noun phrase
        words = text.split()
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 3:
                return word.lower()

        return "general_topic"

    def paraphrase_query(self, query: str) -> List[str]:
        """Generate paraphrased versions of a query"""
        topic = self.extract_topic(query)
        paraphrases = []

        # Generate paraphrases using templates
        for template in self.query_templates[:3]:  # Limit to avoid too many
            paraphrase = template.format(topic=topic)
            if paraphrase != query:
                paraphrases.append(paraphrase)

        # Synonym substitution
        for term, synonyms in self.synonyms.items():
            if term in query.lower():
                for synonym in synonyms[:2]:  # Limit synonyms
                    new_query = re.sub(r'\b' + re.escape(term) + r'\b', synonym, query, flags=re.IGNORECASE)
                    if new_query != query:
                        paraphrases.append(new_query)

        return list(set(paraphrases))  # Remove duplicates

    def paraphrase_document(self, document: str) -> List[str]:
        """Generate paraphrased versions of a document"""
        topic = self.extract_topic(document)
        paraphrases = []

        # Extract definition-like part
        sentences = document.split('.')
        definition = sentences[0] if sentences else document

        # Generate paraphrases using templates
        for template in self.doc_templates[:2]:  # Limit to avoid too many
            paraphrase = template.format(topic=topic, definition=definition)
            if paraphrase != document:
                paraphrases.append(paraphrase)

        # Synonym substitution
        for term, synonyms in self.synonyms.items():
            if term in document.lower():
                for synonym in synonyms[:1]:  # Limit synonyms
                    new_doc = re.sub(r'\b' + re.escape(term) + r'\b', synonym, document, flags=re.IGNORECASE)
                    if new_doc != document:
                        paraphrases.append(new_doc)

        return list(set(paraphrases))  # Remove duplicates

    def augment_pair(self, query: str, document: str, score: float) -> List[Dict[str, Any]]:
        """Augment a single query-document pair"""
        augmented_pairs = []

        # Original pair
        augmented_pairs.append({
            "q": query,
            "d": document,
            "score": score,
            "augmentation_type": "original"
        })

        # Query paraphrases
        query_paraphrases = self.paraphrase_query(query)
        for q_para in query_paraphrases[:2]:  # Limit augmentations
            augmented_pairs.append({
                "q": q_para,
                "d": document,
                "score": score * 0.9,  # Slightly lower score for paraphrases
                "augmentation_type": "query_paraphrase"
            })

        # Document paraphrases
        doc_paraphrases = self.paraphrase_document(document)
        for d_para in doc_paraphrases[:1]:  # Limit augmentations
            augmented_pairs.append({
                "q": query,
                "d": d_para,
                "score": score * 0.95,  # Slightly lower score for paraphrases
                "augmentation_type": "doc_paraphrase"
            })

        # Combined paraphrases (query + doc)
        if query_paraphrases and doc_paraphrases:
            q_para = query_paraphrases[0]
            d_para = doc_paraphrases[0]
            augmented_pairs.append({
                "q": q_para,
                "d": d_para,
                "score": score * 0.85,  # Lower score for combined
                "augmentation_type": "combined_paraphrase"
            })

        return augmented_pairs

    def augment_dataset(self, input_file: str, output_file: str, augmentation_factor: int = 2):
        """Augment entire dataset"""
        print(f"Loading data from {input_file}")

        data = []
        with open(input_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        print(f"Loaded {len(data)} original pairs")

        augmented_data = []
        original_count = 0
        augmented_count = 0

        for item in data:
            query = item["q"]
            document = item["d"]
            score = item.get("score", 1.0)

            # Augment this pair
            augmented_pairs = self.augment_pair(query, document, score)

            # Add to dataset (limit based on augmentation factor)
            max_pairs = min(len(augmented_pairs), augmentation_factor)
            for pair in augmented_pairs[:max_pairs]:
                augmented_data.append(pair)
                if pair["augmentation_type"] == "original":
                    original_count += 1
                else:
                    augmented_count += 1

        # Save augmented dataset
        print(f"Saving {len(augmented_data)} pairs ({original_count} original, {augmented_count} augmented)")

        with open(output_file, 'w') as f:
            for item in augmented_data:
                f.write(json.dumps(item) + '\n')

        print(f"Augmented dataset saved to {output_file}")

        # Print statistics
        augmentation_types = {}
        for item in augmented_data:
            aug_type = item.get("augmentation_type", "unknown")
            augmentation_types[aug_type] = augmentation_types.get(aug_type, 0) + 1

        print("\nAugmentation Statistics:")
        for aug_type, count in augmentation_types.items():
            print(f"  {aug_type}: {count} pairs")

def main():
    parser = argparse.ArgumentParser(description="Data augmentation for SOFIA training")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--factor", type=int, default=3, help="Augmentation factor (max pairs per original)")
    args = parser.parse_args()

    augmentor = SOFIADataAugmentor()
    augmentor.augment_dataset(args.input, args.output, args.factor)

if __name__ == "__main__":
    main()
