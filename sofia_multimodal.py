#!/usr/bin/env python3
"""
SOFIA Multi-modal AGI System
Combines text and image embeddings for advanced understanding
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from typing import List, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SOFIAMultiModal(nn.Module):
    """
    Multi-modal SOFIA model combining text and vision capabilities
    """

    def __init__(self, text_model_path: str = "./SOFIA-v2-lora", vision_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()

        # Load text model (SOFIA)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_model = AutoModel.from_pretrained(text_model_path)

        # Load vision model (CLIP)
        self.vision_processor = CLIPProcessor.from_pretrained(vision_model_name)
        self.vision_model = CLIPModel.from_pretrained(vision_model_name)

        # Projection layers to align text and image embeddings
        self.text_projection = nn.Linear(768, 512)  # MPNet dim to CLIP dim
        self.image_projection = nn.Linear(512, 512)  # CLIP dim (already 512)

        # Multi-modal fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text inputs using SOFIA"""
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        # Project to common space
        embeddings = self.text_projection(embeddings)
        return embeddings

    def encode_image(self, images: Union[List[Image.Image], List[str]]) -> torch.Tensor:
        """Encode image inputs using CLIP"""
        # Handle URLs
        processed_images = []
        for img in images:
            if isinstance(img, str):
                # Load image from URL
                response = requests.get(img)
                img = Image.open(BytesIO(response.content))
            processed_images.append(img)

        inputs = self.vision_processor(images=processed_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.vision_model.get_image_features(**inputs)

        # Project to common space
        embeddings = self.image_projection(outputs)
        return embeddings

    def encode_multimodal(self, texts: List[str], images: Union[List[Image.Image], List[str], None] = None) -> torch.Tensor:
        """Encode multi-modal inputs (text + optional images)"""
        # Encode text
        text_embeddings = self.encode_text(texts)

        if images is None:
            # Text-only mode
            return text_embeddings

        # Encode images
        image_embeddings = self.encode_image(images)

        # Concatenate and fuse
        combined = torch.cat([text_embeddings, image_embeddings], dim=1)
        fused_embeddings = self.fusion_layer(combined)

        return fused_embeddings

    def compute_similarity(self, query_embedding: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between embeddings"""
        # Normalize embeddings
        query_norm = query_embedding / query_embedding.norm(dim=1, keepdim=True)
        target_norm = target_embeddings / target_embeddings.norm(dim=1, keepdim=True)

        # Cosine similarity
        similarity = torch.mm(query_norm, target_norm.t())
        return similarity

    def search_similar(self, query: Union[str, Tuple[str, Union[Image.Image, str]]],
                      candidates: List[Union[str, Tuple[str, Union[Image.Image, str]]]],
                      top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for most similar items to query

        Args:
            query: Either text string or (text, image) tuple
            candidates: List of text strings or (text, image) tuples
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples
        """

        # Encode query
        if isinstance(query, str):
            query_texts = [query]
            query_images = None
        else:
            query_texts = [query[0]]
            query_images = [query[1]]

        query_embedding = self.encode_multimodal(query_texts, query_images)

        # Encode candidates
        candidate_texts = []
        candidate_images = []

        for candidate in candidates:
            if isinstance(candidate, str):
                candidate_texts.append(candidate)
                candidate_images.append(None)
            else:
                candidate_texts.append(candidate[0])
                candidate_images.append(candidate[1])

        # Filter out None images
        valid_images = [img for img in candidate_images if img is not None]
        candidate_embeddings = self.encode_multimodal(candidate_texts, valid_images if valid_images else None)

        # Compute similarities
        similarities = self.compute_similarity(query_embedding, candidate_embeddings)

        # Get top-k results
        top_scores, top_indices = torch.topk(similarities[0], min(top_k, len(candidates)))

        results = [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]
        return results

class MultiModalSOFIA:
    """
    High-level interface for multi-modal SOFIA operations
    """

    def __init__(self, model_path: str = "./SOFIA-v2-lora"):
        self.model = SOFIAMultiModal(model_path)
        logger.info("Multi-modal SOFIA initialized")

    def describe_image(self, image: Union[Image.Image, str], context: str = "") -> str:
        """
        Generate a textual description of an image, optionally with context
        """
        # This is a simplified implementation
        # In a real system, this would use a captioning model
        image_embedding = self.model.encode_image([image])

        # For now, return a placeholder description
        # TODO: Integrate with a proper image captioning model
        return f"Image described with context: {context}"

    def find_similar_images(self, query_image: Union[Image.Image, str],
                           image_database: List[Union[Image.Image, str]],
                           top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find images similar to a query image
        """
        # Convert to tuples for multi-modal search
        query = ("", query_image)  # Empty text, image only
        candidates = [("image_" + str(i), img) for i, img in enumerate(image_database)]

        results = self.model.search_similar(query, candidates, top_k)
        return results

    def search_visual_content(self, text_query: str,
                            image_results: List[Union[Image.Image, str]],
                            top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for images that match a text description
        """
        query = (text_query, None)  # Text only
        candidates = [("image_" + str(i), img) for i, img in enumerate(image_results)]

        results = self.model.search_similar(query, candidates, top_k)
        return results

    def multimodal_retrieval(self, query: Union[str, Tuple[str, Union[Image.Image, str]]],
                           documents: List[Union[str, Tuple[str, Union[Image.Image, str]]]],
                           top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Perform retrieval across multi-modal documents
        """
        return self.model.search_similar(query, documents, top_k)


# Example usage and testing
if __name__ == "__main__":
    # Initialize multi-modal SOFIA
    mm_sofia = MultiModalSOFIA()

    # Example 1: Text-to-image search
    text_query = "a beautiful sunset over mountains"
    sample_images = [
        "https://picsum.photos/300/200?random=1",
        "https://picsum.photos/300/200?random=2",
        "https://picsum.photos/300/200?random=3"
    ]

    print("Searching for images matching:", text_query)
    results = mm_sofia.search_visual_content(text_query, sample_images, top_k=2)
    for idx, score in results:
        print(f"Image {idx}: similarity = {score:.4f}")

    # Example 2: Image-to-image similarity
    print("\nFinding similar images...")
    similar_results = mm_sofia.find_similar_images(sample_images[0], sample_images[1:], top_k=2)
    for idx, score in similar_results:
        print(f"Similar image {idx}: similarity = {score:.4f}")

    print("Multi-modal SOFIA demo completed!")
