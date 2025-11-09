"""
embedder.py
Handles unified embeddings for text and images.
Future-ready for multimodal retrieval (RAG + Vision AI).
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

class Embedder:
    def __init__(self, model_name_text="all-MiniLM-L6-v2", model_name_image="clip-ViT-B-32"):
        """
        Initialize both text and image embedders.
        - SentenceTransformer for text
        - CLIP for image embeddings
        """
        print("[INIT] Loading embedding models...")

        # Text model (fast + accurate)
        self.text_model = SentenceTransformer(model_name_text)

        # Image model (CLIP-based)
        self.image_model = SentenceTransformer(model_name_image)

        print("[READY] Embedder initialized with:")
        print(f" - Text model: {model_name_text}")
        print(f" - Image model: {model_name_image}")

    # ------------------- TEXT -------------------
    def embed_texts(self, texts):
        """
        Takes a list of text strings and returns embeddings as numpy arrays.
        """
        if isinstance(texts, str):
            texts = [texts]

        print(f"[EMBED] Generating text embeddings for {len(texts)} entries...")
        embeddings = self.text_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    # ------------------- IMAGE -------------------
    def embed_images(self, image_paths):
        """
        Takes one or multiple image paths and returns their embeddings.
        Uses CLIP’s vision encoder.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[WARN] Could not load image {path}: {e}")

        if not images:
            return np.array([])

        print(f"[EMBED] Generating image embeddings for {len(images)} images...")
        embeddings = self.image_model.encode(images, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    # ------------------- AUTO-DETECT -------------------
    def embed(self, inputs):
        """
        Automatically detects text or image and embeds accordingly.
        """
        if isinstance(inputs, str) and os.path.exists(inputs):
            return self.embed_images([inputs])
        elif isinstance(inputs, list) and all(isinstance(i, str) and os.path.exists(i) for i in inputs):
            return self.embed_images(inputs)
        else:
            return self.embed_texts(inputs)

# Example usage:
if __name__ == "__main__":
    embedder = Embedder()
    
    # Example 1: text embedding
    texts = ["Tomato leaf spot", "Brown rust on wheat"]
    text_vecs = embedder.embed_texts(texts)
    print("Text embedding shape:", text_vecs.shape)

    # Example 2: image embedding (optional, future use)
    # image_vecs = embedder.embed_images(["data/sample_leaf.jpg"])
    # print("Image embedding shape:", image_vecs.shape)
