"""
query_kb.py
Performs semantic search on ChromaDB using text 
Supports multimodal embedding (via Embedder).
"""

import os
from rag.chroma_setup import get_collection
from rag.embedder import Embedder

# Initialize global embedder (shared for both text and images)
embedder = Embedder()
collection = get_collection("farm_kb")

def query_knowledge_base(query_input, top_k=3):
    """
    Query the knowledge base with either text (disease/symptom) or an image file.
    Automatically detects input type and returns closest matches.
    """

    # Detect if input is an image or text
    if isinstance(query_input, str) and os.path.exists(query_input):
        print(f"[QUERY] Detected image input: {query_input}")
        query_vector = embedder.embed_images([query_input])
    else:
        print(f"[QUERY] Detected text input: {query_input}")
        query_vector = embedder.embed_texts([query_input])

    if query_vector is None or len(query_vector) == 0:
        print("[ERROR] No embedding could be generated for query input.")
        return []

    # Query ChromaDB
    print(f"[SEARCH] Running similarity search on ChromaDB ({top_k} results)...")
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Debug info
    print("DEBUG: raw result keys:", results.keys())

    if not results.get("documents"):
        print("[WARN] No results found.")
        return []

    # Extract and format results
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    formatted_results = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        entry = {
            "crop": meta.get("crop_name", "Unknown"),
            "disease": meta.get("disease_name", "Unknown"),
            "symptoms": meta.get("symptoms", "Unknown"),
            "causes": meta.get("possible_causes", "Unknown"),
            "treatment": meta.get("treatment", "Unknown"),
            "recommended": meta.get("recommended_crops", "Unknown"),
            "distance": float(dist)
        }
        formatted_results.append(entry)

    return formatted_results


def pretty_print_results(results):
    """
    Nicely prints out query results for readability.
    """
    if not results:
        print("⚠️ No relevant results found.")
        return

    for r in results:
        print(f"🌾 Crop: {r['crop']}")
        print(f"🦠 Disease: {r['disease']}")
        print(f"🔍 Symptoms: {r['symptoms']}")
        print(f"⚠️ Possible Causes: {r['causes']}")
        print(f"💡 Treatment: {r['treatment']}")
        print(f"🌱 Recommended Crops: {r['recommended']}")
        print(f"🔢 Distance: {r['distance']:.3f}")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Example 1: text query
    query_text = "dark spots on tomato leaves"
    text_results = query_knowledge_base(query_text, top_k=3)
    pretty_print_results(text_results)

    # Example 2: image query (future use)
    #query_image = "data/uploads/sample_leaf.JPG"
    #if os.path.exists(query_image):
     #   image_results = query_knowledge_base(query_image, top_k=3)
     #   pretty_print_results(image_results)
    #else:
     #   print(f"[INFO] Sample image not found at {query_image}. Skipping image test.")
