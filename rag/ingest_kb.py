import os
import csv
import datetime
from rag.embedder import Embedder
from rag.chroma_setup import get_collection, reset_collection
from rag.utils import normalize_text

LOG_DIR = "logs/ingestion"
os.makedirs(LOG_DIR, exist_ok=True)

def ingest_csv_to_chroma(csv_path: str, collection_name: str = "farm_kb"):
    """Reads CSV, embeds text rows, and stores them in ChromaDB"""
    embedder = Embedder()  # Unified text+image embedding
    chroma_collection = get_collection(collection_name)

    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # comma-separated
        for i, row in enumerate(reader):
            crop_name = row.get("crop_name", "")
            disease_name = row.get("disease_name", "")
            symptoms = row.get("symptoms", "")
            possible_causes = row.get("possible_causes", "")
            treatment = row.get("treatment", "")
            recommended_crops = row.get("recommended_crops", "")

            # Combine all textual info for embedding
            text = (
                f"Crop: {crop_name}\n"
                f"Disease: {disease_name}\n"
                f"Symptoms: {symptoms}\n"
                f"Possible Causes: {possible_causes}\n"
                f"Treatment: {treatment}\n"
                f"Recommended Crops: {recommended_crops}"
            )

            text = normalize_text(text)
            entries.append({
                "id": f"{collection_name}_{i}",
                "content": text,
                "metadata": {
                    "crop_name": crop_name,
                    "disease_name": disease_name,
                    "symptoms": symptoms,
                    "possible_causes": possible_causes,
                    "treatment": treatment,
                    "recommended_crops": recommended_crops
                }
            })

    print(f"[INFO] Loaded {len(entries)} records from {csv_path}")

    # Generate embeddings
    texts = [e["content"] for e in entries]
    embeddings = embedder.embed_texts(texts)

    # Store in Chroma
    chroma_collection.add(
        ids=[e["id"] for e in entries],
        embeddings=embeddings,
        metadatas=[e["metadata"] for e in entries],
        documents=texts
    )

    # Logging
    log_path = os.path.join(LOG_DIR, f"{collection_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Ingested {len(entries)} entries from {csv_path}\n")
        for e in entries:
            f.write(f"{e['id']} → {e['metadata']['crop_name']} – {e['metadata']['disease_name']}\n")
    print(f"[LOG] Ingestion complete. Log saved to {log_path}")


if __name__ == "__main__":
    csv_file = "data/kb_source/diseases.csv"
    if not os.path.exists(csv_file):
        print(f"[ERROR] CSV not found: {csv_file}")
    else:
        ingest_csv_to_chroma(csv_file)
