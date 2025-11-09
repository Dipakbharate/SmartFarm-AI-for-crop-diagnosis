"""
chroma_setup.py
-----------------------------------
Handles initialization and persistent setup for ChromaDB.
Ensures a single shared client for the entire SmartFarm AI project.
"""

from pathlib import Path
import chromadb
from chromadb.config import Settings

# Directory for storing Chroma's persistent data
CHROMA_DB_DIR = Path("./chroma_db")

# Default collection name for SmartFarm Knowledge Base
DEFAULT_COLLECTION = "farm_kb"


def get_chroma_client():
    """
    Initialize and return a persistent Chroma client.
    If the folder does not exist, it is created automatically.
    """
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True  # allows clean rebuild if needed
        )
    )
    return client


def get_collection(collection_name: str = DEFAULT_COLLECTION):
    """
    Returns (or creates) a Chroma collection with consistent configuration.
    """
    client = get_chroma_client()

    try:
        # Try to get or create the collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "SmartFarm AI knowledge base for crop diseases"}
        )
    except Exception as e:
        print(f"[WARN] Issue fetching collection: {e}")
        print("[INFO] Recreating a clean collection...")
        collection = client.create_collection(name=collection_name)

    return collection


def reset_collection(collection_name: str = DEFAULT_COLLECTION):
    """
    Deletes and recreates the collection (used when re-ingesting KB).
    """
    client = get_chroma_client()

    try:
        client.delete_collection(name=collection_name)
        print(f"[RESET] Deleted existing collection '{collection_name}'.")
    except Exception:
        print(f"[INFO] No existing collection '{collection_name}' found to delete.")

    new_collection = client.create_collection(name=collection_name)
    print(f"[RESET] Recreated collection '{collection_name}'.")
    return new_collection


if __name__ == "__main__":
    # Test the setup
    print("🔧 Initializing Chroma client...")
    coll = get_collection()
    print(f"✅ Collection ready: {coll.name}")
