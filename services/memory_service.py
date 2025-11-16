"""
services/memory_service.py
------------------------------------------------------------
Hybrid Memory Service for SmartFarm AI.

Responsibilities:
1️⃣ Store and retrieve image embeddings and past diagnoses.
2️⃣ Use ChromaDB PersistentClient for fast vector search.
3️⃣ Fall back to local JSON memory when ChromaDB is unavailable.
4️⃣ Optionally store text interactions for context continuity.

Author: SmartFarm AI Team
"""


import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


# 🧹 Auto-reset memory cache (temporary ChromaDB for embeddings)
import shutil; shutil.rmtree("data/memory_db", ignore_errors=True)

from core.feature_extractor import FeatureExtractor

from core.feature_extractor import FeatureExtractor
import chromadb

def clear_chroma_runtime(memory_db_dir: str | None = None):
    """
    Forcefully reset on-disk memory_db and attempt to clear chromadb runtime.
    Use during development to get a fresh image cache.
    """
    import shutil, os
    from pathlib import Path
    if memory_db_dir is None:
        memory_db_dir = Path(__file__).resolve().parents[1] / "data" / "memory_db"
    else:
        memory_db_dir = Path(memory_db_dir)
    shutil.rmtree(memory_db_dir, ignore_errors=True)
    memory_db_dir.mkdir(parents=True, exist_ok=True)

    # best-effort: attempt to nudge chromadb runtime to reset
    try:
        import chromadb
        try:
            chromadb.Client().reset()
        except Exception:
            pass
    except Exception:
        pass

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = BASE_DIR / "data" / "memory"
EMBED_FILE = MEMORY_DIR / "embeddings.json"
HISTORY_FILE = MEMORY_DIR / "history.json"

os.makedirs(MEMORY_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Corrupted JSON file: %s", path)
    return {}


def _save_json(path: Path, data: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Failed to save JSON file %s: %s", path, e)


# ============================================================
# Memory Service
# ============================================================
class MemoryService:
    """
    Unified interface for SmartFarm AI's memory subsystem.

    Prefers ChromaDB persistent client for embedding caching,
    falls back to local JSON files when Chroma is unavailable.
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.extractor = FeatureExtractor.get_instance()

        # Attempt to initialize ChromaDB PersistentClient
        self._init_chromadb()

        # Fallback local JSON store
        self.embeddings = _load_json(EMBED_FILE)
        self.history = _load_json(HISTORY_FILE)
        logger.info("✅ MemoryService initialized (Chroma=%s)", self._has_chroma)

    # --------------------------------------------------------
    # Setup ChromaDB (persistent vector store)
    # --------------------------------------------------------
    def _init_chromadb(self):
        self._has_chroma = False
        try:
            chroma_path = Path(__file__).resolve().parents[1] / "data" / "memory_db"
            client = chromadb.PersistentClient(path=str(chroma_path))
            self.collection = client.get_or_create_collection("image_memory")
            self._has_chroma = True
            logger.info("🧠 Using ChromaDB PersistentClient for memory.")
        except Exception as e:
            logger.warning("⚠️ ChromaDB not available, falling back to JSON: %s", e)

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        a_norm = a / (np.linalg.norm(a) + 1e-9)
        b_norm = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a_norm, b_norm))

    @staticmethod
    def _to_jsonable(arr: np.ndarray) -> List[float]:
        return arr.tolist()

    @staticmethod
    def _from_jsonable(data: List[float]) -> np.ndarray:
        return np.array(data, dtype=np.float32)

    # --------------------------------------------------------
    # Main public methods
    # --------------------------------------------------------
    def get_previous_diagnosis(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Check if the image (or similar one) has been seen before.
        Returns similar record if similarity > threshold.
        """
        emb = self.extractor.extract_from_path(image_path)

        # --- 1️⃣ Try ChromaDB vector search ---
        if self._has_chroma:
            try:
                results = self.collection.query(query_embeddings=[emb.tolist()], n_results=1)
                if results and results.get("ids") and results["ids"][0]:
                    meta = results["metadatas"][0][0]
                    sim = results["distances"][0][0] if "distances" in results else 0.0
                    if sim >= self.threshold:
                        logger.info("🔁 Found similar embedding in Chroma (%.2f)", sim)
                        return {
                            "source": "chroma",
                            "similarity": sim,
                            "diagnosis": meta.get("diagnosis"),
                            "confidence": meta.get("confidence"),
                            "timestamp": meta.get("timestamp"),
                        }
            except Exception as e:
                logger.warning("ChromaDB lookup failed: %s", e)

        # --- 2️⃣ Fallback: local JSON similarity search ---
        if not self.embeddings:
            logger.info("No embeddings in local JSON memory.")
            return None

        best_score, best_key = 0.0, None
        for key, entry in self.embeddings.items():
            prev_emb = self._from_jsonable(entry.get("embedding", []))
            sim = self._cosine_similarity(emb, prev_emb)
            if sim > best_score:
                best_score, best_key = sim, key

        if best_key and best_score >= self.threshold:
            logger.info("🧠 Found similar local image (%.2f similarity)", best_score)
            entry = self.embeddings[best_key]
            return {
                "source": "json",
                "similarity": best_score,
                "diagnosis": entry["diagnosis"],
                "confidence": entry["confidence"],
                "timestamp": entry["timestamp"],
            }

        return None

    # --------------------------------------------------------
    def store_diagnosis(self, image_path: str, diagnosis: str, confidence: float):
        """
        Store a new diagnosis result (embedding + metadata).
        Always stores to JSON, and to ChromaDB if available.
        """
        emb = self.extractor.extract_from_path(image_path)
        entry_id = f"img_{datetime.now():%Y%m%d_%H%M%S}"
        meta = {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }

        # Store to ChromaDB
        if self._has_chroma:
            try:
                self.collection.add(
                    ids=[entry_id],
                    embeddings=[emb.tolist()],
                    metadatas=[meta],
                )
                logger.info("💾 Stored embedding in ChromaDB: %s (%.2f%%)", diagnosis, confidence * 100)
            except Exception as e:
                logger.warning("ChromaDB store failed: %s", e)

        # Store to JSON backup
        record = {**meta, "embedding": self._to_jsonable(emb)}
        self.embeddings[entry_id] = record
        _save_json(EMBED_FILE, self.embeddings)
        logger.info("📦 Stored local backup for: %s (%.2f%%)", diagnosis, confidence * 100)

    # --------------------------------------------------------
    def store_interaction(self, user_query: str, system_reply: str):
        """
        Store conversational history for traceability.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "reply": system_reply,
        }
        self.history.setdefault("interactions", []).append(record)
        _save_json(HISTORY_FILE, self.history)
        logger.info("💬 Stored conversation exchange.")

    # --------------------------------------------------------
    def list_recent_diagnoses(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Return recent stored diagnoses.
        """
        if self._has_chroma:
            try:
                results = self.collection.peek(limit)
                if results and results.get("metadatas"):
                    recent = results["metadatas"][:limit]
                    return [
                        {
                            "diagnosis": r.get("diagnosis"),
                            "confidence": r.get("confidence"),
                            "timestamp": r.get("timestamp"),
                        }
                        for r in recent
                    ]
            except Exception as e:
                logger.warning("Failed to peek ChromaDB: %s", e)

        # Fallback to local JSON
        all_items = list(self.embeddings.items())
        sorted_items = sorted(all_items, key=lambda kv: kv[1]["timestamp"], reverse=True)
        return [
            {
                "image_id": k,
                "diagnosis": v.get("diagnosis", ""),
                "confidence": v.get("confidence", 0.0),
                "timestamp": v.get("timestamp"),
            }
            for k, v in sorted_items[:limit]
        ]

    def reset_memory(self):
        """Completely clear the embedding memory for a fresh session."""
        # Clear on-disk DB and attempt to reset in-memory chroma runtime
        clear_chroma_runtime()
        # Reset local JSON backup structures
        self.embeddings = {}
        self.history = {}
        _save_json(EMBED_FILE, self.embeddings)
        _save_json(HISTORY_FILE, self.history)
        logger.info("🧹 Memory database reset successfully.")



# ============================================================
# CLI Test
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SmartFarm Hybrid Memory Test")
    parser.add_argument("--image", "-i", type=str, help="Path to leaf image")
    args = parser.parse_args()

    ms = MemoryService()

    if args.image:
        prev = ms.get_previous_diagnosis(args.image)
        if prev:
            print(
                f"🧠 Found similar ({prev['similarity']*100:.1f}% match, {prev['source']}): {prev['diagnosis']}"
            )
        else:
            print("No match found — storing new diagnosis...")
            ms.store_diagnosis(args.image, "Potato___Late_blight", 0.78)
            print("Stored successfully!")

    print("\nRecent records:")
    for rec in ms.list_recent_diagnoses():
        print(f" - {rec['diagnosis']} ({rec['confidence']*100:.1f}%)")
