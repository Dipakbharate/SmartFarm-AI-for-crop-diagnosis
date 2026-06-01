"""
services/memory_service.py
------------------------------------------------------------
Simplified Memory Service for SmartFarm AI.

Responsibilities:
1️⃣ Store and retrieve image embeddings and past diagnoses (Caching).
2️⃣ Use ChromaDB PersistentClient for fast vector search.
3️⃣ Fall back to local JSON memory when ChromaDB is unavailable.

Author: SmartFarm AI Team
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import chromadb
from core.feature_extractor import FeatureExtractor

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = BASE_DIR / "data" / "memory"
EMBED_FILE = MEMORY_DIR / "embeddings.json"

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

def clear_chroma_runtime(memory_db_dir: str | None = None):
    """Forcefully reset on-disk memory_db and clear runtime state."""
    import shutil, time
    if memory_db_dir is None:
        memory_db_dir = Path(__file__).resolve().parents[1] / "data" / "memory_db"
    else:
        memory_db_dir = Path(memory_db_dir)
    
    # Attempt to clear directory with retries (in case of file locks)
    for _ in range(3):
        try:
            shutil.rmtree(memory_db_dir, ignore_errors=True)
            time.sleep(0.1)
            memory_db_dir.mkdir(parents=True, exist_ok=True)
            break
        except Exception:
            time.sleep(0.5)
            
    logger.info("🧹 Memory database cleared.")

# ============================================================
# Memory Service
# ============================================================
class MemoryService:
    """
    Focused interface for SmartFarm AI's result caching.
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.extractor = FeatureExtractor.get_instance()
        self._has_chroma = False
        self._init_chromadb()
        self.embeddings = _load_json(EMBED_FILE)

    def _init_chromadb(self):
        try:
            chroma_path = Path(__file__).resolve().parents[1] / "data" / "memory_db"
            self.client = chromadb.PersistentClient(path=str(chroma_path))
            self.collection = self.client.get_or_create_collection("image_memory")
            self._has_chroma = True
            logger.info("🧠 ChromaDB initialized for result caching.")
        except Exception as e:
            logger.warning("⚠️ ChromaDB unavailable, using JSON fallback: %s", e)

    def get_previous_diagnosis(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Check if similar image has been seen before."""
        try:
            emb = self.extractor.extract_from_path(image_path)
            
            if self._has_chroma:
                results = self.collection.query(query_embeddings=[emb.tolist()], n_results=1)
                if results and results.get("ids") and results["ids"][0]:
                    dist = results.get("distances", [[1.0]])[0][0]
                    sim = 1.0 - dist # Chroma uses distance (0 is identical)
                    if sim >= self.threshold:
                        meta = results["metadatas"][0][0]
                        logger.info("🔁 Found similar case in Chroma (sim=%.2f)", sim)
                        return {**meta, "similarity": sim}

            # JSON fallback
            best_score, best_entry = 0.0, None
            for entry in self.embeddings.values():
                prev_emb = np.array(entry.get("embedding", []), dtype=np.float32)
                # Cosine similarity
                norm_a = np.linalg.norm(emb) + 1e-9
                norm_b = np.linalg.norm(prev_emb) + 1e-9
                sim = float(np.dot(emb, prev_emb) / (norm_a * norm_b))
                
                if sim > best_score:
                    best_score, best_entry = sim, entry

            if best_entry and best_score >= self.threshold:
                logger.info("🧠 Found similar case in JSON (sim=%.2f)", best_score)
                return {**best_entry, "similarity": best_score}

        except Exception as e:
            logger.error("Error in memory lookup: %s", e)
        return None

    def store_diagnosis(self, image_path: str, diagnosis: str, confidence: float, diagnosis_text: str):
        """Store new diagnosis in cache."""
        try:
            emb = self.extractor.extract_from_path(image_path)
            entry_id = f"img_{datetime.now():%Y%m%d_%H%M%S}"
            meta = {
                "diagnosis": diagnosis,
                "confidence": confidence,
                "diagnosis_text": diagnosis_text,
                "timestamp": datetime.now().isoformat(),
            }

            if self._has_chroma:
                self.collection.add(
                    ids=[entry_id],
                    embeddings=[emb.tolist()],
                    metadatas=[meta],
                )

            # JSON backup
            self.embeddings[entry_id] = {**meta, "embedding": emb.tolist()}
            _save_json(EMBED_FILE, self.embeddings)
            logger.info("💾 Saved diagnosis to cache: %s", diagnosis)
        except Exception as e:
            logger.warning("Failed to store diagnosis in cache: %s", e)

    def reset_memory(self):
        """Clear all memory."""
        # 1. Try to delete the collection via the client (more effective than just deleting files)
        if self._has_chroma:
            try:
                self.client.delete_collection("image_memory")
                self.collection = self.client.get_or_create_collection("image_memory")
                logger.info("🗑️ ChromaDB collection deleted and recreated.")
            except Exception as e:
                logger.warning(f"Failed to delete collection: {e}")
        
        # 2. Clear on-disk files as fallback
        clear_chroma_runtime()
        
        # 3. Reset JSON memory
        self.embeddings = {}
        _save_json(EMBED_FILE, {})
        logger.info("🧹 All memory reset.")
