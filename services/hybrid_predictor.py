"""
services/hybrid_predictor.py
------------------------------------------------------------
Master orchestrator for SmartFarm AI.

Pipeline:
1️⃣ CNN model predicts disease from image.
2️⃣ If CNN confidence is high → Grok LLM explains result.
3️⃣ If CNN confidence is low → Gemini Vision analyzes image → Grok refines explanation.
4️⃣ All embeddings are stored in ChromaDB PersistentClient cache for fast lookup.
5️⃣ If both fail → Query ChromaDB Knowledge Base (RAG) for a fallback answer.
6️⃣ MemoryService (JSON) used only as backup when ChromaDB unavailable.

Author: SmartFarm AI Team
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np

# ---------------- Local project imports ----------------
from core.predict import predict_image
from core.feature_extractor import FeatureExtractor
from rag.query_kb import query_knowledge_base
from ai_modules.llm_client import (
    grok_disease_response,     # CNN explanation (Grok)
    gemini_vision_response,    # Vision analysis (Gemini)
    grok_refine_response,      # Refinement for Vision / RAG
)
from services.memory_service import MemoryService

import chromadb

# --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Confidence thresholds
CNN_CONF_THRESHOLD = 0.85
VISION_CONF_THRESHOLD = 0.70

# --------------------------------------------------------
# ChromaDB setup (Persistent cache for embeddings)
# --------------------------------------------------------
try:
    CHROMA_PATH = Path(__file__).resolve().parents[1] / "data" / "memory_db"
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    image_cache = client.get_or_create_collection("image_memory")
    _HAS_CHROMA = True
    logger.info("✅ ChromaDB PersistentClient initialized at %s", CHROMA_PATH)
except Exception as e:
    _HAS_CHROMA = False
    logger.warning("⚠️ ChromaDB unavailable: %s", e)

# --------------------------------------------------------
# Helper: build user-friendly response payload
# --------------------------------------------------------
def _build_response(stage: str, text: str, meta: Optional[Dict[str, Any]] = None):
    return {
        "stage": stage,      # cnn_grok / gemini_vision_grok / chroma_rag / error
        "message": text,
        "metadata": meta or {},
        "timestamp": datetime.now().isoformat(),
    }


# ========================================================
# Core pipeline
# ========================================================
def hybrid_predict(image_path: str) -> Dict[str, Any]:
    """
    Main inference orchestration.
    Automatically routes low-confidence CNN predictions
    through Gemini Vision → Grok fallback.
    """

    path = Path(image_path)
    if not path.exists():
        msg = f"Image not found: {path}"
        logger.error(msg)
        return _build_response("error", msg)

    extractor = FeatureExtractor.get_instance()
    memory = MemoryService()

    # ====================================================
    # STEP 1: CNN PREDICTION
    # ====================================================
    try:
        label, conf, topk = predict_image(str(path))
        logger.info("🧠 CNN predicted %s (%.2f%%)", label, conf * 100)

        # ✅ compute fresh embedding immediately after CNN
        emb = extractor.extract_from_path(str(path))

    except Exception as e:
        logger.exception("CNN prediction failed: %s", e)
        label, conf, topk, emb = None, 0.0, [], None
    # ====================================================
    # STEP 1.5: CHROMA CACHE LOOKUP
    # ====================================================
    try:
        results = image_cache.query(query_embeddings=[emb.tolist()], n_results=1)
        if results and results.get("ids") and results["ids"][0]:
            # Chroma returns distances (lower = closer); convert to similarity
            dist = None
            if "distances" in results and len(results["distances"]) and len(results["distances"][0]):
                dist = float(results["distances"][0][0])
                sim_score = 1.0 - dist
            else:
                sim_score = 0.0  # unknown; skip cached match

            logger.info("🧮 Chroma distance=%.6f similarity≈%.6f", dist if dist is not None else -1.0, sim_score)

            # only accept extremely high similarity as cached (avoid false positives)
            if sim_score > 0.95:
                existing = results["metadatas"][0][0]
                logger.info("🔁 Found cached similar embedding (sim=%.3f) => using cached_result", sim_score)
                return _build_response(
                    "cached_result",
                    existing.get("diagnosis", "Previously diagnosed case"),
                    {"similarity": sim_score, "metadata": existing},
                )
    except Exception as e:
        logger.warning("Chroma cache lookup failed (non-fatal): %s", e)


    # ====================================================
    # STEP 2: DECISION BASED ON CONFIDENCE
    # ====================================================
    if label and conf >= CNN_CONF_THRESHOLD:
        # ✅ High-confidence → CNN + Grok
        try:
            response = grok_disease_response(label, conf, topk)
            _store_embedding_cache(image_cache, emb, label, conf)
            memory.store_diagnosis(str(path), label, conf)
            return _build_response(
                "cnn_grok",
                response,
                {"cnn_label": label, "cnn_conf": conf, "topk": topk},
            )
        except Exception as e:
            logger.warning("Grok LLM failed after CNN: %s", e)

    else:
        # ⚙️ Low confidence → Gemini Vision → Grok
        try:
            vision_result = gemini_vision_response(str(path))
            if vision_result and vision_result.get("confidence", 0) >= VISION_CONF_THRESHOLD:
                final_text = grok_refine_response(vision_result["description"])
                _store_embedding_cache(image_cache, emb, label or "unknown", conf)
                memory.store_diagnosis(str(path), label or "unknown", conf)
                return _build_response(
                    "gemini_vision_grok",
                    final_text,
                    {"vision_result": vision_result},
                )
        except Exception as e:
            logger.warning("Gemini Vision fallback failed: %s", e)

    # ====================================================
    # STEP 3: CHROMADB KNOWLEDGE BASE (RAG)
    # ====================================================
    try:
        query_text = label if label else "unknown plant disease"
        kb_results = query_knowledge_base(query_text)
        if kb_results:
            top = kb_results[0]
            text = (
                f"🌾 Crop: {top['crop']}\n"
                f"🦠 Disease: {top['disease']}\n"
                f"🔍 Symptoms: {top['symptoms']}\n"
                f"💡 Treatment: {top['treatment']}\n"
                f"🌱 Recommended Crops: {top['recommended']}"
            )
            if emb is not None:
                _store_embedding_cache(image_cache, emb, label, conf)
            return _build_response("chroma_rag", text, {"kb_results": kb_results})
    except Exception as e:
        logger.exception("ChromaDB knowledge fallback failed: %s", e)

    # ====================================================
    # STEP 4: TOTAL FAILURE
    # ====================================================
    msg = (
        "❌ Unable to identify disease confidently. "
        "Please try uploading a clearer image or specify the crop name."
    )
    return _build_response("error", msg)


# ========================================================
# Helper: store embedding to ChromaDB cache
# ========================================================
def _store_embedding_cache(image_cache, emb: np.ndarray, label: str, conf: float):
    """
    Store new embedding into ChromaDB PersistentClient.
    Also keep minimal JSON record through MemoryService.
    """
    if not _HAS_CHROMA:
        return

    try:
        img_id = f"img_{datetime.now():%Y%m%d_%H%M%S}"
        meta = {
            "diagnosis": label,
            "confidence": conf,
            "timestamp": datetime.now().isoformat(),
        }
        image_cache.add(
            ids=[img_id],
            embeddings=[emb.tolist()],
            metadatas=[meta],
        )
        logger.info("💾 Cached embedding to ChromaDB: %s (%.2f%%)", label, conf * 100)
    except Exception as e:
        logger.warning("Failed to store embedding to ChromaDB: %s", e)


# ========================================================
# Manual test
# ========================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    test_image = "data/uploads/sample_leaf.JPG"
    result = hybrid_predict(test_image)
    print("\n=============================")
    print("Stage:", result["stage"])
    print("-----------------------------")
    print(result["message"])
    print("=============================")


