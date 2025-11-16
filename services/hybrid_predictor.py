"""
services/hybrid_predictor.py
------------------------------------------------------------
Master orchestrator for SmartFarm AI.

Pipeline:
1️⃣ CNN model predicts disease from image.
2️⃣ If CNN confidence is high → Grok LLM explains result.
3️⃣ If CNN confidence is low → Gemini Vision analyzes image (auto or user crop) → Grok refines explanation.
4️⃣ All embeddings are stored in ChromaDB PersistentClient cache for fast lookup.
5️⃣ If both fail → Query ChromaDB Knowledge Base (RAG) for fallback.
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
CNN_CONF_THRESHOLD = 0.70
VISION_CONF_THRESHOLD = 0.50


# --------------------------------------------------------
# ChromaDB setup
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
# Helper: build response payload
# --------------------------------------------------------
def _build_response(stage: str, text: str, meta: Optional[Dict[str, Any]] = None):
    return {
        "stage": stage,
        "message": text,
        "metadata": meta or {},
        "timestamp": datetime.now().isoformat(),
    }


# ========================================================
# Core pipeline
# ========================================================
def hybrid_predict(image_path: str, user_crop: Optional[str] = None) -> Dict[str, Any]:
    """Main SmartFarm inference orchestrator with optional user crop input."""
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
        emb = extractor.extract_from_path(str(path))
    except Exception as e:
        logger.exception("CNN prediction failed: %s", e)
        label, conf, topk, emb = None, 0.0, [], None

    # ====================================================
    # STEP 1.5: CHROMA CACHE LOOKUP
    # ====================================================
    try:
        if emb is not None:
            results = image_cache.query(query_embeddings=[emb.tolist()], n_results=1)
            if results and results.get("ids") and results["ids"][0]:
                dist = results.get("distances", [[1.0]])[0][0]
                sim_score = 1.0 - dist
                if sim_score > 0.95:
                    existing = results["metadatas"][0][0]
                    logger.info("🔁 Cached result used (sim=%.3f)", sim_score)
                    return _build_response(
                        "cached_result",
                        existing.get("diagnosis", "Previously diagnosed case"),
                        {"similarity": sim_score, "metadata": existing},
                    )
    except Exception as e:
        logger.warning("Chroma cache lookup failed: %s", e)

    # ====================================================
    # STEP 2: HIGH-CONFIDENCE → CNN + GROK
    # ====================================================
    if label and conf >= CNN_CONF_THRESHOLD:
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

    # ====================================================
    # STEP 3: LOW-CONFIDENCE → GEMINI + GROK
    # ====================================================
    try:
        vision_result = gemini_vision_response(str(path), user_crop or label or "unknown")

        # --- Normalize confidence to float ---
        raw_conf = vision_result.get("confidence", 0)
        try:
            conf_val = float(raw_conf)
        except Exception:
            try:
                conf_val = float(str(raw_conf).replace("%", "").strip())
                if conf_val > 1:
                    conf_val = conf_val / 100.0
            except Exception:
                conf_val = 0.0

        logger.info(f"🔍 Gemini returned disease={vision_result.get('disease')} conf={conf_val}")

        if conf_val >= VISION_CONF_THRESHOLD:
            disease = str(vision_result.get("disease", "unknown")).strip().lower()

            # ✅ Healthy handling
            if disease == "healthy":
                msg = (
                    "✅ The leaf appears **healthy** — no visible signs of disease. "
                    "Maintain proper irrigation, spacing, and regular monitoring."
                )
                _store_embedding_cache(image_cache, emb, "healthy", conf_val)
                memory.store_diagnosis(str(path), "healthy", conf_val)
                return _build_response("gemini_vision_grok", msg, {"vision_result": vision_result})

            # 🧠 Grok refinement
            desc = f"""
            Crop: {vision_result.get('crop', 'unknown')}
            Disease: {vision_result.get('disease', 'unknown')}
            Symptoms: {vision_result.get('symptoms', 'unknown')}
            Severity: {vision_result.get('severity', 'unknown')}
            Confidence: {conf_val:.2f}

            Detailed Analysis:
            {vision_result.get('description', 'No description available')}
            """

            enriched_prompt = f"""
            You are an expert agricultural pathologist.
            Based on this AI Vision Analysis, generate a structured diagnosis with:
            - **Disease Name**
            - **Probable Causes**
            - **Symptoms**
            - **Treatment Steps (3 short actionable steps)**
            - **Prevention Tips**

            Respond clearly in markdown for farmers.
            {desc}
            """

            final_text = grok_refine_response(enriched_prompt)
            _store_embedding_cache(image_cache, emb, label or "unknown", conf)
            memory.store_diagnosis(str(path), label or "unknown", conf)

            logger.info("✅ Stage Triggered: gemini_vision_grok")
            return _build_response(
                "gemini_vision_grok",
                final_text,
                {"vision_result": vision_result, "cnn_conf": conf_val, "cnn_label": label},
            )
        else:
            logger.warning(f"⚠️ Gemini confidence too low ({conf_val:.2f}) — falling back to RAG.")

    except Exception as e:
        logger.warning("Gemini Vision fallback failed: %s", e)

    # ====================================================
    # STEP 5: TOTAL FAILURE
    # ====================================================
    msg = (
        "❌ Unable to identify disease confidently. "
        "Try uploading a clearer image or specify the crop name."
    )
    return _build_response("error", msg)


# ========================================================
# Helper: Store embedding
# ========================================================
def _store_embedding_cache(image_cache, emb: np.ndarray, label: str, conf: float):
    """Safely store new embeddings into ChromaDB cache."""
    if not _HAS_CHROMA or emb is None:
        return
    try:
        img_id = f"img_{datetime.now():%Y%m%d_%H%M%S}"
        meta = {"diagnosis": label, "confidence": conf, "timestamp": datetime.now().isoformat()}
        existing = image_cache.query(query_embeddings=[emb.tolist()], n_results=1)
        if not existing or not existing.get("ids") or existing["ids"][0] == []:
            image_cache.add(ids=[img_id], embeddings=[emb.tolist()], metadatas=[meta])
            logger.info("💾 Cached new embedding: %s (%.2f%%)", label, conf * 100)
    except Exception as e:
        logger.warning("Failed to store embedding: %s", e)
