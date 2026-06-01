"""
services/hybrid_predictor.py
------------------------------------------------------------
Main orchestrator for SmartFarm AI.

Pipeline:
1️⃣ CNN model predicts disease.
2️⃣ Check Image Cache (ChromaDB) for similar recent results.
3️⃣ Trigger Gemini Vision if CNN confidence is low or forced.
4️⃣ Grok LLM always generates the final farmer-friendly response.

Author: SmartFarm AI Team
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np

# ---------------- Local project imports ----------------
from core.predict import predict_image
from ai_modules.llm_client import (
    grok_disease_response,     # CNN explanation (Grok)
    gemini_vision_response,    # Vision analysis (Gemini)
    grok_refine_response,      # Refinement for Vision
)
from services.memory_service import MemoryService

# --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Confidence thresholds
CNN_CONF_THRESHOLD = 0.75
VISION_CONF_THRESHOLD = 0.50

def _build_response(stage: str, message: str, meta: Optional[Dict[str, Any]] = None):
    return {
        "stage": stage,
        "message": message,
        "metadata": meta or {},
        "timestamp": datetime.now().isoformat(),
    }

def hybrid_predict(image_path: str, user_crop: Optional[str] = None, force_gemini: bool = False) -> Dict[str, Any]:
    """
    Main inference pipeline: CNN -> Cache -> Gemini (conditional) -> Grok (Final).
    """
    path = Path(image_path)
    if not path.exists():
        return _build_response("error", "⚠️ Image not found. Please upload a valid leaf photo.")

    memory = MemoryService()
    
    # 1. Check Image Cache First (Bypass if forced)
    if not force_gemini:
        try:
            cached = memory.get_previous_diagnosis(str(path))
            if cached and cached.get("similarity", 0) > 0.98:
                logger.info("⚡ Using high-confidence cached result.")
                # Retrieve the full saved explanation
                saved_text = cached.get("diagnosis_text", "Previous diagnosis found.")
                
                # If the cached entry is stale (missing explanation), skip it and run fresh
                if saved_text == "Previous diagnosis found.":
                    logger.info("⚠️ Cached entry is stale. Running fresh analysis.")
                else:
                    return _build_response("cached_result", saved_text, cached)
        except Exception as e:
            logger.warning("Cache lookup failed: %s", e)

    # 2. CNN Prediction (Always runs)
    try:
        cnn_label, cnn_conf, topk = predict_image(str(path))
        logger.info(f"🧠 CNN: {cnn_label} ({cnn_conf:.2%})")
    except Exception as e:
        logger.error(f"CNN failed: {e}")
        cnn_label, cnn_conf, topk = "Unknown", 0.0, []

    # 3. Determine if Gemini is needed
    use_gemini = force_gemini or (cnn_conf < CNN_CONF_THRESHOLD)
    
    final_output = ""
    stage = "cnn_grok"

    if use_gemini:
        try:
            logger.info("🔍 Triggering Gemini Vision analysis...")
            vision_result = gemini_vision_response(str(path), user_crop or cnn_label)
            
            # Use Grok to refine the Gemini output for the farmer
            desc = vision_result.get("description", "No description available.")
            
            # Stage is now definitely Gemini+Grok
            stage = "gemini_vision_grok"
            final_output = grok_refine_response(desc, user_crop)
            
            # Store result in cache if confidence is okay
            if vision_result.get("confidence", 0) > VISION_CONF_THRESHOLD:
                memory.store_diagnosis(
                    str(path), 
                    vision_result.get("disease", "Unknown"), 
                    vision_result.get("confidence", 0),
                    final_output
                )
                
        except Exception as e:
            logger.warning(f"Gemini/Grok refinement failed: {e}")
            if force_gemini:
                # If forced, we stay in this stage but show what we can
                stage = "gemini_vision_grok"
                final_output = f"⚠️ Vision Analysis encountered an issue: {e}. Please try again later."
            else:
                # If not forced, fallback to CNN
                use_gemini = False

    if not use_gemini or not final_output:
        # 4. Final Grok Output for CNN result
        try:
            final_output = grok_disease_response(cnn_label, cnn_conf, topk, user_crop)
            memory.store_diagnosis(str(path), cnn_label, cnn_conf, final_output)
        except Exception as e:
            logger.error(f"Final Grok stage failed: {e}")
            final_output = f"The AI identifies this as **{cnn_label}**. Please consult a local expert for treatment."

    return _build_response(
        stage, 
        final_output, 
        {
            "cnn_label": cnn_label, 
            "cnn_conf": cnn_conf, 
            "use_gemini": use_gemini,
            "force_gemini": force_gemini
        }
    )
