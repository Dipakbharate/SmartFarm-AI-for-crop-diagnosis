"""
ai_modules/llm_client.py
------------------------------------------------------------
Handles communication with external Large Language Models (LLMs)
like Gemini Vision and Grok.

Modified pipeline:
    - CNN prediction explanation → Grok
    - Vision analysis → Gemini Vision
    - Refinement / fallback → Grok

Environment Variables:
    GEMINI_API_KEY   -> for Google Gemini Vision API
    GROK_API_KEY     -> for Grok (X.AI / Groq API)

If API keys are missing, mock responses are used for offline testing.

Author: SmartFarm AI Team
"""

import os
import logging
import time
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import prompt templates
from ai_modules.prompt_templates import (
    disease_explanation_prompt,
    vision_analysis_prompt,
    refinement_prompt,
)

# ------------------------------------------------------------
# Optional LLM libraries
# ------------------------------------------------------------
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

try:
    from groq import Groq  # Grok client library
    _HAS_GROK = True
except Exception:
    _HAS_GROK = False

# ------------------------------------------------------------
# Setup & Logging
# ------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

if GEMINI_API_KEY and _HAS_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini Vision API configured.")
else:
    logger.warning("⚠️ Gemini Vision API not configured. Using mock responses.")

# ============================================================
# 1️⃣ CNN → Grok LLM (text-based disease explanation)
# ============================================================
def grok_disease_response(label: str, confidence: float, topk: list) -> str:
    """
    Uses Grok LLM to generate a natural-language explanation
    for the CNN-predicted disease.
    """
    if not _HAS_GROK or not GROK_API_KEY:
        logger.info("Using mock Grok CNN response (offline mode).")
        return (
            f"The plant likely suffers from {label} (confidence {confidence*100:.1f}%). "
            "Ensure proper watering, improve air circulation, and apply a suitable fungicide."
        )

    try:
        client = Groq(api_key=GROK_API_KEY)
        prompt = (
            f"The CNN model identified the disease as '{label}' "
            f"with {confidence*100:.1f}% confidence.\n\n"
            "Explain what this disease is, its causes, and provide 3 clear treatment "
            "steps suitable for farmers in simple language."
        )
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        logger.warning("Grok disease response failed: %s", e)
        # Retry once if rate-limited or transient error
        if "429" in str(e) or "quota" in str(e).lower():
            logger.info("Waiting 40s before retrying Grok...")
            time.sleep(40)
            return grok_disease_response(label, confidence, topk)

        # Fallback text
        return (
            f"The plant likely suffers from {label}. "
            "Maintain optimal soil moisture and apply a protective fungicide."
        )

# ============================================================
# 2️⃣ Gemini Vision (image-based analysis)
# ============================================================
def gemini_vision_response(image_path: str) -> Dict[str, Any]:
    """
    Analyze an image using Gemini Vision model (multimodal input).
    Returns dict: {'description': str, 'confidence': float}
    """
    if GEMINI_API_KEY and _HAS_GEMINI:
        try:
            model = genai.GenerativeModel("gemini-1.5-pro-vision")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            prompt = vision_analysis_prompt()
            response = model.generate_content([prompt, image_bytes])
            text = response.text.strip() if response and hasattr(response, "text") else "No description."
            return {"description": text, "confidence": 0.8}
        except Exception as e:
            logger.warning("Gemini Vision API failed: %s", e)

    # Offline fallback
    logger.info("Using mock Gemini Vision response (offline mode).")
    return {
        "description": "Detected leaf discoloration consistent with fungal infection (possible Late Blight).",
        "confidence": 0.78,
    }

# ============================================================
# 3️⃣ Grok Refinement (used for Vision / RAG outputs)
# ============================================================
def grok_refine_response(text: str) -> str:
    """
    Uses Grok LLM to refine a given text (e.g. from Gemini Vision or RAG)
    into a clear, farmer-friendly explanation.
    """
    prompt = refinement_prompt(text)

    if GROK_API_KEY and _HAS_GROK:
        try:
            client = Groq(api_key=GROK_API_KEY)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400,
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            logger.warning("Grok refinement failed: %s", e)

    # Offline fallback
    logger.info("Using mock Grok refinement (offline mode).")
    return (
        f"{text}\n\n👉 Tip: Monitor nearby plants and spray preventive fungicides if symptoms spread."
    )

# ============================================================
# Manual test entry
# ============================================================
if __name__ == "__main__":
    print("🧠 Testing CNN → Grok text generation...")
    res = grok_disease_response("Potato Late Blight", 0.79, [])
    print("\nResponse:\n", res)

    print("\n🖼️ Testing Gemini Vision analysis...")
    vis = gemini_vision_response("data/uploads/sample_leaf.JPG")
    print("\nVision output:\n", vis)

    print("\n💬 Testing Grok refinement...")
    refined = grok_refine_response("Detected fungal infection on the leaf.")
    print("\nRefined output:\n", refined)
