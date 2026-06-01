
"""
ai_modules/llm_client.py
------------------------------------------------------------
Handles communication with external Large Language Models (LLMs)
like Gemini Vision and Grok.

    - CNN prediction explanation → Expert AI
    - Vision analysis → Advanced Vision
    - Refinement → Expert AI

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
import google.generativeai as genai


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------
# Prompt imports
# ------------------------------------------------------------
from ai_modules.prompt_templates import (
    disease_explanation_prompt,
    vision_analysis_prompt,
    refinement_prompt,
)

# ------------------------------------------------------------
# Setup Gemini
# ------------------------------------------------------------
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE"))
    google_model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    google_model = None
    print(f"⚠️ Gemini model initialization failed: {e}")

# ------------------------------------------------------------
# Optional LLM libraries
# ------------------------------------------------------------
try:
    from groq import Groq
    _HAS_GROK = True
except Exception:
    _HAS_GROK = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini Vision API configured.")
else:
    logger.warning("⚠️ Gemini Vision API not configured. Using mock responses.")


# ============================================================
# 1️⃣ CNN → Grok Explanation
# ============================================================
def grok_disease_response(label: str, confidence: float, topk: list, user_crop: str = None) -> str:
    """Generate a natural-language explanation for CNN-predicted disease."""
    if not _HAS_GROK or not GROK_API_KEY:
        logger.info("Using mock AI CNN response (offline mode).")
        return (
            f"The plant likely suffers from {label} (confidence {confidence*100:.1f}%). "
            "Ensure proper watering, improve air circulation, and apply a suitable fungicide."
        )

    try:
        client = Groq(api_key=GROK_API_KEY)
        prompt = disease_explanation_prompt(label, confidence, user_crop)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        logger.warning("Grok disease response failed: %s", e)
        if "429" in str(e) or "quota" in str(e).lower():
            time.sleep(30)
            return grok_disease_response(label, confidence, topk)
        return (
            f"The plant likely suffers from {label}. "
            "Maintain optimal soil moisture and apply a protective fungicide."
        )


# ============================================================
# 2️⃣ Gemini Vision (Auto Crop + Disease Detection)
# ============================================================
def gemini_vision_response(image_path: str, user_crop: str = None) -> dict:
    """
    Optimized Gemini Vision: single-call, no quota spam.
    Detects crop + disease + severity + treatment in one request.
    """

    from PIL import Image
    import re
    import google.generativeai as genai

    try:
        image = Image.open(image_path).convert("RGB")
        if image.size[0] > 1024 or image.size[1] > 1024:
            image = image.resize((1024, 1024), Image.Resampling.LANCZOS)

        prompt = vision_analysis_prompt(user_crop)

        # ✅ Single Gemini call for both crop + disease
        response = google_model.generate_content([prompt, image])
        result_text = getattr(response, "text", "").strip()

        # --- parse important parts ---
        def extract(pattern, default="Unknown"):
            m = re.search(pattern, result_text, re.IGNORECASE)
            return m.group(1).strip() if m else default

        crop = extract(r"Crop[:\-–]\s*(.+)", user_crop or "Unknown")
        disease = extract(r"Disease[:\-–]\s*(.+)", "Unknown")
        health = extract(r"Health[:\-–]\s*(.+)", "Unknown")
        conf_str = extract(r"Confidence[:\-–]\s*([0-9\.]+%?|high|medium|low)", "0.7")

        conf_str = conf_str.lower().replace("%", "").strip()
        if "high" in conf_str:
            confidence = 0.9
        elif "medium" in conf_str:
            confidence = 0.7
        elif "low" in conf_str:
            confidence = 0.5
        else:
            try:
                confidence = float(conf_str)
                if confidence > 1:
                    confidence = confidence / 100.0
            except:
                confidence = 0.7

        return {
            "crop": crop,
            "health_status": health,
            "disease": disease,
            "confidence": confidence,
            "description": result_text,
        }

    except Exception as e:
        return {
            "crop": "Unknown",
            "disease": "Error",
            "confidence": 0.0,
            "description": f"❌ Advanced Vision Analysis failed: {str(e)}",
        }


# ============================================================
# 3️⃣ Grok Refinement (Vision / RAG)
# ============================================================
def grok_refine_response(text: str, user_crop: str = None) -> str:
    """Refine a given text into farmer-friendly explanation."""
    prompt = refinement_prompt(text, user_crop)

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

    logger.info("Using offline AI refinement fallback.")
    return f"{text}\n\n👉 Tip: Monitor nearby plants and apply organic fungicide if symptoms spread."


# ============================================================
# Manual Test
# ============================================================
if __name__ == "__main__":
    print("🧠 Testing CNN → Grok text generation...")
    res = grok_disease_response("Potato Late Blight", 0.82, [])
    print("\nResponse:\n", res)

    print("\n🖼️ Testing Gemini Vision analysis...")
    vis = gemini_vision_response("data/uploads/sample_leaf.JPG")
    print("\nVision output:\n", vis)

    print("\n💬 Testing Grok refinement...")
    refined = grok_refine_response("Detected fungal infection on the leaf.")
    print("\nRefined output:\n", refined)

