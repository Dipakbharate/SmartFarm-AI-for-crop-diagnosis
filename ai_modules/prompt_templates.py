"""
ai_modules/prompt_templates.py
------------------------------------------------------------
Centralized repository of text and multimodal prompts
used by SmartFarm AI's LLMs (Gemini, Grok, etc.).

Author: SmartFarm AI Team
"""

# ============================================================
# 1️⃣ CNN → Gemini Text Prompt
# ============================================================
def disease_explanation_prompt(label: str, confidence: float) -> str:
    """
    Build a concise instruction for Gemini LLM to explain
    the CNN-detected disease in farmer-friendly language.
    """
    return f"""
You are SmartFarm AI, a helpful agricultural assistant.

A CNN model has analyzed a plant image and predicted:
  • Disease: {label}
  • Confidence: {confidence * 100:.2f}%

Explain clearly what this disease is, what causes it,
and give short, actionable treatment steps a farmer can follow.
Avoid jargon and keep it under 6 sentences.
"""

# ============================================================
# 2️⃣ Gemini Vision Prompt
# ============================================================
def vision_analysis_prompt() -> str:
    """
    Instruction for Gemini Vision model when analyzing an image directly.
    """
    return (
        "Analyze this plant leaf image carefully. "
        "Identify the crop type, any disease symptoms, and estimate severity. "
        "If healthy, say 'Healthy plant'. "
        "If diseased, describe the most probable disease and confidence level."
    )

# ============================================================
# 3️⃣ Grok Refinement Prompt
# ============================================================
def refinement_prompt(text: str) -> str:
    """
    Build a prompt for Grok (or any LLM) to refine a diagnosis
    into natural, clear farmer language.
    """
    return f"""
Refine the following plant diagnosis into a short, clear summary suitable for a farmer.
Add one practical tip if relevant.

Diagnosis:
{text}
"""

# ============================================================
# 4️⃣ ChromaDB / Knowledge Retrieval Prompt (optional)
# ============================================================
def rag_query_prompt(query: str) -> str:
    """
    Used when retrieving info from ChromaDB knowledge base.
    Helps the LLM interpret the search results for the farmer.
    """
    return f"""
A farmer asked: "{query}"
Based on the knowledge base, summarize what crop disease matches this,
and list treatments or preventive actions.
Keep it factual and simple.
"""

