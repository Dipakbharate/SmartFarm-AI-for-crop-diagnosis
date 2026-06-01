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
def disease_explanation_prompt(label: str, confidence: float, user_crop: str = None) -> str:
    """
    Build a detailed, structured instruction for Gemini/Grok to explain
    the CNN-detected disease in a specific farmer-friendly format.
    """
    crop_info = f"Crop: {user_crop}" if user_crop else "Auto-detect crop from label"
    return f"""
You are SmartFarm AI, an expert agricultural assistant.
A plant image has been analyzed and identified as: {label}
{crop_info}
Confidence: {confidence * 100:.2f}%

Please provide a detailed diagnosis report using the EXACT structure below:

**[Disease Name]** is a disease that affects **[Crop Name]** plants, causing [brief description of impact]. It's usually caused by a combination of factors, including:

- **Fungal infections** (such as [examples])
- **Bacterial infections**
- **Environmental stress** (like [examples])
- **Nutrient deficiencies**
- **Pest infestations** (like [examples])

Here are 3 simple treatment steps for farmers:

1. **[Step 1 Title]**: [Detailed description]
2. **[Step 2 Title]**: [Detailed description]
3. **[Step 3 Title]**: [Detailed description]

Remember, prevention is key. Regularly monitoring your **[Crop Name]** plants, maintaining good hygiene, and providing optimal growing conditions can help reduce the risk of **[Disease Name]**.

Tone: Professional, helpful, and farmer-friendly.
"""

# ============================================================
# 2️⃣ Gemini Vision Prompt
# ============================================================
def vision_analysis_prompt(crop_hint: str = None) -> str:
    """
    Instruction for Gemini Vision model when analyzing an image directly.
    """
    hint = f" The user identifies this as a {crop_hint} plant." if crop_hint else ""
    return (
        f"Analyze this plant leaf image carefully.{hint} "
        "Identify the crop type, any disease symptoms, and estimate severity. "
        "Provide a detailed description of the findings including symptoms and potential causes."
    )

# ============================================================
# 3️⃣ Grok Refinement Prompt
# ============================================================
def refinement_prompt(text: str, user_crop: str = None) -> str:
    """
    Build a prompt for Grok to refine a diagnosis into the 
    standardized structured farmer language.
    """
    crop_info = f"Crop: {user_crop}" if user_crop else "Identify crop from diagnosis data"
    return f"""
You are SmartFarm AI. Refine the following plant diagnosis into a structured, farmer-friendly report.
{crop_info}
Use the EXACT structure below:

**[Disease Name]** is a disease that affects **[Crop Name]** plants, causing [brief description of impact]. It's usually caused by a combination of factors, including:

- **Fungal infections** (such as [examples])
- **Bacterial infections**
- **Environmental stress** (like [examples])
- **Nutrient deficiencies**
- **Pest infestations** (like [examples])

Here are 3 simple treatment steps for farmers:

1. **[Step 1 Title]**: [Detailed description]
2. **[Step 2 Title]**: [Detailed description]
3. **[Step 3 Title]**: [Detailed description]

Remember, prevention is key. Regularly monitoring your **[Crop Name]** plants, maintaining good hygiene, and providing optimal growing conditions can help reduce the risk of **[Disease Name]**.

Diagnosis Data:
{text}
"""


