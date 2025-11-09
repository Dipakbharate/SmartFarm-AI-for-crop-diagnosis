"""
app.py
------------------------------------------------------------
🌾 SmartFarm AI Web App (Streamlit Frontend)

Modules Used:
 - services.hybrid_predictor → orchestrates CNN + LLM + RAG
 - utils.image_utils → preprocesses uploaded images
 - ai_modules.llm_client → Grok & Gemini Vision integrations

Features:
1️⃣ Upload a plant leaf image
2️⃣ Run hybrid prediction (CNN + Grok)
3️⃣ View diagnosis, confidence, and treatment recommendations
4️⃣ Optional: View previous diagnoses (via Memory/ChromaDB)
5️⃣ Future tab: Crop recommendation (from RAG KB)

Author: SmartFarm AI Team
"""

import os
import time
import streamlit as st
from PIL import Image

# ---------------- Local imports ----------------
from services.hybrid_predictor import hybrid_predict
from services.memory_service import MemoryService

# ------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="🌾 SmartFarm AI",
    page_icon="🌱",
    layout="wide",
)

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.title("🌿 SmartFarm AI Assistant")
st.sidebar.markdown("A CNN + AI powered crop health diagnosis tool.")
st.sidebar.divider()
if st.sidebar.button("🧹 Reset Memory DB"):
    from services.memory_service import clear_chroma_runtime
    clear_chroma_runtime()
    st.sidebar.success("Memory database reset successfully!")
st.sidebar.markdown("👨‍🌾 **Developed by:** SmartFarm AI Team")
st.sidebar.markdown("📚 Uses CNN + Grok + Gemini Vision + ChromaDB")

# ------------------------------------------------
# Tabs
# ------------------------------------------------
tab1, tab2 = st.tabs(["🩺 Plant Health Diagnosis", "🌾 Crop Recommendations"])

# ------------------------------------------------
# TAB 1: Plant Health Diagnosis
# ------------------------------------------------
with tab1:
    st.header("🩺 Upload a Leaf Image for Diagnosis")
    st.markdown("Upload a clear photo of your crop’s leaf to identify diseases and get treatment advice.")

    
    from services.memory_service import MemoryService
    from core.feature_extractor import FeatureExtractor

        # ✅ Only one uploader (with a unique key)
    uploaded_image = st.file_uploader(
        "📤 Choose a leaf image", 
        type=["jpg", "jpeg", "png"], 
        key="leaf_upload"
    )

   

    analyze_button = st.button("🔍 Analyze")



    if uploaded_image and analyze_button:
        with st.spinner("Analyzing image... please wait ⏳"):
            # Save uploaded image to temp path
            upload_dir = "data/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, uploaded_image.name)

            # Convert to PIL and save
            image = Image.open(uploaded_image).convert("RGB")
            image.save(image_path)

            # Run hybrid prediction
            result = hybrid_predict(image_path)

            st.success("✅ Analysis complete!")

            # Display image & result
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Uploaded Leaf", use_container_width=True)
            with col2:
                st.subheader("🌿 Diagnosis Result")
                st.markdown(f"**Stage:** `{result['stage']}`")
                st.markdown("### 🧠 Analysis Summary")
                st.write(result["message"])
                if "cnn_conf" in result.get("metadata", {}):
                    st.progress(result["metadata"]["cnn_conf"])
                if result.get("metadata"):
                    with st.expander("🔍 Technical Metadata"):
                        st.json(result["metadata"])

            # Store result in memory (if available)
            try:
                memory = MemoryService()
                if "cnn_label" in result["metadata"]:
                    memory.store_diagnosis(
                        image_path,
                        result["metadata"]["cnn_label"],
                        result["metadata"]["cnn_conf"],
                    )
            except Exception as e:
                st.warning(f"⚠️ Memory storage skipped: {e}")

# ------------------------------------------------
# TAB 2: Crop Recommendations
# ------------------------------------------------
with tab2:
    st.header("🌾 Crop Recommendation Assistant")
    st.markdown(
        "Ask SmartFarm AI for crop suggestions based on your soil, region, or weather conditions."
    )

    query = st.text_area("💬 Enter your query (e.g. 'Best crops for clay soil in summer')", height=100)
    recommend_button = st.button("🌱 Get Recommendations")

    if recommend_button and query.strip():
        with st.spinner("Fetching recommendations from knowledge base..."):
            from rag.query_kb import query_knowledge_base

            kb_results = query_knowledge_base(query)
            if kb_results:
                top = kb_results[0]
                st.success("✅ Recommendations Ready!")
                st.markdown(f"**🌾 Crop:** {top['crop']}")
                st.markdown(f"**🦠 Disease:** {top['disease']}")
                st.markdown(f"**🔍 Symptoms:** {top['symptoms']}")
                st.markdown(f"**💡 Treatment:** {top['treatment']}")
                st.markdown(f"**🌱 Recommended Crops:** {top['recommended']}")
            else:
                st.warning("No suitable recommendations found in the knowledge base.")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>🌿 SmartFarm AI © 2025 | Powered by CNN + Grok + Gemini + ChromaDB</p>",
    unsafe_allow_html=True,
)
