"""
app.py
------------------------------------------------------------
🌾 SmartFarm AI Web App (Streamlit Frontend)

Modules Used:
 - services.hybrid_predictor → orchestrates CNN + LLM + RAG
 - ai_modules.llm_client → Grok & Gemini Vision integrations

Features:
1️⃣ Upload a plant leaf image (with optional crop hint)
2️⃣ Hybrid prediction pipeline (CNN + Grok + Gemini + Chroma)
3️⃣ Automatic crop detection using Gemini Vision
4️⃣ Displays disease, treatment, and prevention details
5️⃣ Crop recommendation assistant via RAG KB

Author: SmartFarm AI Team
"""

import os
import streamlit as st
from PIL import Image
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

# ✅ Memory Reset Button
if st.sidebar.button("🧹 Reset Memory DB"):
    from services.memory_service import clear_chroma_runtime
    try:
        clear_chroma_runtime()
        st.sidebar.success("✅ Memory and cache reset successfully!")
    except Exception as e:
        st.sidebar.error(f"⚠️ Reset failed: {e}")

st.sidebar.markdown("👨‍🌾 **Developed by:** SmartFarm AI Team")
st.sidebar.markdown("📚 Uses CNN + Grok + Gemini Vision + ChromaDB")

# ------------------------------------------------
# Tabs
# ------------------------------------------------
#tab1, tab2 = st.tabs(["🩺 Plant Health Diagnosis","🌾 Crop Recommendations"])
tab1 =st.tabs(["🩺 Plant Health Diagnosis"])[0]


# ------------------------------------------------
# TAB 1: Plant Health Diagnosis
# ------------------------------------------------
with tab1:
    st.header("🩺 Upload a Leaf Image for Diagnosis")
    st.markdown(
        "Upload a clear photo of your crop’s leaf for **AI-powered disease detection and treatment advice.** "
        "Optionally specify the crop type if known — otherwise, Gemini Vision will detect it automatically."
    )

    # Optional crop input
    crop_input = st.text_input(
        "🌱 Optional: Enter Crop Name (e.g. Tomato, Potato, Rice)",
        placeholder="Leave blank for auto-detection",
    )

    uploaded_image = st.file_uploader(
        "📤 Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_upload",
    )

    analyze_button = st.button("🔍 Analyze")

    if uploaded_image and analyze_button:
        with st.spinner("Analyzing image... please wait ⏳"):
            # Save uploaded image
            upload_dir = "data/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, uploaded_image.name)
            image = Image.open(uploaded_image).convert("RGB")
            image.save(image_path)
            

            # Run hybrid AI prediction
            result = hybrid_predict(image_path, user_crop=crop_input.strip() if crop_input else None)

            st.success("✅ Analysis complete!")

            # ---------------- Display results ----------------
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(image, caption="Uploaded Leaf", use_container_width=True)

            with col2:
                st.subheader("🌿 Diagnosis Result")
                st.markdown(f"**Stage:** `{result['stage']}`")

                # Highlight cached results visually
                if result["stage"] == "cached_result":
                    st.info("⚡ Result loaded from SmartFarm memory cache for similar image.")

                st.markdown("### 🧠 Analysis Summary")
                st.markdown(result["message"], unsafe_allow_html=True)

                # Confidence bar (if available)
                meta = result.get("metadata", {})
                if "cnn_conf" in meta:
                    st.progress(meta["cnn_conf"])

                # Technical metadata
                if meta:
                    with st.expander("🔍 Technical Metadata"):
                        st.json(meta)

            # 🧠 Save result in memory (if available)
            try:
                memory = MemoryService()
                if "cnn_label" in meta:
                    memory.store_diagnosis(
                        image_path,
                        meta["cnn_label"],
                        meta.get("cnn_conf", 0.0),
                    )
            except Exception as e:
                st.warning(f"⚠️ Memory storage skipped: {e}")


# ------------------------------------------------
# TAB 2: Crop Recommendations
# ------------------------------------------------

#with tab2:
    #st.header("🌾 Crop Recommendation Assistant")
    #st.markdown(
     #   "Ask SmartFarm AI for crop suggestions based on soil, region, or weather conditions.")

    #query = st.text_area(
        #"💬 Enter your query (e.g. 'Best crops for black cotton soil in Maharashtra during summer')",
        #height=100,
    #)

    #recommend_button = st.button("🌱 Get Recommendations")

    #if recommend_button and query.strip():
        #with st.spinner("Fetching recommendations from knowledge base..."):
            #from rag.query_kb import query_knowledge_base

            #kb_results = query_knowledge_base(query)
            #if kb_results:
                #top = kb_results[0]
                #st.success("✅ Recommendations Ready!")
                #st.markdown(f"**🌾 Crop:** {top['crop']}")
                #st.markdown(f"**🦠 Disease:** {top['disease']}")
                #st.markdown(f"**🔍 Symptoms:** {top['symptoms']}")
                #st.markdown(f"**💡 Treatment:** {top['treatment']}")
                #st.markdown(f"**🌱 Recommended Crops:** {top['recommended']}")
            #else:
                #st.warning("No suitable recommendations found in the knowledge base.")


# ------------------------------------------------
# Footer
# ------------------------------------------------
#st.markdown("---")
#st.markdown(
#    "<p style='text-align:center;'>🌿 SmartFarm AI © 2025 | Powered by CNN + Grok + Gemini + ChromaDB</p>",
#    unsafe_allow_html=True,
#)