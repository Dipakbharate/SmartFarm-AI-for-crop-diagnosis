
import os
import streamlit as st
from PIL import Image
from services.hybrid_predictor import hybrid_predict

# Page logic for Diagnosis Tool


# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.title("🌿 SmartFarm AI")
st.sidebar.markdown("Advanced Crop Health Diagnosis")
st.sidebar.divider()

# Advanced Options
st.sidebar.subheader("⚙️ Settings")
force_ai = st.sidebar.checkbox("🔍 Force Advanced Analysis", help="Enable this to bypass the standard CNN and use Advanced AI Vision for deeper analysis.")

# ✅ Memory Reset Button
if st.sidebar.button("🧹 Reset System Cache"):
    from services.memory_service import MemoryService
    try:
        ms = MemoryService()
        ms.reset_memory()
        st.sidebar.success("✅ Cache cleared successfully!")
        st.toast("System reset. Next analysis will be fresh.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"⚠️ Reset failed: {e}")

st.sidebar.divider()
st.sidebar.markdown("👨‍🌾 **Project Status:** Stable Release")
st.sidebar.markdown("📚 Technologies: CNN + Advanced Vision + Expert AI")

# ------------------------------------------------
# Main UI
# ------------------------------------------------
st.title("🩺 Plant Health Diagnosis Assistant")
st.markdown(
    "Upload a clear photo of your crop’s leaf for **AI-powered disease detection and treatment advice.** "
)

# ------------------------------------------------
# Input Section
# ------------------------------------------------
col_in1, col_in2 = st.columns(2)

with col_in1:
    uploaded_image = st.file_uploader(
        "📤 Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_upload",
    )

with col_in2:
    crop_input = st.text_input(
        "🌱 Optional: Enter Crop Name",
        placeholder="e.g. Tomato, Potato (leave blank for auto-detect)",
    )

analyze_button = st.button("🔍 Run Diagnosis", type="primary", use_container_width=True)
# ------------------------------------------------
# Processing & Results
# ------------------------------------------------
if uploaded_image and analyze_button:
    with st.spinner("Analyzing plant health... please wait ⏳"):
        # Save uploaded image
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, uploaded_image.name)
        image = Image.open(uploaded_image).convert("RGB")
        image.save(image_path)
        
        # Run stable hybrid AI prediction
        result = hybrid_predict(
            image_path, 
            user_crop=crop_input.strip() if crop_input else None,
            force_gemini=force_ai
        )

        if result["stage"] == "error":
            st.error(result["message"])
        else:
            st.success("✅ Analysis Complete!")

            # Display results
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

            with res_col2:
                st.subheader("🌿 Diagnosis Result")
                
                # Show Stage Badge (Anonymized)
                if result["stage"] == "gemini_vision_grok":
                    st.info("🧬 **Mode:** Advanced AI Vision (Deep Analysis)")
                elif result["stage"] == "cnn_grok":
                    st.success("🧠 **Mode:** Standard CNN Diagnosis")
                
                # Visual highlight for cache
                if result["stage"] == "cached_result":
                    st.info("⚡ **Mode:** Result loaded from Cache")

                st.markdown("### 🧠 Analysis Summary")
                st.markdown(result["message"], unsafe_allow_html=True)

                # Metadata & Confidence
                meta = result.get("metadata", {})
                if "cnn_conf" in meta and meta["cnn_conf"] > 0:
                    st.markdown(f"**AI Confidence:** `{meta['cnn_conf']:.1%}`")
                    st.progress(meta["cnn_conf"])


# ------------------------------------------------
# Footer
# ------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; color: gray;'>🌿 SmartFarm AI 2025-26 | Powered by CNN + Advanced Pathological Intelligence</p>",
    unsafe_allow_html=True,
)
