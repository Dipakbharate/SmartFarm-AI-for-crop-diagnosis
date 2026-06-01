
import streamlit as st

# Page logic for Home


# ------------------------------------------------
# Custom CSS for Premium Look
# ------------------------------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f1f8e9;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #2E7D32;
        height: 100%;
        color: #1B5E20; /* Dark Green text for contrast */
    }
    .feature-card h3 {
        color: #2E7D32;
        margin-top: 0;
    }
    .feature-card p {
        color: #33691E;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Hero Section
# ------------------------------------------------
st.markdown('<div class="main-header">🌾 SmartFarm AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Pathological Intelligence for Sustainable Farming</div>', unsafe_allow_html=True)

col_hero1, col_hero2 = st.columns([2, 1])

with col_hero1:
    st.image("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&w=800&q=80", use_container_width=True)
    
with col_hero2:
    st.markdown("### 🚀 Mission")
    st.write("""
    SmartFarm AI leverages state-of-the-art Deep Learning and Advanced AI Vision to provide farmers with 
    instant, accurate, and actionable crop health diagnostics. 
    
    By identifying diseases early, we help reduce crop loss and minimize the use of harmful chemicals.
    """)
    if st.button("🌱 Get Started Now", type="primary", use_container_width=True):
        st.switch_page("pages/1_🩺_Diagnosis_Tool.py")

st.divider()

# ------------------------------------------------
# Technology Stack
# ------------------------------------------------
st.header("🔬 Our Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🧠 Deep Learning (CNN)</h3>
        <p>A custom-trained Convolutional Neural Network analyzes leaf patterns to identify common diseases with high precision.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>👁️ Advanced AI Vision</h3>
        <p>For complex cases, our system uses advanced pathological vision analysis to describe symptoms and estimate severity.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>💾 Long-Term Memory</h3>
        <p>Integrated Vector Database allows the system to recall past diagnoses and improve accuracy over time.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ------------------------------------------------
# How it Works
# ------------------------------------------------
st.header("🛠️ How It Works")
st.steps = [
    "**Upload**: Take a photo of the affected plant leaf.",
    "**Analyze**: Our AI pipeline processes the image through multiple neural layers.",
    "**Report**: Receive a detailed treatment plan and prevention tips instantly."
]

for i, step in enumerate(st.steps):
    st.markdown(f"{i+1}. {step}")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.sidebar.title("🌿 SmartFarm AI")
st.sidebar.markdown("Advanced Crop Health Diagnosis")
st.sidebar.divider()
st.sidebar.markdown("""
    ---
    > *"Precision in Diagnosis, Excellence in Harvest. Harnessing AI for a sustainable future."*
""")


st.markdown(
    "<br><hr><p style='text-align:center; color: gray;'>🌿 SmartFarm AI  2025-26  | Developed by Team SmartFarm</p>",
    unsafe_allow_html=True,
)