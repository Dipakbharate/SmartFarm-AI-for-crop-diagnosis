import streamlit as st

# ------------------------------------------------
# Navigation Configuration
# ------------------------------------------------
# This approach allows us to name the pages exactly how we want in the sidebar
# without being restricted by the underlying filenames.

pages = [
    st.Page("🏠_Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_🩺_Diagnosis_Tool.py", title="Diagnosis Tool", icon="🩺"),
    st.Page("pages/2_📊_Farm_History.py", title="Farm History", icon="📊"),
]

# Initialize Navigation
pg = st.navigation(pages)

# Global Page Configuration
st.set_page_config(
    page_title="SmartFarm AI",
    page_icon="🌾",
    layout="wide",
)

# Run the application
pg.run()
