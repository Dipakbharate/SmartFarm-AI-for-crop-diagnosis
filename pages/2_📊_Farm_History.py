
import streamlit as st
import json
from pathlib import Path

# Page logic for Farm History


st.title("📊 Farm Diagnosis History")
st.markdown("Review past diagnoses saved in the system's long-term memory.")

# ------------------------------------------------
# Load History
# ------------------------------------------------
MEMORY_FILE = Path("data/memory/embeddings.json")

if not MEMORY_FILE.exists():
    st.info("📭 No diagnosis history found yet. Run some analyses in the Diagnosis Tool first!")
else:
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        if not history:
            st.info("📭 History is currently empty.")
        else:
            # Sort by timestamp (newest first)
            sorted_history = sorted(
                history.items(), 
                key=lambda x: x[1].get("timestamp", ""), 
                reverse=True
            )

            for entry_id, data in sorted_history:
                with st.expander(f"📅 {data.get('timestamp', 'Unknown Date')} - {data.get('diagnosis', 'Unknown')}"):
                    st.markdown(data.get("diagnosis_text", "No details available."))
                    
                    st.caption(f"Entry ID: {entry_id}")

    except Exception as e:
        st.error(f"Error loading history: {e}")

st.sidebar.title("🌿 SmartFarm AI")
st.sidebar.markdown("Advanced Crop Health Diagnosis")
st.sidebar.divider()
#st.sidebar.info("This history is powered by ChromaDB & Local Vector Memory.")
