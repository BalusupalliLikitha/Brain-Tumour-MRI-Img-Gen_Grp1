import streamlit as st
from app import show_detection
from dashboard import show_dashboard

# Set page config (Keep this at the top)
st.set_page_config(page_title="Brain Tumor AI", layout="wide")

# --- SIDEBAR ---
# Change "Navigation" to "Project Components"
st.sidebar.title("AI Brain Tumor Detection")

# Adding the small theory/summary you asked for
st.sidebar.info(
    """
    **About this Project:** This AI system uses GAN-augmented data and Grad-CAM 
    visualization to detect brain tumors in MRI scans with 
    high precision and explainability.
    """
)

st.sidebar.markdown("---") # Visual separator

# The radio button menu
choice = st.sidebar.radio(
    "Modules:",
    ["Tumor Detection Tool", "Performance Dashboard"]
)

# --- PAGE ROUTING ---
if choice == "Tumor Detection Tool":
    show_detection()
elif choice == "Performance Dashboard":
    show_dashboard()