"""
Streamlit Web Dashboard App.
Provides interactive frontend UI for user uploads, prediction displays, and XAI views.
Responsible Team Member: Member 5 (Frontend Interface & UI)

Usage:
    streamlit run ui/app.py
"""

import streamlit as st

def main():
    st.set_page_config(page_title="SignalScope - AI Image Detector", layout="wide")
    st.title("🔍 SignalScope")
    st.subheader("AI-Generated Image Detection & Explainability Tool")
    st.info("Upload an image to inspect forensic authenticity and visual explainability heatmaps.")

if __name__ == "__main__":
    main()
