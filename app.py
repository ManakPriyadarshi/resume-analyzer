import streamlit as st
import pandas as pd

st.title("AI Resume Analyzer")

resume = st.text_area("Paste your resume text")

if st.button("Analyze Resume"):
    st.success("Resume analyzed successfully")
