import runpy
import streamlit as st

st.set_page_config(layout="wide")
st.title("Global Earthquake Visualization 🌍")
st.caption("This page dynamically loads the Streamlit application from your earthquake/app.py file.")
st.markdown("---")

# Path is relative to the directory where streamlit run Home.py is executed (ML/)
try:
    runpy.run_path("../earthquake/app.py", run_name="__main__")
except Exception as e:
    st.error(f"Error loading the application. Check the file path and confirm that the earthquake_1995-2023.csv file is accessible for data loading.")
    st.exception(e)
