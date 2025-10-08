import runpy
import streamlit as st

st.set_page_config(layout="wide")
st.title("Loan Approval Prediction System 🏦")
st.caption("This page dynamically loads the Streamlit application from your Loan Approval/app.py file.")
st.markdown("---")

# Path is relative to the directory where streamlit run Home.py is executed (ML/)
try:
    runpy.run_path("../Loan Approval/app.py", run_name="__main__")
except Exception as e:
    st.error(f"Error loading the application. Please check the file path and ensure all dependencies and data files (like train_1.csv) are accessible.")
    st.exception(e)
