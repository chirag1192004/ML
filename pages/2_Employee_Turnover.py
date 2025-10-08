import runpy
import streamlit as st

st.set_page_config(layout="wide")
st.title("Employee Turnover Predictor 🧑‍💼")
st.caption("This page dynamically loads the Streamlit application from your Employee-turnover-main/Employee-turnover-main/app.py file.")
st.markdown("---")

# Path is relative to the directory where streamlit run Home.py is executed (ML/)
try:
    runpy.run_path("../Employee-turnover-main/Employee-turnover-main/app.py", run_name="__main__")
except Exception as e:
    st.error(f"Error loading the application. Please check the file path and ensure necessary models (like employee_turnover_optimized.pkl) are present.")
    st.exception(e)
