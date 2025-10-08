import runpy
import streamlit as st

st.set_page_config(layout="wide")
st.title("Conversational AI Chatbot 🤖")
st.caption("This page dynamically loads the Streamlit application from your chatbot/chatbot_streamlit_app.py file.")
st.markdown("---")

# Path is relative to the directory where streamlit run Home.py is executed (ML/)
try:
    runpy.run_path("../chatbot/chatbot_streamlit_app.py", run_name="__main__")
except Exception as e:
    st.error(f"Error loading the application. Ensure PyTorch, your model files (e.g., chatbot_model_pytorch.pth), and intents.json are correctly configured.")
    st.exception(e)
