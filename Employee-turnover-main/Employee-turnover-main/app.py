import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Turnover Predictor", layout="centered")
st.title("🔍 Employee Turnover Prediction")

# ───────────────────── Load Model ─────────────────────
model = None
try:
    with open("employee_turnover_optimized.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ───────────────────── Input Form ─────────────────────
if model:
    st.header("Employee Details")

    age = st.number_input("Age", 18, 70, 30)
    stag = st.number_input("Tenure (months)", 0, 600, 24)
    coach = st.selectbox("Received Coaching?", ["yes", "no"])
    extraversion = st.slider("Extraversion", 0.0, 10.0, 5.0)
    independ = st.slider("Independence", 0.0, 10.0, 5.0)
    selfcontrol = st.slider("Self-control", 0.0, 10.0, 5.0)
    anxiety = st.slider("Anxiety", 0.0, 10.0, 5.0)
    novator = st.slider("Novator", 0.0, 10.0, 5.0)
    profession = st.selectbox("Profession", ["HR", "Commercial", "Sales", "Manager"])  # adjust to actual values
    gender = st.selectbox("Gender", ["m", "f"])
    industry = st.selectbox("Industry", ["Banks", "Retail", "PowerGeneration"])
    traffic = st.selectbox("Traffic Level", ["rabrecNErab", "empjs", "youjs"])  # actual values
    way = st.selectbox("Commute", ["bus", "car", "bike", "walk"])
    head_gender = st.selectbox("Head Gender", ["m", "f"])

    # ───────────────────── Build Input ─────────────────────
    input_df = pd.DataFrame([{
        "stag": stag,
        "gender": 0 if gender == "m" else 1,
        "age": age,
        "industry": 0,  # adjust mapping as per your dataset
        "profession": 0,
        "traffic": 0,
        "coach": 1 if coach == "yes" else 0,
        "head_gender": 0 if head_gender == "m" else 1,
        "way": 0,
        "extraversion": extraversion,
        "independ": independ,
        "selfcontrol": selfcontrol,
        "anxiety": anxiety,
        "novator": novator,
        "tenure_years": stag / 12,
        "tenure_age_ratio": stag / (age + 1),
        "age_group": pd.cut([age], bins=[18,25,35,45,60,100], labels=[0,1,2,3,4])[0]
    }])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: Employee is likely to *{'leave' if prediction == 1 else 'stay'}*.")