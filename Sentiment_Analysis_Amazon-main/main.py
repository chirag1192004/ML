import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer

# Load pipeline model
with open("./Models/pipeline_model.pkl", "rb") as f:
    pipeline_model = pickle.load(f)

# Preprocessing: still needed if your pipeline doesn't include text cleaning
stemmer = PorterStemmer()
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower().split()
    return " ".join([stemmer.stem(word) for word in text])

def predict_sentiment(texts):
    processed = [preprocess_text(t) for t in texts]
    preds = pipeline_model.predict(processed)
    return ["Positive" if p == 1 else "Negative" for p in preds]

# Pie chart of prediction distribution
def plot_sentiment_distribution(predictions):
    df = pd.Series(predictions).value_counts()
    fig, ax = plt.subplots()
    df.plot(kind="pie", autopct="%1.1f%%", colors=["green", "red"], startangle=90, shadow=True, ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("Sentiment Analysis App (Pipeline Model)")
st.markdown("This app predicts customer sentiment using a Pipeline-based model (TfidfVectorizer  + LogisticRegression).")

# Single prediction
st.subheader("🔍 Single Review Prediction")
text_input = st.text_area("Enter a review:")
if st.button("Predict Sentiment"):
    if text_input.strip():
        result = predict_sentiment([text_input])[0]
        st.success(f"Predicted Sentiment: **{result}**")
    else:
        st.warning("Please enter a valid review.")

# Bulk prediction
st.subheader("📁 Bulk Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with a 'Sentence' column", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "Sentence" in df.columns:
        predictions = predict_sentiment(df["Sentence"].tolist())
        df["Predicted Sentiment"] = predictions
        st.write("### Predictions:", df)

        # Pie chart
        plot_sentiment_distribution(predictions)

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
    else:
        st.error("The uploaded CSV must contain a 'Sentence' column.")
