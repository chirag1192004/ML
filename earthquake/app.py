import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
@st.cache_resource
def load_and_train_model():
    """
    Loads the data, preprocesses it, and trains the Random Forest Regressor model.
    This function is cached by Streamlit to avoid retraining on every user interaction.
    """
    try:
        df = pd.read_csv('earthquake_1995-2023.csv')
    except FileNotFoundError:
        st.error("The file 'earthquake_1995-2023.csv' was not found. Please upload it to the same directory as this script.")
        st.stop()

    df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
    df.dropna(subset=['date_time'], inplace=True)

    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day

    features = ['latitude', 'longitude', 'depth', 'year', 'month', 'day']
    target = 'magnitude'

    df.dropna(subset=features + [target], inplace=True)

    X = df[features]
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    r2_score_train = r2_score(y_train, y_pred_train)
    st.write(f"Model Training R² Score: {r2_score_train:.4f}")
    
    return model

model = load_and_train_model()


st.title("Earthquake Magnitude Predictor")
st.write("Enter the parameters below to predict the earthquake's magnitude.")
st.write("---")

st.header("Input Parameters")
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=30.0)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-120.0)
depth = st.number_input("Depth (km)", min_value=0.0, value=10.0)


st.subheader("Date")
today = datetime.date.today()
d = st.date_input("Date of Event", value=today)
year = d.year
month = d.month
day = d.day


if st.button("Predict Magnitude"):

    input_data = pd.DataFrame([[latitude, longitude, depth, year, month, day]],
                              columns=['latitude', 'longitude', 'depth', 'year', 'month', 'day'])

    prediction = model.predict(input_data)

    st.header("Prediction Result")
    st.success(f"The predicted earthquake magnitude is: **{prediction[0]:.2f}**")