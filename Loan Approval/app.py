import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Set Matplotlib/Seaborn style for consistent plotting
plt.rcParams['font.family'] = 'Inter'
sns.set_style("whitegrid")

# Define the file path (assumes train_1.csv is accessible in the same environment)
DATA_FILE = "train_1.csv"
TARGET_COL = 'label'

@st.cache_data
def load_data():
    """Loads the dataset and performs initial cleanup."""
    try:
        df = pd.read_csv(DATA_FILE)
        # Drop identifier columns
        df = df.drop(columns=['loan_id', 'id'], errors='ignore')
        return df
    except FileNotFoundError:
        st.error(f"Error: {DATA_FILE} not found. Please ensure the file is in the correct path.")
        return None

@st.cache_resource
def train_model(df):
    """
    Sets up the preprocessing pipeline and trains the Logistic Regression model.
    This logic mirrors the steps in your 'loan_approval_model.py' Canvas.
    """
    if df is None:
        return None, None, None, None, None, None, None # Added one extra None for X_train

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # Identify numerical and categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    # FIX: Ensure all categorical columns are uniformly strings for OneHotEncoder (as implemented in Canvas)
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    # Split data (using the same random state and test size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Preprocessing Pipelines ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # --- Full Modeling Pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # Use the same Logistic Regression parameters
        ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Generate predictions for evaluation
    y_pred = model_pipeline.predict(X_test)

    # Return necessary components (X_train is now included here)
    return model_pipeline, X_train, X_test, y_test, y_pred, numerical_cols, categorical_cols

def plot_confusion_matrix(conf_matrix):
    """Generates and displays the confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected (0)', 'Approved (1)'],
                yticklabels=['Rejected (0)', 'Approved (1)'],
                ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    st.pyplot(fig)

def plot_roc_curve(model, X_test, y_test):
    """Generates and displays the ROC curve."""
    # Predict probabilities for the test set using X_test directly
    y_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='orange', label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax.legend(loc="lower right")
    st.pyplot(fig)

# --- Streamlit Main Application ---

def main():
    st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
    st.title("💰 Loan Approval Prediction Model")
    st.subheader("Interactive Logistic Regression Dashboard")

    # Load data and train model
    data = load_data()
    # Updated main to receive X_train
    model_pipeline, X_train, X_test, y_test, y_pred, numerical_cols, categorical_cols = train_model(data)

    if model_pipeline is None:
        return

    # --- 1. Model Performance Section ---
    st.header("Model Performance Metrics")
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Model Accuracy on Test Set", value=f"{accuracy:.4f}")
        st.text("Classification Report (Test Set):")
        # Display the classification report as a nicely formatted markdown table
        report_df = pd.DataFrame(class_report).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

    with col2:
        st.subheader("Model Visualizations")
        tab1, tab2 = st.tabs(["Confusion Matrix", "ROC Curve"])

        with tab1:
            plot_confusion_matrix(conf_matrix)
        with tab2:
            # Passed X_test to plot_roc_curve
            plot_roc_curve(model_pipeline, X_test, y_test)

    # --- 2. Loan Prediction Input Form ---
    st.markdown("---")
    st.header("Test a New Loan Application")
    st.markdown("Adjust the features below to see the model's prediction:")

    # Create input dictionary
    input_data = {}

    # Helper function to get the most common value for categorical defaults
    def get_mode(col):
        # Handle case where the column might have NaT or non-string types
        try:
            # Use the data directly for mode calculation
            return data[col].astype(str).mode()[0]
        except IndexError:
            # Catches if mode() returns an empty series (e.g., if all values are NaN)
            return "N/A"
        except Exception:
            # Catches other unexpected exceptions
            return "N/A"

    # For a clean UI, display only a few key features from each category
    NUM_FEATURES_DISPLAY = min(6, len(numerical_cols))
    CAT_FEATURES_DISPLAY = min(6, len(categorical_cols))

    form = st.form(key='prediction_form')
    
    # Numerical inputs
    num_cols_display = form.columns(3)
    form.subheader("Numerical Features")
    for i, col in enumerate(numerical_cols[:NUM_FEATURES_DISPLAY]):
        # Calculate mean for default value
        mean_val = data[col].mean() if data[col].dtype != object else 0.0
        
        input_data[col] = num_cols_display[i % 3].number_input(
            f"{col}",
            value=float(f"{mean_val:.2f}"),
            step=0.01,
            key=f'num_{col}'
        )

    # Categorical inputs
    cat_cols_display = form.columns(3)
    form.subheader("Categorical Features")
    for i, col in enumerate(categorical_cols[:CAT_FEATURES_DISPLAY]):
        # Get unique values for selection
        options = data[col].astype(str).unique().tolist()
        default_val = get_mode(col)
        
        try:
            default_index = options.index(default_val)
        except ValueError:
            default_index = 0 # Fallback if mode isn't a clean option

        input_data[col] = cat_cols_display[i % 3].selectbox(
            f"{col}",
            options=options,
            index=default_index,
            key=f'cat_{col}'
        )

    submit_button = form.form_submit_button(label='Predict Loan Approval', type="primary")

    if submit_button:
        # --- Prepare Single Row for Prediction ---
        
        # 1. Start with a copy of a single row from the training data (to maintain all column names)
        # X_train is now defined in main scope
        X_test_single = X_train.head(1).copy() 
        
        # 2. Update displayed columns with user input
        for key, value in input_data.items():
             X_test_single[key] = value

        # 3. Handle undisplayed columns (fill with mean/mode from training set)
        for col in X_train.columns:
            if col not in input_data:
                if col in numerical_cols:
                    # X_train is now defined in main scope
                    X_test_single[col] = X_train[col].mean()
                elif col in categorical_cols:
                    # X_train is now defined in main scope
                    X_test_single[col] = X_train[col].astype(str).mode()[0]
                    
        # 4. Final type check: ensure categorical columns are strings for the pipeline
        for col in categorical_cols:
             if col in X_test_single.columns:
                 X_test_single[col] = X_test_single[col].astype(str)

        # Make prediction
        prediction = model_pipeline.predict(X_test_single)
        prediction_proba = model_pipeline.predict_proba(X_test_single)[0]

        st.markdown("### Prediction Result")

        if prediction[0] == 1:
            st.success("**Approval Predicted!** 🎉")
            st.markdown(f"The model estimates a **{prediction_proba[1]*100:.2f}%** chance of approval.")
        else:
            st.error("**Rejection Predicted.** 🛑")
            st.markdown(f"The model estimates a **{prediction_proba[0]*100:.2f}%** chance of rejection.")

        st.info("Note: Only a subset of features was displayed for interaction. Missing features were automatically filled with their average/most frequent values from the training data to ensure the model pipeline runs correctly.")


if __name__ == "__main__":
    main()
