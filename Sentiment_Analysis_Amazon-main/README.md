# 🛍️ Amazon Review Sentiment Analyzer

A machine learning project that analyzes customer reviews from the Amazon Alexa dataset to predict sentiment (positive or negative). The project leverages a pipeline approach with **TF-IDF vectorization** and **Logistic Regression** for efficient preprocessing and classification.

---

## 📌 Features

* Preprocessing of textual review data.
* Sentiment classification using a scikit-learn pipeline.
* TF-IDF feature extraction.
* Logistic Regression model for sentiment prediction.
* Evaluation metrics: accuracy, confusion matrix, classification report.
* Jupyter Notebook for training and visualization.
* Optional Streamlit web app (if applicable in future).

---

## 🗂️ Dataset

The dataset used is the [Amazon Alexa Reviews dataset](https://github.com/itz-Pratham/Sentiment_Analysis_Amazon/blob/main/Data/amazon_alexa.tsv), which contains:

* `verified_reviews`: Customer review text
* `feedback`: Target variable (1 = positive, 0 = negative)

---

## 🧠 Tech Stack

* **Python**
* **scikit-learn**
* **pandas**, **numpy**
* **matplotlib**, **seaborn** (for visualization)
* **TF-IDF Vectorizer**
* **Logistic Regression**

---

## 🚀 How to Run

1. **Clone the Repository**

   ```bash
   git clone https://github.com/itz-Pratham/Sentiment_Analysis_Amazon.git
   cd Sentiment_Analysis_Amazon
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**

   ```bash
   jupyter notebook Sentiment_Analysis.ipynb
   ```

---

## 📊 Results

* Achieved high accuracy with Logistic Regression and TF-IDF features.
* Clear distinction between positive and negative reviews observed in classification results.

*(You can include a confusion matrix or accuracy figure here)*

---

## 📎 Future Improvements

* Integrate with a Streamlit web app for user interaction.
* Add support for more ML models (e.g., Naive Bayes, SVM).
* Implement hyperparameter tuning using GridSearchCV.

---

## 🙌 Acknowledgements

* Dataset from [Kaggle](https://github.com/itz-Pratham/Sentiment_Analysis_Amazon/blob/main/Data/amazon_alexa.tsv)
* Inspired by real-world sentiment analysis applications.

---

## 📬 Contact

Created with 💻 by **[Pratham](https://github.com/itz-Pratham)**
Feel free to connect or suggest improvements!

---

Would you like help generating a badge section or requirements.txt as well?

