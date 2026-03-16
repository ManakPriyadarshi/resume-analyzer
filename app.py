import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

st.title("AI Resume Analyzer")

# Load dataset
data = pd.read_csv("ml_resume_dataset_4500.csv")

# Detect columns automatically
resume_col = None
category_col = None

for col in data.columns:
    if "resume" in col.lower() or "text" in col.lower():
        resume_col = col
    if "category" in col.lower() or "label" in col.lower():
        category_col = col

if resume_col is None or category_col is None:
    st.error("Dataset columns not detected correctly")
else:

    # Text cleaning function
    def clean_text(text):
        text = re.sub(r"http\S+", "", str(text))
        text = re.sub(r"[^a-zA-Z ]", "", text)
        text = text.lower()
        return text

    data["clean_resume"] = data[resume_col].apply(clean_text)

    # Convert text to numbers
    tfidf = TfidfVectorizer(stop_words="english")

    X = tfidf.fit_transform(data["clean_resume"])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(data[category_col])

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # User input
    resume_input = st.text_area("Paste your resume text")

    if st.button("Analyze Resume"):

        resume_clean = clean_text(resume_input)

        vector = tfidf.transform([resume_clean])

        prediction = model.predict(vector)

        job_role = le.inverse_transform(prediction)

        st.success("Predicted Job Role:")
        st.write(job_role[0])
