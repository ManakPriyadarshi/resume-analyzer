import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("ml_resume_dataset_4500.csv")

# Text cleaning
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.lower()
    return text

data["clean_resume"] = data["Resume"].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(data["clean_resume"])

y = data["Category"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

# Streamlit UI
st.title("AI Resume Analyzer")

resume = st.text_area("Paste your resume text")

if st.button("Analyze Resume"):

    resume_clean = clean_text(resume)

    vector = tfidf.transform([resume_clean])

    prediction = model.predict(vector)

    st.success("Predicted Job Role:")

    st.write(prediction[0])
