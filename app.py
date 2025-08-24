# ============================
# app.py (Streamlit App)
# ============================
#to run this code: 
# python -m streamlit run app.py

import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🔐 AI-Powered Phishing URL Detector")

url = st.text_input("Enter a URL to check:")

if st.button("Analyze"):
    features = vectorizer.transform([url])
    prediction = int(model.predict(features)[0])  # ✅ defined here
    
    if prediction == 1:
        st.error("⚠️ Warning: This URL looks like a phishing attempt!")
    else:
        st.success("✅ This URL seems safe.")
