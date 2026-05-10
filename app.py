import streamlit as st
import joblib

# page setup
st.set_page_config(page_title="Mental Health Prediction")

st.title("🧠 Mental Health Prediction Using ML")
st.subheader("Detect Stress / Depression / Normal")

# load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# input
user_input = st.text_area("Enter your text")

# predict button
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)

        # 🔥 FINAL OUTPUT (IMPORTANT)
        st.success(f"Prediction: {prediction[0]}")