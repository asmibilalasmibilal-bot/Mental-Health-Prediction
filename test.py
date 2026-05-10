import streamlit as st

# Page config
st.set_page_config(page_title="Mental Health Prediction")

# MAIN TITLE
st.title("🧠 Mental Health Prediction Using ML")

# SUB TITLE
st.subheader("Detect Stress / Depression / Normal using Machine Learning")

# Simple test input
user_input = st.text_input("Enter text")

if user_input:
    st.success(f"You entered: {user_input}")