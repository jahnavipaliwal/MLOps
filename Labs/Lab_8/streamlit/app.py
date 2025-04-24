import streamlit as st
import joblib

st.title("Reddit Comment Classification")
st.markdown("### All you have to do to use this app is enter a comment and hit the Predict button.")

reddit_comment = [st.text_area("Input your comment here:")]

def load_artifacts():
    model_pipeline = joblib.load("reddit_model_pipeline.joblib")
    return model_pipeline

model_pipeline = load_artifacts()

def predict(reddit_comment):
    X = reddit_comment
    predictions = model_pipeline.predict_proba(X)
    return {'Predictions': predictions}

preds = predict(reddit_comment)
st.write(preds)