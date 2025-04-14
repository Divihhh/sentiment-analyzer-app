import streamlit as st

import pandas as pd

from transformers import pipeline

import plotly.express as px



# Title and description

st.set_page_config(page_title="Review Sentiment & Emotion Analysis", layout="centered")

st.title("Product Review Sentiment & Emotion Analysis")

st.markdown("""

Analyze **sentiment** (positive/negative) and **emotions** (joy, anger, fear, etc.) in product reviews.

You can either input a single review or upload a CSV file with multiple reviews (with a column named `review`).

""")



# Load Hugging Face models

@st.cache_resource

def load_models():

    sentiment_model = pipeline("sentiment-analysis")

    emotion_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    return sentiment_model, emotion_model



sentiment_model, emotion_model = load_models()

emotion_labels = ["joy", "anger", "fear", "sadness", "surprise", "disgust"]



# --- SINGLE REVIEW ANALYSIS ---

st.header("1. Analyze a Single Review")

single_review = st.text_area("Enter your product review here:")



if single_review.strip():

    with st.spinner("Analyzing..."):

        sentiment = sentiment_model(single_review)[0]['label']

        emotion = emotion_model(single_review, candidate_labels=emotion_labels)['labels'][0]

    

    st.success("Analysis Complete!")

    st.write(f"**Sentiment:** {sentiment}")

    st.write(f"**Dominant Emotion:** {emotion}")



# --- MULTIPLE REVIEWS VIA CSV ---

st.header("2. Analyze Multiple Reviews from CSV")

uploaded_file = st.file_uploader("Upload a CSV file with a column named 'review'", type=['csv'])



if uploaded_file:

    try:

        df = pd.read_csv(uploaded_file)



        if 'review' not in df.columns:

            st.error("The CSV file must contain a column named 'review'.")

        else:

            df = df[df['review'].notna()].reset_index(drop=True)

            st.write("Preview of uploaded reviews:")

            st.write(df.head())



            st.info("Performing sentiment and emotion analysis...")

            progress = st.progress(0)

            sentiment_list, emotion_list = [], []



            for i, review in enumerate(df['review']):

                sentiment = sentiment_model(review)[0]['label']

                emotion = emotion_model(review, candidate_labels=emotion_labels)['labels'][0]

                sentiment_list.append(sentiment)

                emotion_list.append(emotion)

                progress.progress((i + 1) / len(df))



            df['Sentiment'] = sentiment_list

            df['Emotion'] = emotion_list

            st.success("Analysis complete!")



            st.write("Preview of results:")

            st.write(df.head())



            # Download button

            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button("Download Results as CSV", csv, "sentiment_emotion_results.csv", "text/csv")



            # Visualization

            st.subheader("Sentiment & Emotion Distribution")

            sentiment_fig = px.pie(df, names='Sentiment', title="Sentiment Distribution")

            emotion_fig = px.pie(df, names='Emotion', title="Emotion Distribution")

            st.plotly_chart(sentiment_fig)

            st.plotly_chart(emotion_fig)



    except Exception as e:

        st.error(f"Error reading the file: {e}")
