import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px

# Load models

@st.cache_resource

def load_models():

    sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    return sentiment_pipeline, emotion_pipeline



sentiment_model, emotion_model = load_models()

# Map sentiment labels

label_map = {

    "LABEL_0": "Negative",

    "LABEL_1": "Neutral",

    "LABEL_2": "Positive"

}



def analyze_text(text):

    sentiment_result = sentiment_model(text)[0]

    emotion_result = emotion_model(text)[0]

    sentiment = label_map.get(sentiment_result['label'], sentiment_result['label'])

    emotion = emotion_result['label']

    return sentiment, emotion



# App layout

st.title("Product Review Sentiment & Emotion Analyzer")
st.markdown("Analyze product reviews by detecting both **sentiment** and **emotion** using Hugging Face transformers.")

# Option 1: Manual input

st.header("1. Analyze a Single Review")

review_input = st.text_area("Enter a product review")



if st.button("Analyze Review") and review_input.strip():

    sentiment, emotion = analyze_text(review_input)

    st.success(f"**Sentiment**: {sentiment} | **Emotion**: {emotion}")

# Option 2: CSV Upload

st.header("2. Analyze Multiple Reviews from CSV")

uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type="csv")

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)



        if 'review' not in df.columns:

            st.error("The uploaded CSV must contain a column named 'review'.")

        else:

            sentiments = []

            emotions = []



            with st.spinner("Analyzing reviews..."):

                for review in df['review']:

                    if isinstance(review, str) and review.strip():

                        sentiment, emotion = analyze_text(review)

                    else:

                        sentiment, emotion = "N/A", "N/A"

                    sentiments.append(sentiment)

                    emotions.append(emotion)



            df['Sentiment'] = sentiments

            df['Emotion'] = emotions



            st.success("Analysis completed!")

            st.dataframe(df)

            # Pie Charts

            st.subheader("Sentiment Distribution")

            sentiment_counts = df['Sentiment'].value_counts().reset_index()

            sentiment_chart = px.pie(sentiment_counts, names='index', values='Sentiment',

                                     color_discrete_sequence=px.colors.sequential.RdBu,

                                     title="Sentiment Pie Chart")

            st.plotly_chart(sentiment_chart)



            st.subheader("Emotion Distribution")

            emotion_counts = df['Emotion'].value_counts().reset_index()

            emotion_chart = px.pie(emotion_counts, names='index', values='Emotion',

                                   color_discrete_sequence=px.colors.sequential.Viridis,

                                   title="Emotion Pie Chart")

            st.plotly_chart(emotion_chart)

            # Download button

            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button("Download Results as CSV", data=csv, file_name="analyzed_reviews.csv", mime='text/csv')



    except Exception as e:

        st.error(f"Error reading file: {e}")
