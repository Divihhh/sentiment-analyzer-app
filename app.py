import pandas as pd

import streamlit as st

from transformers import pipeline

import plotly.express as px



# Function to handle the file upload and CSV reading

def handle_csv_upload(uploaded_file):

    try:

        # Read the CSV file (skip index column if necessary)

        df = pd.read_csv(uploaded_file)



        # Check if 'review' column exists in the dataframe

        if 'review' not in df.columns:

            st.error("CSV must contain a 'review' column.")

            return None

        

        # Check for empty reviews in the 'review' column and remove those rows

        df = df[df['review'].notna()]  # Drop rows where 'review' is NaN



        # Display a success message and preview the first few rows of the file

        st.success("CSV uploaded and read successfully!")

        st.write(df.head())  # Show preview of the data for user to confirm

        

        return df

    

    except pd.errors.ParserError as e:

        st.error(f"Error reading the CSV file: {e}")

    except Exception as e:

        st.error(f"Unexpected error: {e}")



# Load Hugging Face models for sentiment and emotion analysis

@st.cache_resource

def load_models():

    sentiment_analyzer = pipeline("sentiment-analysis")

    emotion_analyzer = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    return sentiment_analyzer, emotion_analyzer



# Function to analyze sentiment and emotion of each review

def analyze_reviews(df, sentiment_analyzer, emotion_analyzer):

    # Initialize result lists

    sentiments = []

    emotions = []

    

    # Define possible emotions for classification

    possible_emotions = ["joy", "anger", "fear", "sadness", "surprise", "disgust"]



    # Process each review in the dataframe

    for review in df['review']:

        # Sentiment analysis

        sentiment_result = sentiment_analyzer(review)[0]

        sentiments.append(sentiment_result['label'])

        

        # Emotion analysis

        emotion_result = emotion_analyzer(review, candidate_labels=possible_emotions)

        emotions.append(emotion_result['labels'][0])

    

    # Add sentiment and emotion columns to the dataframe

    df['sentiment'] = sentiments

    df['emotion'] = emotions

    return df



# Plotting sentiment and emotion distributions using Plotly

def plot_results(df):

    # Check sentiment column content before plotting

    sentiment_counts = df['sentiment'].value_counts()

    if sentiment_counts.empty:

        st.warning("No sentiment data available for plotting.")

        return



    # Plot sentiment distribution

    sentiment_fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")



    # Plot emotion distribution

    emotion_counts = df['emotion'].value_counts()

    if emotion_counts.empty:

        st.warning("No emotion data available for plotting.")

        return

    emotion_fig = px.pie(values=emotion_counts.values, names=emotion_counts.index, title="Emotion Distribution")



    # Show the plots

    st.plotly_chart(sentiment_fig)

    st.plotly_chart(emotion_fig)



# Streamlit file uploader widget

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])



# Load models on app startup

sentiment_analyzer, emotion_analyzer = load_models()



if uploaded_file is not None:

    # Process the uploaded file

    df = handle_csv_upload(uploaded_file)



    if df is not None:

        # Perform sentiment and emotion analysis

        df = analyze_reviews(df, sentiment_analyzer, emotion_analyzer)



        # Show a preview of the dataframe with sentiment and emotion

        st.write("Preview of the data with sentiment and emotion analysis:")

        st.write(df.head())



        # Optionally, allow users to download the results as a new CSV file

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(

            label="Download Results as CSV",

            data=csv,

            file_name='product_reviews_with_analysis.csv',

            mime='text/csv'

        )



        # Display the pie charts for sentiment and emotion distribution

        plot_results(df)
