import streamlit as st 
import pandas as pd 
from transformers import pipeline
import matplotlib.pyplot as plt  

# Load Hugging Face sentiment analysis model
def load_model(): 
    return pipeline("sentiment-analysis") 

sentiment_pipeline = load_model() 

# Analyze a single text input 
def analyze_sentiment(text): 
    result = sentiment_pipeline(text)[0] 
    sentiment = result['label'] 
    score = result['score'] 
    return sentiment, score 

# Visualize sentiment distribution 
def visualize_sentiment(sentiments): 
    sentiment_counts = sentiments.value_counts() 
    colors = {
        "POSITIVE": "green",
        "NEGATIVE": "red",
        "NEUTRAL": "gray"
    }
    fig, ax = plt.subplots() 
    sentiment_counts.plot(
        kind='bar',
        color=[colors.get(label, "blue") for label in sentiment_counts.index],
        ax=ax
    ) 
    ax.set_title("Sentiment Distribution") 
    ax.set_ylabel("Number of Reviews") 
    st.pyplot(fig) 

# Streamlit app layout 
st.title("Product Review Sentiment Analyzer (Hugging Face)") 

# Single review analysis 
st.subheader("Analyze a Single Review") 
single_review = st.text_area("Enter a product review:") 
if st.button("Analyze"): 
    if single_review.strip(): 
       sentiment, score = analyze_sentiment(single_review) 
       st.success(f"**Sentiment:** {sentiment} | **Confidence:** {score:.2f}") 
    else: 
       st.warning("Please enter a review before clicking Analyze.") 

# Multiple review analysis via CSV upload 
st.subheader("Analyze Reviews from a CSV File") 
uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=["csv"]) 

if uploaded_file: 
    try: 
        df = pd.read_csv(uploaded_file) 

        if 'review' not in df.columns: 
            st.error("The CSV must contain a column named 'review'") 
        else: 
            # Run sentiment analysis 
            df['Sentiment'], df['Confidence'] = zip(*df['review'].astype(str).apply(analyze_sentiment)) 
            st.dataframe(df) 

            # Visualize sentiment distribution 
            st.subheader("Sentiment Distribution") 
            visualize_sentiment(df["Sentiment"]) 

            # Download results 
            csv = df.to_csv(index=False) 
            st.download_button( 
                label="Download Results as CSV", 
                data=csv, 
                file_name="sentiment_results.csv", 
                mime="text/csv", 
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
