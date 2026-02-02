import streamlit as st
from transformers import pipeline
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

# Load models
hf_model = pipeline("sentiment-analysis")
vader = SentimentIntensityAnalyzer()

# Streamlit UI
st.title("🧠 Multi-Sentiment Analyzer")
st.write("Compare 3 models with **Positive / Neutral / Negative** labels:")

user_input = st.text_area("Enter your text:", "AI is helpful, but sometimes frustrating!")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):

            # 🤗 HuggingFace
            hf_result = hf_model(user_input)[0]
            hf_score = hf_result['score']
            hf_label = hf_result['label']

            # Adjust label for 3-category view
            if hf_score < 0.6:
                hf_sentiment = "Neutral"
            else:
                hf_sentiment = "Positive" if hf_label == "POSITIVE" else "Negative"

            # 📘 TextBlob
            blob = TextBlob(user_input)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            if polarity > 0.1:
                tb_sentiment = "Positive"
            elif polarity < -0.1:
                tb_sentiment = "Negative"
            else:
                tb_sentiment = "Neutral"

            # 🔍 VADER
            vader_scores = vader.polarity_scores(user_input)
            compound = vader_scores["compound"]
            if compound >= 0.05:
                vader_sentiment = "Positive"
            elif compound <= -0.05:
                vader_sentiment = "Negative"
            else:
                vader_sentiment = "Neutral"

               # Display results
        st.subheader("📊 Sentiment Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🤗 HuggingFace**")
            st.write(f"Sentiment: `{hf_sentiment}`")

        with col2:
            st.markdown("**📘 TextBlob**")
            st.write(f"Sentiment: `{tb_sentiment}`")

        with col3:
            st.markdown("**🔍 VADER**")
            st.write(f"Sentiment: `{vader_sentiment}`")
#NOTE:FOR EXECUTION,FIRST"pip install streamlit transformers textblob nltk",then execute in terminal


