import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="AI Sentiment & Emotion Analyzer", layout="centered")

st.title("🧠 AI Sentiment & Emotion Analyzer (NLP)")
st.write("Detects Sentiment + Emotions like Happy, Sad, Anger, Fear, Pain, Thinking")

text = st.text_area("✍️ Enter your text", height=150)

analyzer = SentimentIntensityAnalyzer()

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("⚠️ Please enter text")
    else:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        scores = analyzer.polarity_scores(text)

        # ---------- SENTIMENT ----------
        if polarity > 0.3:
            sentiment = "😊 Positive"
        elif polarity < -0.3:
            sentiment = "😞 Negative"
        else:
            sentiment = "😐 Neutral"

        # ---------- EMOTION DETECTION ----------
        emotion = "🤔 Thinking"

        if scores["pos"] > 0.6:
            emotion = "😊 Happy"
        elif scores["neg"] > 0.6 and scores["compound"] < -0.6:
            emotion = "😢 Pain / Sadness"
        elif scores["neg"] > 0.5:
            emotion = "😠 Anger"
        elif scores["compound"] < -0.4:
            emotion = "😨 Fear"
        elif scores["neu"] > 0.6:
            emotion = "😐 Neutral"
        else:
            emotion = "🤔 Thinking"

        # ---------- OUTPUT ----------
        st.success("Analysis Completed")

        st.write("### 🔍 Results")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Emotion:** {emotion}")

        st.write("### 📊 Score Details")
        st.json(scores)
