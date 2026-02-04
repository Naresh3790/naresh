import streamlit as st
from transformers import pipeline
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from collections import Counter

# ------------------ Setup ------------------
try:
    nltk.data.find("sentiment/vader_lexicon")
except:
    nltk.download("vader_lexicon")

hf_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

vader = SentimentIntensityAnalyzer()

# ------------------ Helper functions ------------------
def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def read_docx(file):
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)

# HuggingFace long text safe function
def hf_analyze(text):
    max_len = 400
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    results = []
    for chunk in chunks:
        res = hf_model(chunk, truncation=True)[0]
        results.append(res["label"])
    return Counter(results).most_common(1)[0][0]

# ------------------ UI ------------------
st.title("🧠 Multi-Sentiment Analyzer")
st.write("Consolidated Sentiment Analysis with Additional Traits")

uploaded_file = st.file_uploader(
    "📄 Upload a document (TXT, PDF, DOCX)",
    type=["txt", "pdf", "docx"]
)

user_input = st.text_area(
    "✍️ Or enter text manually:",
    "i am very happy today"
)

# ------------------ Text selection ------------------
text_data = ""

if uploaded_file:
    file_type = uploaded_file.type

    if file_type == "text/plain":
        text_data = read_pdf( uploaded_file)

    elif file_type in ["application/pdf", "application/octet-stream"]:
        text_data = read_txt(uploaded_file)

    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text_data = read_docx(uploaded_file)

    st.text_area("📃 Extracted Text", text_data, height=200)
else:
    text_data = user_input

# ------------------ Analysis ------------------
if st.button("🔍 Analyze Sentiment"):

    if not text_data.strip():
        st.warning("Please provide some text.")
    else:
        with st.spinner("Analyzing..."):

            # 🤗 HuggingFace
            hf_label = hf_analyze(text_data)
            hf_sentiment = "Positive" if hf_label == "POSITIVE" else "Negative"

            # 📘 TextBlob
            blob = TextBlob(text_data)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            tb_sentiment = (
                "Positive" if polarity > 0.1
                else "Negative" if polarity < -0.1
                else "Neutral"
            )

            # 🔍 VADER
            vader_scores = vader.polarity_scores(text_data)
            compound = vader_scores["compound"]

            vader_sentiment = (
                "Positive" if compound >= 0.05
                else "Negative" if compound <= -0.05
                else "Neutral"
            )

        # ------------------ Consolidation ------------------
        sentiments = [hf_sentiment, tb_sentiment, vader_sentiment]
        counts = Counter(sentiments)
        final_sentiment, votes = counts.most_common(1)[0]

        if votes == 1:
            final_sentiment = "Neutral"

        # ------------------ Extra Traits ------------------
        if compound >= 0.6:
            emotion = "Joy"
        elif compound <= -0.6:
            emotion = "Sadness"
        elif compound < -0.2:
            emotion = "Anger"
        else:
            emotion = "Neutral"

        subjectivity_label = "Subjective" if subjectivity > 0.5 else "Objective"

        abs_score = abs(compound)

        intensity = "Low" if abs_score < 0.3 else "Medium" if abs_score < 0.6 else "High"
        strength = "Weak" if abs_score < 0.3 else "Moderate" if abs_score < 0.6 else "Strong"

        confidence = int((votes / 3) * 100)

        # ------------------ Display ------------------
        st.subheader("🤖 Final Consolidated Sentiment")

        if final_sentiment == "Positive":
            st.success("POSITIVE 😊")
        elif final_sentiment == "Negative":
            st.error("NEGATIVE 😞")
        else:
            st.info("NEUTRAL 😐")

        st.subheader("📌 Additional Sentiment Traits")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Emotion** : {emotion}")
            st.write(f"**Subjectivity** : {subjectivity_label}")
            st.write(f"**Intensity** : {intensity}")

        with col2:
            st.write(f"**Confidence** : {confidence}%")
            st.write(f"**Strength** : {strength}")

        st.caption(f"Agreement: {votes}/3 models")
