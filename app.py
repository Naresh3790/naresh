import streamlit as st
from transformers import pipeline
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from pypdf import PdfReader
import docx
from collections import Counter

# Download VADER lexicon
nltk.download("vader_lexicon", quiet=True)

# Load models
hf_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
vader = SentimentIntensityAnalyzer()

# ---------- Helper functions ----------
def read_txt(file):
    return file.read().decode("utf-8")

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_docx(file):
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)

# ---------- Streamlit UI ----------
st.title("🧠 Multi-Sentiment Analyzer")
st.write("Analyze sentiment using HuggingFace, TextBlob, and VADER")

uploaded_file = st.file_uploader(
    "📄 Upload a document (TXT, PDF, DOCX)",
    type=["txt", "pdf", "docx"]
)

user_input = st.text_area(
    "✍️ Or enter text manually:",
    "i am good"
)

# ---------- Text Selection ----------
text_data = ""

if uploaded_file:
    if uploaded_file.type == "text/plain":
        text_data = read_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        text_data = read_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text_data = read_docx(uploaded_file)

    st.text_area("📃 Extracted Text", text_data, height=200)
else:
    text_data = user_input

# ---------- Analysis ----------
if st.button("🔍 Analyze Sentiment"):

    if not text_data.strip():
        st.warning("Please provide some text.")
    else:
        with st.spinner("Analyzing..."):

            # 🤗 HuggingFace
            hf = hf_model(text_data)[0]
            hf_sentiment = (
                "Neutral"
                if hf["score"] < 0.6
                else "Positive" if hf["label"] == "POSITIVE" else "Negative"
            )

            # 📘 TextBlob
            polarity = TextBlob(text_data).sentiment.polarity
            tb_sentiment = (
                "Positive" if polarity > 0.1
                else "Negative" if polarity < -0.1
                else "Neutral"
            )

            # 🔍 VADER
            compound = vader.polarity_scores(text_data)["compound"]
            vader_sentiment = (
                "Positive" if compound >= 0.05
                else "Negative" if compound <= -0.05
                else "Neutral"
            )

        # ---------- Consolidated Result ----------
        sentiments = [hf_sentiment, tb_sentiment, vader_sentiment]
        counts = Counter(sentiments)

        final_sentiment, votes = counts.most_common(1)[0]

        # If all models disagree
        if votes == 1:
            final_sentiment = "Neutral"

        # ---------- Display ----------
        st.subheader("🤖 Final Consolidated Sentiment")

        if final_sentiment == "Positive":
            st.success("POSITIVE 😊")
        elif final_sentiment == "Negative":
            st.error("NEGATIVE 😞")
        else:
            st.info("NEUTRAL 😐")

        st.caption(f"Agreement: {votes}/3 models")
 

