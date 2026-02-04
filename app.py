import st 
from transformers import pipeline
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from pypdf import PdfReader
import docx
from collections import Counter

# ------------------ Setup ------------------
nltk.download("vader_lexicon", quiet=True)

hf_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

vader = SentimentIntensityAnalyzer()

# ------------------ Helper functions ------------------
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
    if uploaded_file.type == "text/plain":
        text_data = read_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        text_data = read_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
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
            hf = hf_model(text_data)[0]
            hf_sentiment = (
                "Neutral"
                if hf["score"] < 0.6
                else "Positive" if hf["label"] == "POSITIVE" else "Negative"
            )

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

        # Emotion
        if compound >= 0.6:
            emotion = "Joy"
        elif compound <= -0.6:
            emotion = "Sadness"
        elif compound < -0.2:
            emotion = "Anger"
        else:
            emotion = "Neutral"

        # Subjectivity
        subjectivity_label = "Subjective" if subjectivity > 0.5 else "Objective"

        # Intensity
        abs_score = abs(compound)
        if abs_score < 0.3:
            intensity = "Low"
        elif abs_score < 0.6:
            intensity = "Medium"
        else:
            intensity = "High"

        # Confidence
        confidence = int((votes / 3) * 100)

        # Strength
        if abs_score < 0.3:
            strength = "Weak"
        elif abs_score < 0.6:
            strength = "Moderate"
        else:
            strength = "Strong"

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

