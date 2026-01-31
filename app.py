st.header("Text Analysis")
user_text = st.text_area("Enter text:", "I feel tired and lonely today.")
if st.button("Analyze Text"):
    if user_text.strip():
        # Basic Sentiment
        s = sentiment_model(user_text)[0]
        if s["score"] < 0.6:
            basic_sentiment = "Neutral"
        else:
            basic_sentiment = "Positive" if s["label"] == "POSITIVE" else "Negative"
        # Emotion detection
        emotions = emotion_model(user_text)[0]
        emotions = sorted(emotions, key=lambda x: x["score"], reverse=True)
        top_emotion = emotions[0]["label"]
        # Thinking guess (rule-based)
        blob = TextBlob(user_text)
        pol = blob.sentiment.polarity
        if pol < -0.4:
            thinking = "Suffering / Pain"
        elif pol < -0.1:
            thinking = "Confused / Overthinking"
        elif pol > 0.3:
            thinking = "Hopeful / Positive Thinking"
        else:
            thinking = "Neutral / Reflective"
        st.success("Text Analysis Result")
        st.write("**Basic Sentiment:**", basic_sentiment)
        st.write("**Emotion Detected:**", top_emotion)
        st.write("**Thinking Guess:**", thinking)
    else:
        st.warning("Please enter some text")
