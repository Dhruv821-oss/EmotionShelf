import streamlit as st
from transformers import pipeline
import requests
import random

st.set_page_config(page_title="AI Mood-Based Book Finder", layout="centered")

# Emotion to literary topic mapping
emotion_to_topic = {
    "joy": "happiness",
    "sadness": "grief",
    "anger": "revenge",
    "fear": "horror",
    "love": "romance",
    "surprise": "mystery",
    "disgust": "tragedy",
    "neutral": "philosophy"
}

@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

emotion_model = load_emotion_model()

def detect_emotion(text):
    result = emotion_model(text)
    emotion = result[0]['label'].lower()
    return emotion

def fetch_books_by_topic(topic):
    url = f"https://gutendex.com/books/?topic={topic}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json()
    return data.get("results", [])

def extract_text_url(book):
    formats = book.get("formats", {})
    for key, val in formats.items():
        if "text/plain" in key and val.endswith(".txt"):
            return val
        elif "text/html" in key and val.endswith(".htm"):
            return val
    return None

# UI
st.title("📚 EmotionShelf")
st.write("Enter a sentence, and we'll detect your emotion and find a matching book!")

user_input = st.text_input("Describe how you're feeling or write anything:")
s="For YOU"
if user_input:
    try:
        emotion = detect_emotion(user_input)
        st.markdown(f"**Detected Emotion:** `{emotion}`")

        topic = emotion_to_topic.get(emotion, "philosophy")
        st.markdown(f"**Searching books :** `{s}`")

        books = fetch_books_by_topic(topic)

        if books:
            book = random.choice(books)
            title = book.get("title", "Unknown Title")
            authors = ", ".join([a.get("name", "Unknown") for a in book.get("authors", [])])
            text_url = extract_text_url(book)

            st.success(f"**{title}** by {authors}")
            if text_url:
                st.markdown(f"[📖 Read Book]({text_url})")
            else:
                st.warning("No readable text link found.")
        else:
            st.error("😔 No books found for this mood. Try a different feeling or phrasing.")

    except Exception as e:
        st.error(f"Emotion detection or book retrieval failed: {e}")
