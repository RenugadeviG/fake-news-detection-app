import streamlit as st
import joblib
import requests
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the pre-trained Naive Bayes model and vectorizer
vectorizer = joblib.load("vectorizer_crossval.jb")
nb_model = joblib.load("NB_model_crossval.jb")

# Function to preprocess text
def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fetch strictly political news from top sources
def fetch_political_news():
    API_KEY = '67092e3cd29d480ba12a9501a2131dd0'  # Replace with your NewsAPI key
    sources = "bbc-news,cnn"
    keywords = "politics OR government OR election OR parliament OR legislation"
 
    URL = f"https://newsapi.org/v2/everything?q={keywords}&sources={sources}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok" or "articles" not in data:
            return []
        
        return [
            {"title": article.get("title", "No Title Available"),
             "content": preprocess_text(article.get("description", "No description available."))}
            for article in data.get("articles", [])[:15]
        ]
    except requests.exceptions.RequestException as e:
        st.error(f"\U0001F6A8 Error fetching news: {e}")
        return []

# Function to update confusion matrix
def update_confusion_matrix(true_label, predicted_label):
    cm = confusion_matrix([true_label], [predicted_label], labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Function to dynamically set background color based on prediction
def set_background(is_real):
    color = "linear-gradient(to right, #8BC34A, #4CAF50)" if is_real else "linear-gradient(to right, #FF6F61, #D32F2F)"
    page_bg = f"""
    <style>
    .stApp {{
        background: {color};
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Set initial background color to turquoise
def set_initial_background():
    page_bg = """
    <style>
    .stApp {
        background: linear-gradient(to right, #40E0D0, #20B2AA);
    }
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# UI Header
set_initial_background()
st.title("\U0001F4F0 Fake News Detector")
st.subheader("\U0001F50E Verify News Authenticity in Real Time")
st.markdown("---")

def predict_news(article):
    transform_input = vectorizer.transform([article])
    nb_prob = nb_model.predict_proba(transform_input)[0][1]  # Probability of being real
    final_prediction = nb_model.predict(transform_input)[0]  # Direct prediction (0 = Fake, 1 = Real)
    return nb_prob, final_prediction

# Function to display the result with emoji in the upper-right corner
def display_result(is_real, confidence):
    emoji_unicode = "\U0001F60A" if is_real else "\U0001F622"  # 😊 for real, 😢 for fake
    label_text = "The News is Real!" if is_real else "The News is Fake!"
    
    custom_html = f"""
    <div style="position: relative; background-color: {'#DFF2BF' if is_real else '#FFBABA'}; padding: 15px; border-radius: 10px; font-size: 20px; text-align: center;">
        <span style="position: absolute; top: 10px; right: 10px; font-size: 30px;">{emoji_unicode}</span>
        <strong>{label_text}</strong> <br>
        <span style="font-size: 16px;">Confidence: {confidence:.2f}</span>
    </div>
    """
    st.markdown(custom_html, unsafe_allow_html=True)

# User input options
option = st.radio("\U0001F4CD Choose an input method:", ["\U0001F4DD Enter News Text", "\U0001F4F0 Analyze Latest News Articles"])

if option == "\U0001F4DD Enter News Text":
    inputn = st.text_area("\U0001F4DD Enter news text:")
    if st.button("\U00002705 Check News"):
        if inputn.strip():
            preprocessed_article = preprocess_text(inputn)
            nb_prob, final_prediction = predict_news(preprocessed_article)
            set_background(final_prediction == 1)  # Set background color based on prediction
            
            display_result(final_prediction == 1, nb_prob)  # Display result with emoji in the upper-right corner
            update_confusion_matrix(1 if final_prediction == 1 else 0, final_prediction)
        else:
            st.warning("\U000026A0 Please enter some text.")

elif option == "\U0001F4F0 Analyze Latest News Articles":
    articles = fetch_political_news()
    
    if articles:
        selected_article = st.selectbox("\U0001F5DE Select a news article", [article['title'] for article in articles])
        
        if selected_article:
            article_content = next(article['content'] for article in articles if article['title'] == selected_article)
            st.markdown(f"**\U0001F4DD Selected Article:** {selected_article}")
            st.write(article_content)
            
            if st.button("\U00002705 Check News"):
                nb_prob, final_prediction = predict_news(article_content)
                set_background(final_prediction == 1)  # Set background color based on prediction
                
                display_result(final_prediction == 1, nb_prob)  # Display result with emoji in the upper-right corner
                update_confusion_matrix(1 if final_prediction == 1 else 0, final_prediction)
    else:
        st.error("\U000026A0 Could not fetch news articles. Please try again later.")
