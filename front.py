import streamlit as st
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.layers import TextVectorization

# --- Load vectorizer and model only once ---
@st.cache_resource
def load_vectorizer_and_model():
    df = pd.read_csv(os.path.join('data', 'train.csv'))
    X = df['comment_text']
    
    MAX_FEATURES = 200000    
    vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
    vectorizer.adapt(X.values)
    
    model = load_model(os.path.join('models', 'Toxicity_final.h5'))
    
    return vectorizer, model

vectorizer, model = load_vectorizer_and_model()

# --- Predict function ---
def predict(input_text):
    vectorized_comment = vectorizer([input_text])
    result = model.predict(vectorized_comment)
    prediction = (result > 0.5).astype(int)
    
    return "✅ The comment is Not Toxic" if prediction[0][0] == 0 else "⚠️ The comment is Toxic"

# --- Inject enhanced CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(to bottom right, #002366, #4169E1);
        color: #ffffff;
    }

    textarea {
        background-color: rgba(255, 255, 255, 0.8);
        color: #1a1a1a !important;
        border: 2px solid #1d4ed8;
        border-radius: 12px;
        padding: 14px;
        font-size: 16px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        transition: 0.3s ease;
    }

    textarea:focus {
        border-color: #60a5fa;
        box-shadow: 0 0 12px rgba(96, 165, 250, 0.6);
        outline: none;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1e40af, #60a5fa);
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 30px;
        padding: 12px 30px;
        border: none;
        cursor: pointer;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05) translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
    }

    .stMarkdown h1 {
        font-size: 2.8rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.5);
    }

    .result-box {
        padding: 15px;
        background-color: rgba(255,255,255,0.2);
        border-radius: 12px;
        border: 1px solid #60a5fa;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }

    .char-counter {
        font-size: 13px;
        color: #d1d5db;
        text-align: right;
        margin-top: -10px;
        margin-bottom: 15px;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background-color: #1e40af;
        border-radius: 8px;
    }

    hr {
        border: 1px solid #60a5fa;
        margin: 2rem 0;
    }

    .footer {
        margin-top: 40px;
        font-size: 14px;
        color: #dddddd;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- UI Content ---
st.markdown("<h1>🔍 Toxic Comment Classifier</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3>Detect Toxic Comments with AI</h3>", unsafe_allow_html=True)
st.markdown("""
    <p>Enter a comment below to check if it's toxic or not. 
    The model will analyze the text and provide feedback on its toxicity.</p>
""", unsafe_allow_html=True)

user_input = st.text_area("👇👇👇", placeholder="Type your comment here...", height=150)
st.markdown(f"<div class='char-counter'>📝 {len(user_input)} characters</div>", unsafe_allow_html=True)

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        if len(user_input.strip()) == 0:
            st.warning("⚠️ Please enter a comment first.")
        else:
            result = predict(user_input)
            result_style = f"<div class='result-box'>{result}</div>"
            st.markdown(result_style, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <div class="footer">
        Made using Streamlit | Developed by <strong>Samarth Singh</strong>
    </div>
""", unsafe_allow_html=True)


