import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# --- Page Config ---
st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide", page_icon="🤖")

# --- Constants ---
MODEL_PATH = "cat_dog_cnn_model.keras"
IMG_SIZE = 128
CLASS_NAMES = ['🐱 Cat', '🐶 Dog']

# --- Custom CSS for high-tech style ---
st.markdown("""
    <style>
        .reportview-container {
            background-color: #0f0f0f;
            color: #f0f0f0;
        }
        .block-container {
            padding: 2rem 2rem 2rem 2rem;
        }
        .stButton>button {
            background-color: #00ffcc;
            color: black;
            font-weight: bold;
            border-radius: 10px;
        }
        .stFileUploader {
            border: 2px dashed #00ffcc;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_cnn_model():
    if not os.path.isfile(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    return load_model(MODEL_PATH)

# Load the model safely
model = load_cnn_model()

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>Cat vs Dog Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload an image below to let our Neural Network decide — Cat or Dog?</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Upload + Predict in One Frame ---
uploaded_file = st.file_uploader("🚀 Upload your image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Preprocess image
    img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = CLASS_NAMES[class_index]

    # --- Layout in Two Columns ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, width=300, caption="Preview")


    with col2:
        st.markdown("## 🔍 Prediction Result")
        st.markdown(f"### 🧠 Model Prediction: `{predicted_class}`")
        st.markdown(f"### ⚡ Confidence: `{confidence:.2f}%`")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>High-Tech Classifier Interface | Yuvraj Singh & Nexa AI Systems</p>", unsafe_allow_html=True)
