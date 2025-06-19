import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
from tensorflow.keras.models import load_model

# --- Page Config ---
st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide", page_icon="🤖")


MODEL_URL = "https://drive.google.com/file/d/1d1_t3NNWxuCJxnbgUblh5aUqY9jCkSOy"  # Replace with actual ID
MODEL_PATH = "cat_dog_cnn_model.keras"

@st.cache_resource
def load_cnn_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.warning("📥 Downloading model from Google Drive...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

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
