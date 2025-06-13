import streamlit as st
import cv2
import numpy as np
import joblib
import time
import os
import gdown

# --- Constants ---
MODEL_PATH = "svm_model.pkl"
MODEL_URL = "https://drive.google.com/uc?id=1fdrlivNBQeZu1WakowTn6F5PA3LjUMN9"  # Direct download link
IMG_SIZE = 64
CLASS_NAMES = ['🐱 Cat', '🐶 Dog']

# --- Model Download ---
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# --- Load Model ---
model = joblib.load(MODEL_PATH)

# --- Page Config ---
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

# --- UI Header ---
st.title("🐾 Cat vs Dog Image Classifier")
st.write("Upload an image of a **cat or dog**, and this app will tell you what it sees using a trained SVM model.")

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload a JPEG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="📷 Uploaded Image", channels="BGR", use_column_width=True)

    # --- Preprocessing ---
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().reshape(1, -1)

    # --- Prediction ---
    start_time = time.time()
    pred = model.predict(flat)[0]
    prob = model.decision_function(flat)[0]
    elapsed = time.time() - start_time
    confidence = 1 / (1 + np.exp(-abs(prob))) * 100

    # --- Output ---
    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Prediction:** {CLASS_NAMES[pred]}")
    st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
    st.markdown(f"**Prediction Time:** `{elapsed:.4f} seconds`")
