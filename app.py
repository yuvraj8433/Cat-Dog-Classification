import streamlit as st
import cv2
import numpy as np
import joblib
import time

# Constants
IMG_SIZE = 64
CLASS_NAMES = ['🐱 Cat', '🐶 Dog']

# Load model
model = joblib.load("svm_model.pkl")

# Page config
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

# UI Title
st.title("🐾 Cat vs Dog Image Classifier")
st.write("Upload an image of a **cat or dog**, and this app will tell you what it sees using a trained SVM model.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a JPEG/PNG image", type=["jpg", "jpeg", "png"])

# On image upload
if uploaded_file is not None:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display original image
    st.image(image, caption="📷 Uploaded Image", channels="BGR", use_column_width=True)

    # Preprocess
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().reshape(1, -1)

    # Predict with timer
    start_time = time.time()
    pred = model.predict(flat)[0]
    prob = model.decision_function(flat)[0]  # Confidence distance from hyperplane
    elapsed = time.time() - start_time

    # Normalize confidence to a 0-100 scale using sigmoid approximation
    confidence = 1 / (1 + np.exp(-abs(prob))) * 100

    # Output
    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Prediction:** {CLASS_NAMES[pred]}")
    st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
    st.markdown(f"**Prediction Time:** `{elapsed:.4f} seconds`")
