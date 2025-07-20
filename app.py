import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
import os
from PIL import Image

# Constants
MODEL_PATH = "skin_cancer_model.h5"
GOOGLE_DRIVE_ID = "1aLa3vqZ4L6aVs9NJvfyEA7fK3B274yku"
LABELS = ["Melanoma", "Nevus", "Seborrheic Keratosis"]

# Download the model if not already available
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}", MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.title("🧪 Skin Cancer Detection App")
st.write("Upload a skin lesion image and the model will predict the type of skin condition.")

uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Convert RGBA to RGB
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = LABELS[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.markdown(f"### 🩺 Prediction: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")
