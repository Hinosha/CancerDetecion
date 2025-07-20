import streamlit as st
import numpy as np
from PIL import Image
import requests
import h5py
import tensorflow as tf
import os

# Download model only if it doesn't exist
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    url = "https://drive.google.com/uc?id=1aLa3vqZ4L6aVs9NJvfyEA7fK3B274yku"
    r = requests.get(url)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)
    st.success("Model downloaded!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.title("Skin Cancer Detection")
st.write("Upload an image to classify it using the trained CNN model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))  # Adjust to your model input
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)[0]
        st.write("Prediction Scores:", prediction)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
