import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os
from PIL import Image

# Title
st.title("🧠 Skin Cancer Detection")
st.write("Upload a skin image to predict the type of lesion using a pre-trained CNN model.")

# Download model from Google Drive if not present
MODEL_PATH = "skin_cancer_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1aLa3vqZ4L6aVs9NJvfyEA7fK3B274yku"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (adjust based on your dataset)
class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 
               'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

# Upload image
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Predict
    prediction = model.predict(x)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🔍 Predicted: **{predicted_class}**")
