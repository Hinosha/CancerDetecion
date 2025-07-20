# 🔬 Skin Cancer Detection App using CNN and Streamlit

This Streamlit app uses a pre-trained Convolutional Neural Network (CNN) to detect types of skin cancer from dermatoscopic images. The model was trained on the HAM10000 dataset and saved as a `.h5` file for efficient deployment.

---

## 🚀 Features

- Upload any dermatoscopic skin image (`.jpg`, `.png`, `.jpeg`)
- Predict the most likely skin condition (e.g., melanoma, nevus, etc.)
- Displays confidence level for the prediction
- Built with TensorFlow and deployed using Streamlit

---

## 🧠 Model Info

- Architecture: CNN with multiple Conv2D and MaxPooling layers
- Input size: 224x224 pixels RGB
- Output: Softmax probabilities across skin disease classes
- Trained on: HAM10000 dataset
- Saved model: `skin_cancer_model.h5` (included in this repo)

---

## 📂 Project Structure

skin-cancer-classifier/
├── app.py # Main Streamlit app
├── skin_cancer_model.h5 # Pre-trained CNN model
├── requirements.txt # Project dependencies
└── README.md # Project info and instructions


---



