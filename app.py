import streamlit as st
import numpy as np
import joblib
import cv2
import pydicom
import os
from PIL import Image
from sklearn.svm import SVC  # Ensure correct import
import joblib
from sklearn.svm import SVC  # Ensure SVM is imported

# Load the model
model = joblib.load("ecoc_svm.pkl")
print("✅ Model loaded successfully!")

# Page Configuration
st.set_page_config(page_title="Lung Cancer Detector", page_icon="🫁", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stApp {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #ffcc00;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🫁 Lung Cancer Classification System 🚀</p>', unsafe_allow_html=True)
st.write("## Upload a Lung Scan to Detect Cancer Stage")

# File Uploader
uploaded_file = st.file_uploader("Upload a lung scan (PNG, JPG, DICOM)", type=["png", "jpg", "jpeg", "dcm"])

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Load Image
    if file_extension == "dcm":
        dcm_data = pydicom.dcmread(uploaded_file)
        image = dcm_data.pixel_array
    else:
        image = np.array(Image.open(uploaded_file))

    # Resize Image for Model (Assume model takes 224x224 images)
    image_resized = cv2.resize(image, (224, 224)).reshape(1, -1)  # Flattening for SVM

    # Prediction
    prediction = model.predict(image_resized)[0]

    # Define Cancer Stage with Colors
    if prediction == 0:
        label, color, message = "Benign (No Cancer)", "#4CAF50", "✅ Your lungs are healthy!"
    elif prediction == 1:
        label, color, message = "Intermediate (Warning)", "#FFC107", "⚠️ Some risk detected. Please consult a doctor."
    else:
        label, color, message = "Malignant (High Risk)", "#FF0000", "🚨 High risk of lung cancer. Immediate medical attention needed!"

    # Display Image
    st.image(image, caption="Uploaded Lung Scan", use_column_width=True)

    # Display Prediction Result
    st.markdown(f"""
        <div style="text-align:center; padding: 20px; border-radius: 10px; background-color: {color}; color: white; font-size: 28px;">
            {label}
        </div>
        <h3 style="text-align:center; color: {color};">{message}</h3>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<hr><p style='text-align:center;'>Developed by <b>Katlaganti Tejkumar Reddy</b> | Powered by AI</p>", unsafe_allow_html=True)
