import streamlit as st
import numpy as np
import joblib
import cv2
import pydicom
from PIL import Image
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ✅ Step 1: Set Page Config
st.set_page_config(page_title="Lung Cancer Detector", page_icon="🫁", layout="wide")

# ✅ Step 2: ECOC_SVM Class
class ECOC_SVM:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.svm_models = {}

    def train_classifiers(self, features, labels):
        for i in range(self.num_classes):
            svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))
            svm_model.fit(features, (labels == i).astype(int))
            self.svm_models[i] = svm_model

    def make_predictions(self, input_features):
        predictions = np.zeros((input_features.shape[0], self.num_classes))
        for i in range(self.num_classes):
            predictions[:, i] = self.svm_models[i].predict_proba(input_features)[:, 1]
        return predictions

# ✅ Step 3: Load Model
@st.cache_resource
def load_model():
    return joblib.load("ecoc_svm.pkl")

model = load_model()
st.success("✅ Model loaded successfully!")

# ✅ Step 4: Custom CSS (Light Pink Theme)
st.markdown("""
    <style>
        .stApp { background: linear-gradient(to right, #FFDEE9, #B5FFFC); color: black; }
        .title { font-size: 42px; font-weight: bold; color: #ff4d79; text-align: center; }
        .result-box { text-align: center; padding: 20px; border-radius: 10px; font-size: 28px; color: white; }
    </style>
""", unsafe_allow_html=True)

# ✅ Step 5: Title & Upload Section
st.markdown('<p class="title">🫁 Lung Cancer Classification System 🚀</p>', unsafe_allow_html=True)
st.write("## Upload a Lung Scan to Detect Cancer Stage")
uploaded_file = st.file_uploader("Upload a lung scan (PNG, JPG, DICOM)", type=["png", "jpg", "jpeg", "dcm"])

# ✅ Step 6: Process Uploaded Image
if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Load image (DICOM or Standard Image)
    if file_extension == "dcm":
        dcm_data = pydicom.dcmread(uploaded_file)
        image = dcm_data.pixel_array
    else:
        image = np.array(Image.open(uploaded_file).convert("L"))  # Convert to grayscale

    # Resize Image for Model (Standard Input Size)
    image_resized = cv2.resize(image, (16, 8)).flatten().reshape(1, -1)  # Match model input size

    # ✅ Step 7: Prediction
    prediction = model.make_predictions(image_resized)  # Returns probabilities
    predicted_class = np.argmax(prediction)  # Extract highest probability class

    # ✅ Step 8: Define Cancer Stages
    if predicted_class == 0:
        label, color, message = "Benign (No Cancer)", "#4CAF50", "✅ Your lungs are healthy!"
    elif predicted_class == 1:
        label, color, message = "Intermediate (Warning)", "#FFC107", "⚠️ Some risk detected. Please consult a doctor."
    else:
        label, color, message = "Malignant (High Risk)", "#FF0000", "🚨 High risk of lung cancer. Immediate medical attention needed!"

    # ✅ Step 9: Display Image & Result
    st.image(image, caption="Uploaded Lung Scan", use_column_width=True)
    st.markdown(f"""
        <div class="result-box" style="background-color: {color};">
            {label}
        </div>
        <h3 style="text-align:center; color: {color};">{message}</h3>
    """, unsafe_allow_html=True)

# ✅ Step 10: Footer
st.markdown("<hr><p style='text-align:center;'>Developed by <b>Katlaganti Tejkumar Reddy</b> | Powered by AI</p>", unsafe_allow_html=True)
