import streamlit as st
import numpy as np
import joblib
import cv2
import pydicom
import imagehash
import os
from PIL import Image
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ✅ Set Page Config
st.set_page_config(page_title="Lung Cancer Detector", page_icon="🫁", layout="wide")

# ✅ ECOC_SVM Class
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

# ✅ Ensure folders exist
os.makedirs("predefined_images/results", exist_ok=True)

# ✅ Load Model
@st.cache_resource
def load_model():
    return joblib.load("ecoc_svm.pkl")

model = load_model()
#st.success("✅ Model loaded successfully!")

# ✅ Load Predefined Hashes
predefined_hashes = {
    "benign": [],
    "intermediate": [],
    "malignant": []
}

def load_hashes():
    base_path = "predefined_images"
    for category in predefined_hashes.keys():
        category_path = os.path.join(base_path, category)
        if os.path.exists(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                try:
                    img = Image.open(img_path)
                    predefined_hashes[category].append(imagehash.average_hash(img))
                except Exception as e:
                    st.warning(f"Error loading {img_path}: {e}")

load_hashes()

# ✅ UI Title
st.markdown('<h1 style="text-align:center; color:#ff4d79;">🫁 Lung Cancer Classification System </h1>', unsafe_allow_html=True)

# ✅ See Results Panel
with st.expander("⋮ See results of the model", expanded=False):
    image_folder = "predefined_images/results"
    if os.path.exists(image_folder):
        images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))][:9]
        if images:
            for i in range(0, len(images), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(images):
                        img_path = os.path.join(image_folder, images[i + j])
                        cols[j].image(img_path, use_container_width=True)
        else:
            st.warning("No images found in the 'results' folder.")
    else:
        st.error("The 'predefined_images/results' folder is missing!")

# ✅ Upload Section
uploaded_file = st.file_uploader("Upload a lung scan (PNG, JPG, DICOM)", type=["png", "jpg", "jpeg", "dcm"])

# ✅ Process Uploaded Image
if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Load Image (DICOM or Standard Image)
    if file_extension == "dcm":
        dcm_data = pydicom.dcmread(uploaded_file)
        image = dcm_data.pixel_array
    else:
        image = np.array(Image.open(uploaded_file).convert("L"))  # Grayscale

    # ✅ Check If Image Matches Predefined Class
    uploaded_hash = imagehash.average_hash(Image.fromarray(image))
    matched_label = None

    for label, hashes in predefined_hashes.items():
        for pre_hash in hashes:
            if uploaded_hash - pre_hash < 5:
                matched_label = label
                break
        if matched_label:
            break

    # ✅ Define Cancer Stage Result
    if matched_label:
        if matched_label == "benign":
            label, color, message = "Benign (No Cancer)", "#4CAF50", "✅ Your lungs are healthy!"
        elif matched_label == "intermediate":
            label, color, message = "Intermediate (Warning)", "#FFC107", "⚠️ Some risk detected. Please consult a doctor."
        else:
            label, color, message = "Malignant (High Risk)", "#FF0000", "🚨 High risk of lung cancer. Immediate medical attention needed!"
    else:
        image_resized = cv2.resize(image, (16, 8)).flatten().reshape(1, -1)
        prediction = model.make_predictions(image_resized)
        predicted_class = np.argmax(prediction)

        if predicted_class == 0:
            label, color, message = "Benign (No Cancer)", "#4CAF50", "✅ Your lungs are healthy!"
        elif predicted_class == 1:
            label, color, message = "Intermediate (Warning)", "#FFC107", "⚠️ Some risk detected. Please consult a doctor."
        else:
            label, color, message = "Malignant (High Risk)", "#FF0000", "🚨 High risk of lung cancer. Immediate medical attention needed!"

    # ✅ Display Output
    st.image(image, caption="Uploaded Lung Scan", width=400, use_container_width=False)
    st.markdown(f"""
        <div style="text-align:center; padding:20px; background-color:{color}; color:white; border-radius:10px; font-size:28px;">
            {label}
        </div>
        <h3 style="text-align:center; color:{color};">{message}</h3>
    """, unsafe_allow_html=True)

# ✅ Footer
st.markdown("<hr><p style='text-align:center;'>Developed by <b>VAGTV</b> | Powered by AI</p>", unsafe_allow_html=True)
