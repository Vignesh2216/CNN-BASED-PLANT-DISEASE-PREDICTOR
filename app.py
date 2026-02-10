import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crop Disease Detector", layout="centered")

MODEL_PATH = "crop_model_15classes.keras"
FILE_ID = "1UCwUCrrVmFL2NifYhbrJwW4NsVjGLZCV"

# ---------------- DOWNLOAD MODEL ----------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# ---------------- CLASS LABELS ----------------
class_labels = {
    0: "Pepper Bacterial Spot",
    1: "Pepper Healthy",
    2: "Potato Early Blight",
    3: "Potato Late Blight",
    4: "Potato Healthy",
    5: "Tomato Bacterial Spot",
    6: "Tomato Early Blight",
    7: "Tomato Late Blight",
    8: "Tomato Leaf Mold",
    9: "Tomato Septoria Leaf Spot",
    10: "Tomato Spider Mites",
    11: "Tomato Target Spot",
    12: "Tomato Yellow Leaf Curl Virus",
    13: "Tomato Mosaic Virus",
    14: "Tomato Healthy"
}

# ---------------- DISEASE INFO ----------------
disease_info = {
    "Tomato Target Spot": {
        "desc": "Fungal disease causing brown spots with concentric rings.",
        "remedy": "Use fungicides and remove infected leaves. Improve air circulation."
    },
    "Tomato Early Blight": {
        "desc": "Causes dark spots with concentric rings on older leaves.",
        "remedy": "Apply fungicide and remove affected leaves."
    },
    "Tomato Late Blight": {
        "desc": "Serious fungal disease causing dark lesions on leaves and fruit.",
        "remedy": "Use copper-based fungicides and avoid overhead watering."
    },
    "Tomato Healthy": {
        "desc": "The plant appears healthy with no visible disease.",
        "remedy": "Maintain proper watering, sunlight, and nutrition."
    }
}

# Default remedy
default_remedy = {
    "desc": "Disease detected in plant.",
    "remedy": "Remove infected parts and apply appropriate fungicide or pesticide."
}

# ---------------- UI HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57;'>
        🌿 Crop Disease Detection System
    </h1>
    <p style='text-align: center;'>
        Upload a leaf image to detect disease and get treatment advice.
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf", use_container_width=True)

    # Preprocess image
    img_resized = img.resize((128,128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    disease = class_labels[predicted_class]

    # Get remedy
    info = disease_info.get(disease, default_remedy)

    # ---------------- RESULT SECTION ----------------
    st.markdown("## 🔍 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Detected Disease", value=disease)

    with col2:
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

    st.markdown("### 🩺 Disease Description")
    st.info(info["desc"])

    st.markdown("### 🌱 Recommended Remedy")
    st.success(info["remedy"])

    # ---------------- GRAPHICAL REPRESENTATION ----------------
    st.markdown("### 📊 Prediction Confidence")

    probs = prediction[0]
    top_indices = probs.argsort()[-3:][::-1]

    top_labels = [class_labels[i] for i in top_indices]
    top_values = [probs[i] * 100 for i in top_indices]

    chart_data = pd.DataFrame({
        "Disease": top_labels,
        "Confidence (%)": top_values
    })

    fig, ax = plt.subplots()
    ax.bar(chart_data["Disease"], chart_data["Confidence (%)"])
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Top 3 Predictions")
    plt.xticks(rotation=20)

    st.pyplot(fig)
