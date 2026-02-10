import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import json

# PDF libraries
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

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

default_remedy = {
    "desc": "Disease detected in plant.",
    "remedy": "Remove infected parts and apply appropriate fungicide or pesticide."
}

# ---------------- PDF GENERATION FUNCTION ----------------
def generate_pdf(image, disease, confidence, description, remedy, chart_fig):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp_file.name

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Crop Disease Diagnosis Report", styles['Title']))
    elements.append(Spacer(1, 10))

    img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(img_temp.name)

    elements.append(Paragraph("<b>Uploaded Leaf Image:</b>", styles['Heading2']))
    elements.append(RLImage(img_temp.name, width=200, height=200))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Disease:</b> {disease}", styles['Normal']))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Description:</b>", styles['Heading2']))
    elements.append(Paragraph(description, styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Recommended Remedy:</b>", styles['Heading2']))
    elements.append(Paragraph(remedy, styles['Normal']))
    elements.append(Spacer(1, 15))

    chart_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    chart_fig.savefig(chart_temp.name, bbox_inches='tight')

    elements.append(Paragraph("<b>Prediction Confidence Chart:</b>", styles['Heading2']))
    elements.append(RLImage(chart_temp.name, width=400, height=250))

    doc.build(elements)
    return pdf_path

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

    img_resized = img.resize((128,128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    disease = class_labels[predicted_class]

    info = disease_info.get(disease, default_remedy)

    st.markdown("## 🔍 Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Disease", disease)
    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    st.markdown("### 🩺 Disease Description")
    st.info(info["desc"])

    st.markdown("### 🌱 Recommended Remedy")
    st.success(info["remedy"])

    # ---------------- CHART ----------------
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

    # ---------------- DOWNLOAD SECTION ----------------
    st.markdown("## 📥 Download Report")

    # Generate PDF
    pdf_path = generate_pdf(
        img,
        disease,
        confidence,
        info["desc"],
        info["remedy"],
        fig
    )

    # Generate JSON
    report_data = {
        "disease": disease,
        "confidence": round(confidence, 2),
        "description": info["desc"],
        "remedy": info["remedy"]
    }
    json_data = json.dumps(report_data, indent=4)

    col1, col2 = st.columns(2)

    with col1:
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download PDF",
                data=f,
                file_name="crop_diagnosis_report.pdf",
                mime="application/pdf"
            )

    with col2:
        st.download_button(
            label="🧾 Download JSON",
            data=json_data,
            file_name="crop_diagnosis_report.json",
            mime="application/json"
        )
