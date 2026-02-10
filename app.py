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

# ---------------- DETAILED DISEASE INFO ----------------
disease_info = {
    "Pepper Bacterial Spot": {
        "desc": "This bacterial disease causes small, water-soaked lesions on leaves and fruit. Over time, the spots turn brown and necrotic, leading to leaf drop. It spreads rapidly in warm and humid weather conditions.",
        "remedy": "Remove and destroy infected leaves immediately to reduce spread. Avoid overhead irrigation and keep foliage dry. Apply copper-based bactericides as recommended."
    },
    "Pepper Healthy": {
        "desc": "The pepper plant shows no visible signs of disease or pest infestation. Leaves are uniformly green with proper growth. The plant is in a healthy and stable condition.",
        "remedy": "Maintain proper watering and fertilization schedules. Ensure adequate sunlight and spacing between plants. Continue regular inspection for early disease detection."
    },
    "Potato Early Blight": {
        "desc": "Early blight causes dark brown spots with concentric rings on older leaves. The disease usually begins at the lower part of the plant. It can reduce photosynthesis and yield if untreated.",
        "remedy": "Remove affected leaves and dispose of them properly. Apply recommended fungicides at early stages. Practice crop rotation to prevent recurrence."
    },
    "Potato Late Blight": {
        "desc": "Late blight is a severe fungal disease causing water-soaked lesions on leaves and stems. It spreads rapidly in cool, moist conditions. If untreated, it can destroy entire crops.",
        "remedy": "Use certified disease-free seed potatoes. Apply copper-based fungicides at the first sign of infection. Avoid excessive irrigation and maintain proper spacing."
    },
    "Potato Healthy": {
        "desc": "The potato plant appears healthy with strong green foliage. There are no visible lesions or signs of infection. Growth and leaf structure look normal.",
        "remedy": "Continue balanced fertilization and irrigation. Monitor regularly for early signs of pests or diseases. Maintain good field sanitation practices."
    },
    "Tomato Bacterial Spot": {
        "desc": "This disease causes small, dark, water-soaked spots on leaves and fruits. Over time, the spots enlarge and cause defoliation. It reduces fruit quality and yield.",
        "remedy": "Remove infected plants and avoid handling wet foliage. Use copper-based sprays as recommended. Ensure proper spacing for air circulation."
    },
    "Tomato Early Blight": {
        "desc": "Early blight produces dark spots with concentric rings on older leaves. It usually starts near the base of the plant. If untreated, it can spread upward and cause severe defoliation.",
        "remedy": "Apply fungicide at early stages of the disease. Remove infected leaves to limit spread. Practice crop rotation and avoid overhead watering."
    },
    "Tomato Late Blight": {
        "desc": "Late blight causes dark, greasy lesions on leaves and fruit. It spreads quickly during cool and moist weather. The disease can destroy entire fields within days.",
        "remedy": "Use copper-based fungicides as soon as symptoms appear. Remove infected plants immediately. Avoid excessive moisture and ensure good drainage."
    },
    "Tomato Leaf Mold": {
        "desc": "Leaf mold appears as yellow patches on the upper leaf surface. A greenish mold develops underneath the leaves. It thrives in humid and poorly ventilated environments.",
        "remedy": "Improve air circulation around plants. Reduce humidity by spacing plants properly. Apply fungicides when symptoms first appear."
    },
    "Tomato Septoria Leaf Spot": {
        "desc": "This disease causes small circular spots with dark borders on leaves. It usually begins on lower leaves and spreads upward. Severe infection leads to defoliation.",
        "remedy": "Remove infected leaves promptly. Apply recommended fungicides. Avoid overhead watering and improve air circulation."
    },
    "Tomato Spider Mites": {
        "desc": "Spider mites are tiny pests that cause yellow speckling on leaves. Leaves may dry, curl, and eventually drop. Infestation spreads quickly in hot, dry conditions.",
        "remedy": "Spray plants with water to remove mites. Use insecticidal soap or neem oil. Maintain proper humidity to discourage mites."
    },
    "Tomato Target Spot": {
        "desc": "Target spot causes circular brown lesions with concentric rings. It typically appears on leaves and stems. Severe cases can cause defoliation and fruit damage.",
        "remedy": "Remove infected leaves immediately. Apply recommended fungicides. Improve air circulation and avoid excessive moisture."
    },
    "Tomato Yellow Leaf Curl Virus": {
        "desc": "This viral disease causes yellowing and upward curling of leaves. Plants become stunted and produce fewer fruits. It spreads through whiteflies.",
        "remedy": "Control whitefly populations using insecticides or traps. Remove infected plants to prevent spread. Use resistant plant varieties if available."
    },
    "Tomato Mosaic Virus": {
        "desc": "Mosaic virus causes mottled patterns of light and dark green on leaves. It leads to distorted growth and reduced fruit yield. The virus spreads through contact.",
        "remedy": "Remove infected plants immediately. Disinfect tools and hands after handling plants. Avoid using contaminated soil or seeds."
    },
    "Tomato Healthy": {
        "desc": "The tomato plant appears healthy with vibrant green leaves. There are no visible signs of disease or pest damage. Growth is uniform and stable.",
        "remedy": "Maintain regular watering and fertilization schedules. Ensure proper sunlight and spacing. Continue periodic inspection for early detection."
    }
}

# ---------------- PDF FUNCTION ----------------
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
    elements.append(RLImage(chart_temp.name, width=400, height=250))

    doc.build(elements)
    return pdf_path

# ---------------- UI ----------------
st.markdown("<h1 style='text-align: center;'>Crop Disease Detection System</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)

    img_resized = img.resize((128,128))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    disease = class_labels[predicted_class]
    info = disease_info[disease]

    st.metric("Detected Disease", disease)
    st.metric("Confidence", f"{confidence:.2f}%")

    st.info(info["desc"])
    st.success(info["remedy"])

    # Chart
    probs = prediction[0]
    top_indices = probs.argsort()[-3:][::-1]
    top_labels = [class_labels[i] for i in top_indices]
    top_values = [probs[i]*100 for i in top_indices]

    fig, ax = plt.subplots()
    ax.bar(top_labels, top_values)
    st.pyplot(fig)

    # Downloads
    st.markdown("## Download PDF")
    pdf_path = generate_pdf(img, disease, confidence, info["desc"], info["remedy"], fig)

    report_data = {
        "disease": disease,
        "confidence": round(confidence,2),
        "description": info["desc"],
        "remedy": info["remedy"]
    }
    json_data = json.dumps(report_data, indent=4)

    col1, col2 = st.columns(2)
    with col1:
        with open(pdf_path,"rb") as f:
            st.download_button("Download PDF", f, "report.pdf")
    with col2:
        st.download_button("Download JSON", json_data, "report.json")
