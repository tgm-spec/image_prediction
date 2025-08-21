import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

st.title("🔍 Image Detection - Demo with Kiwi Check")
st.write("Upload an image to see predictions. Kiwi detection has priority over general ImageNet predictions.")

# Load models
@st.cache_resource
def load_imagenet_model():
    return MobileNetV2(weights="imagenet")

@st.cache_resource
def load_kiwi_model():
    return load_model("kiwi.h5")  # Your fine-tuned Kiwi model

imagenet_model = load_imagenet_model()
kiwi_model = load_kiwi_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- Kiwi prediction ---
    kiwi_input = image.resize((160, 160))
    kiwi_arr = np.expand_dims(np.array(kiwi_input) / 255.0, axis=0)
    kiwi_pred = kiwi_model.predict(kiwi_arr)
    kiwi_conf = float(kiwi_pred[0][0])  # single-class output

    if kiwi_conf > 0.5:
        st.subheader("Prediction Results:")
        st.write(f"**Kiwi detected** — Confidence: {kiwi_conf*100:.2f}%")
    else:
        # --- Fallback to ImageNet ---
        img_input = image.resize((224, 224))
        arr = np.expand_dims(np.array(img_input), axis=0)
        arr = preprocess_input(arr)
        preds = imagenet_model.predict(arr)
        decoded = decode_predictions(preds, top=3)[0]

        st.subheader("Prediction Results (Pretrained ImageNet):")
        for _, name, prob in decoded:
            st.write(f"**{name}** — {prob*100:.2f}%")
