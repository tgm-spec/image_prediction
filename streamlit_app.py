import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)

# Load demo model (pre-trained ImageNet model)
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

st.title("🔍 Image Detection - Demo Prototype")
st.write("This is a demo test using MobileNetV2. Upload an image to see predictions.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        arr = np.array(image)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        preds = model.predict(arr)
        decoded = decode_predictions(preds, top=3)[0]

    st.subheader("Prediction Results (Demo):")
    for _, name, prob in decoded:
        st.write(f"**{name}** — {prob*100:.2f}%")
