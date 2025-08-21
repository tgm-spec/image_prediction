import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load models
@st.cache_resource
def load_imagenet_model():
    return MobileNetV2(weights="imagenet")
imagenet_model = load_imagenet_model()

@st.cache_resource
def load_kiwi_model():
    return load_model("kiwi.h5")
kiwi_model = load_kiwi_model()

st.title("🔍 Image Detection Demo")
st.write("Uploads are first checked for Kiwi. If not Kiwi, general predictions are shown.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Uploaded Image", use_container_width=True)

    # ---------- Kiwi Prediction ----------
    kiwi_input = original_image.resize((224, 224))
    arr_kiwi = np.array(kiwi_input)
    arr_kiwi = np.expand_dims(arr_kiwi, axis=0)
    arr_kiwi = arr_kiwi / 255.0

    with st.spinner("Checking the uploaded image..."):
        preds_kiwi = kiwi_model.predict(arr_kiwi)
        confidence = preds_kiwi[0][0]

    if confidence >= 0.5:
        st.subheader("Kiwi fruit Detected!")
        st.write(f"**Kiwi** — {confidence*100:.2f}%")
    else:
        # ---------- Fall back to ImageNet ----------
        imagenet_input = original_image.resize((224, 224))
        arr_im = np.array(imagenet_input)
        arr_im = np.expand_dims(arr_im, axis=0)
        arr_im = preprocess_input(arr_im)

        with st.spinner("Analyzing with ImageNet model..."):
            preds_im = imagenet_model.predict(arr_im)
            decoded = decode_predictions(preds_im, top=3)[0]

        st.subheader("ImageNet Prediction Results:")
        for _, name, prob in decoded:
            st.write(f"**{name}** — {prob*100:.2f}%")
