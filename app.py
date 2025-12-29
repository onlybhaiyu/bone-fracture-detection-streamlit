import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Bone Fracture Detection")

st.title("🦴 Bone Fracture Detection")
st.write("Upload an X-ray image to detect fractures")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Fracture"):
        with st.spinner("Detecting..."):
            results = model(image, conf=0.1)
            result_image = results[0].plot()
            st.image(result_image, caption="Detection Result", use_column_width=True)
