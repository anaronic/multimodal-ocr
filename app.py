import streamlit as st
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import tempfile
import os

st.title("GPAI OCR Prototype")

uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    doc = DocumentFile.from_images(tmp_path)
    ocr_model = ocr_predictor(pretrained=True)
    result = ocr_model(doc)

    st.header("Extracted Text")
    extracted_text = result.render()
    st.text_area("", extracted_text, height=300)

    image = Image.open(tmp_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    os.unlink(tmp_path)
