import streamlit as st
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import tempfile
import os


st.markdown(
    """
    <style>
    .title {
        font-size: 32px;
        font-weight: 700;
        color: #2c3e50;
        padding-bottom: 10px;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .uploader {
        background-color: #f5f7fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .extracted-text {
        font-family: 'Courier New', Courier, monospace;
        font-size: 16px;
        background-color: #f0f0f0;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">GP-AI OCR Prototype (Images only)</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="uploader">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)


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

    st.markdown(f'<div class="extracted-text">{extracted_text}</div>', unsafe_allow_html=True)

    image = Image.open(tmp_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    os.unlink(tmp_path)
