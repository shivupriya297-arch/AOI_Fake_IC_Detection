import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI-Driven Fake IC Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Driven Fake IC Detection")
st.markdown("### College Project Demo")

uploaded_file = st.file_uploader(
    "📂 Upload IC Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

else:

    image = cv2.imread("images/original/ic2.jpg")

if image is None:

    st.error("❌ Image not found!")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼 Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")

with st.spinner("🤖 AI is analyzing the IC image..."):
    import time
    time.sleep(2)    

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3,3),0)

edges = cv2.Canny(blur,50,150)

_, thresh = cv2.threshold(
    blur,
    120,
    255,
    cv2.THRESH_BINARY
)

with col2:
    st.subheader("⚫ Processed Image")
    st.image(thresh,width="stretch")

white_pixels = np.sum(thresh==255)

black_pixels = np.sum(thresh==0)

total = white_pixels + black_pixels

confidence = round(
    (max(white_pixels,black_pixels)/total)*100,
    2
)

st.markdown("---")

if white_pixels > black_pixels:
    st.markdown(f"""
    <div style="
        background-color:#d4edda;
        padding:20px;
        border-radius:15px;
        text-align:center;
        font-size:26px;
        font-weight:bold;
        color:green;">
        ✅ GENUINE IC DETECTED<br>
        Confidence : {confidence}%
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="
        background-color:#f8d7da;
        padding:20px;
        border-radius:15px;
        text-align:center;
        font-size:26px;
        font-weight:bold;
        color:red;">
        ❌ FAKE IC DETECTED<br>
        Confidence : {confidence}%
    </div>
    """, unsafe_allow_html=True)

st.subheader("📊 Pixel Analysis")

c1,c2,c3=st.columns(3)

c1.metric(
    "White Pixels",
    white_pixels
)

c2.metric(
    "Black Pixels",
    black_pixels
)

c3.metric(
    "Confidence %",
    confidence
)

st.progress(confidence/100)

st.subheader("📊 Pixel Distribution")

fig, ax = plt.subplots()

ax.pie(
    [white_pixels, black_pixels],
    labels=["White Pixels", "Black Pixels"],
    autopct="%1.1f%%",
    startangle=90
)

ax.axis("equal")

st.pyplot(fig)

_,buffer=cv2.imencode(".jpg",thresh)

st.download_button(

    "📥 Download Processed Image",

    buffer.tobytes(),

    file_name="processed_ic.jpg",

    mime="image/jpeg"

)

st.markdown("---")

st.info("""

### 📖 About Project

This project detects fake IC images using:

- Image Preprocessing
- Grayscale Conversion
- Gaussian Blur
- Thresholding
- Pixel Analysis

⚠ Note:
This is a **college demonstration project**.
It does not use a trained AI model. The current version uses image processing techniques to simulate the detection workflow.

Future enhancement:
CNN / Deep Learning / YOLO based detection.

""")