import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

# Helper functions
def adjust_brightness_contrast(img, brightness=0, contrast=0):
    return cv2.convertScaleAbs(img, alpha=1 + contrast/100, beta=brightness)

def apply_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_warm_filter(img):
    increase_red = cv2.add(img[:,:,2], 30)
    img[:,:,2] = np.clip(increase_red, 0, 255)
    return img

def apply_sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def portrait_blur(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    center = (img.shape[1]//2, img.shape[0]//2)
    radius = min(img.shape[0], img.shape[1])//3
    cv2.circle(mask, center, radius, 255, -1)
    blurred = cv2.GaussianBlur(img, (25,25), 0)
    return np.where(mask[:,:,None]==255, img, blurred)

def edge_detection(img):
    return cv2.Canny(img, 100, 200)

def sketch_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21,21), 0)
    sketch = cv2.divide(gray, 255-blur, scale=256)
    return sketch

def cartoon_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# Streamlit UI
st.title("Photo Editor using OpenCV + Streamlit")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.sidebar.header("Adjustments")
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", -100, 100, 0)
    img = adjust_brightness_contrast(img, brightness, contrast)

    if st.sidebar.button("Grayscale"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if st.sidebar.button("Blur"):
        img = apply_blur(img)

    if st.sidebar.button("Warm Filter"):
        img = apply_warm_filter(img)

    if st.sidebar.button("Portrait Blur"):
        img = portrait_blur(img)

    if st.sidebar.button("Sharpen"):
        img = apply_sharpen(img)

    # Extra features
    if st.sidebar.button("Edge Detection"):
        img = edge_detection(img)

    if st.sidebar.button("Sketch Effect"):
        img = sketch_effect(img)

    if st.sidebar.button("Cartoon Effect"):
        img = cartoon_effect(img)

    #  use width instead of use_column_width
    st.image(img, caption="Edited Image", width=600)

    # Download button
    result = Image.fromarray(img)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Edited Image", data=byte_im, file_name="edited.png", mime="image/png")

