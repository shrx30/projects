import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.applications import MobileNet, VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

# Load the trained model
model = joblib.load("ensemble_classifier.pkl")

# Load pre-trained CNN models
IMG_SIZE = (224, 224)
base_mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
x = GlobalAveragePooling2D()(base_mobilenet.output)
mobilenet_model = Model(inputs=base_mobilenet.input, outputs=x)

base_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
y = GlobalAveragePooling2D()(base_vgg16.output)
vgg16_model = Model(inputs=base_vgg16.input, outputs=y)

def extract_gabor_features(image):
    kernel = cv2.getGaborKernel((31, 31), 4.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    hist = cv2.calcHist([filtered], [0], None, [16], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype(np.float32) / 255.0
    return image

def extract_features(image):
    image_batch = np.expand_dims(image, axis=0)
    mobilenet_features = mobilenet_model.predict(image_batch, verbose=0)
    vgg16_features = vgg16_model.predict(image_batch, verbose=0)
    gabor_features = extract_gabor_features(image)
    combined_features = np.concatenate([mobilenet_features.flatten(), vgg16_features.flatten(), gabor_features])
    return combined_features.astype(np.float16).reshape(1, -1)

# Streamlit UI
st.title("Tuberculosis Detection System")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Uploaded Image", width=200)
    
    processed_image = preprocess_image(image)
    features = extract_features(processed_image)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    st.write("### Prediction Result")
    if prediction == 1:
        st.error(f"**TB Detected** (Confidence: {probability:.2f})")
    else:
        st.success(f"**Normal** (Confidence: {1 - probability:.2f})")
