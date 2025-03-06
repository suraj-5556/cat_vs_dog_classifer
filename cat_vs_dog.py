import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image

# Load your trained model
MODEL_PATH = "cat_dog_model.joblib"
model = joblib.load(MODEL_PATH)

def preprocess_image(image):
    image = image.resize((256, 256)) 
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

def predict(image):
    prediction = model.predict(image)[0]  # Get prediction
    label = 1 if prediction > 0.5 else 0  # Convert to label (dog=1, cat=0)
    confidence = float(prediction) if label == 1 else float(1 - prediction)
    return {"label": label, "confidence": confidence}

st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide")  # Use wide layout to avoid scrolling
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image to find out whether it's a cat or a dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])  # Create two columns for layout
    
    with col1:
        st.image(image, caption="Uploaded Image", width=400)  # Increase image size
    
    with col2:
        st.write("Processing image...")
        processed_image = preprocess_image(image)
        
        with st.spinner("Classifying..."):
            result = predict(processed_image)
        
        if result:
            label = "Dog 🐶" if result["label"] == 1 else "Cat 🐱"
            confidence = result["confidence"] * 100
            st.success(f"**Prediction:** {label}\n**Confidence:** {confidence:.2f}%")
        else:
            st.error("Error in prediction. Please try again!")
