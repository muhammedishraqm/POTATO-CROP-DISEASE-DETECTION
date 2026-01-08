import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model_path = 'potato_disease_cnn_model.keras'
    
    # Check if the file exists. If not, download it from Google Drive.
    if not os.path.exists(model_path):
        # This is the ID for your specific file on Drive
        file_id = '1EqynfWp6UJrcSWGsbu7QynDkphNLN8ko'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)

    return tf.keras.models.load_model(model_path)

st.title("Potato Disease Classifier")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Image uploader
file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).resize((256, 256)) 
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    img_array = np.array(image)
    img_batch = np.expand_dims(img_array, axis=0)
    
    if st.button("Predict"):
        prediction = model.predict(img_batch)
        st.write(f"Prediction: {prediction}")

