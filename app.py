import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model_path = 'potato_disease_cnn_model.keras'
    
    # Check if the file exists locally. If not, download it from Google Drive.
    if not os.path.exists(model_path):
        # Your specific Google Drive File ID
        file_id = '1EqynfWp6UJrcSWGsbu7QynDkphNLN8ko'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)

    return tf.keras.models.load_model(model_path)

st.title("Potato Disease Classifier")

# Load the model
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Image uploader
file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if file:
    # 1. Open and resize the image to match model input (256x256)
    image = Image.open(file).resize((256, 256)) 
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 2. Convert image to numpy array and create a batch
    img_array = np.array(image)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 3. Predict
    if st.button("Predict"):
        prediction = model.predict(img_batch)
        
        # Define the class names in the exact order
        class_names = ['Early Blight', 'Late Blight', 'Healthy']

        # Find the index of the highest probability
        predicted_index = np.argmax(prediction)
        
        # Get the corresponding class name
        result_name = class_names[predicted_index]
        
        # Optional: Get the confidence score (e.g., 98%)
        confidence = np.max(prediction) * 100

        # Display the results
        st.write("---")
        st.markdown(f"### Prediction: **{result_name}**")
        st.info(f"Confidence: {confidence:.2f}%")


