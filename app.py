
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Define image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256

# 2. Define the class_to_idx dictionary
class_to_idx = {
    'Potato___Late_blight': 0,
    'Potato___Early_blight': 1,
    'Potato___healthy.zip': 2 # Note: Using '.zip' as part of the class name for consistency with dataset loading
}

# 3. Load the trained model
@st.cache_resource
def load_my_model():
    model_path = 'potato_disease_cnn_model.keras'
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'. Please ensure the model is in the correct directory.")
        st.stop()
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model

model = load_my_model()

# Reverse the class_to_idx dictionary for displaying predictions
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 4. Set up the Streamlit application title and description
st.title("Potato Leaf Disease Classifier")
st.write("Upload an image of a potato leaf to classify if it's healthy, or has Early Blight or Late Blight.")

# 5. Implement an image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image
    # Resize the image
    img_resized = image.resize((IMG_HEIGHT, IMG_WIDTH))
    # Convert to NumPy array
    img_array = np.array(img_resized)
    # Expand dimensions to create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)

    # Interpret the prediction
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = idx_to_class[predicted_class_index]
    confidence = np.max(predictions) * 100

    # Display the predicted class name
    st.write(f"Prediction: **{predicted_class_name.replace('___', ' ').replace('.zip', '')}**")
    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Raw Probabilities:")
    for i, class_name in sorted(idx_to_class.items()):
        st.write(f"  {class_name.replace('___', ' ').replace('.zip', '')}: {predictions[0][i]*100:.2f}%")
