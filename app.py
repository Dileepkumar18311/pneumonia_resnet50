import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- Page Config ---
st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")

st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image to see if the AI detects signs of Pneumonia.")

# --- Load Model ---
# @st.cache_resource ensures the model is only loaded once, saving time
@st.cache_resource
def load_model():
    # Your local model path
    model_path = r"C:\Users\dilee\Desktop\Mdel\pneumonia_resnet50_model.keras"
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model. Error: {e}")
    st.stop()

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an X-ray image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Create columns to make the UI look cleaner
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded X-ray', use_container_width=True)
        
    with col2:
        st.write("### Analysis Results")
        with st.spinner("Analyzing image..."):
            # Preprocess the image to match the ResNet50 input format
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0) # Create a batch of 1
            img_array = preprocess_input(img_array)       # Apply ResNet50 preprocessing
            
            # Make prediction
            prediction = model.predict(img_array)[0][0]
            
            # Classify based on the sigmoid output (0.5 threshold)
            # In your dataset structure: 0 = Normal, 1 = Pneumonia
            if prediction > 0.5:
                confidence = prediction * 100
                st.error("🚨 **Prediction: PNEUMONIA**")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write("Please consult with a medical professional for a formal diagnosis.")
            else:
                confidence = (1 - prediction) * 100
                st.success("✅ **Prediction: NORMAL**")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write("No distinct signs of pneumonia detected.")