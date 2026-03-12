import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import time

# --- Page Config ---
st.set_page_config(
    page_title="Pneumonia Detection AI", 
    page_icon="🫁", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100) # Optional generic medical icon
    st.title("About the System")
    st.info(
        "This AI system utilizes a fine-tuned **ResNet50** Convolutional Neural Network "
        "to analyze chest X-ray images and detect signs of pneumonia."
    )
    st.write("### How to use:")
    st.write("1. Upload a clear chest X-ray image.")
    st.write("2. Wait for the model to analyze the textures.")
    st.write("3. Review the AI prediction and confidence score.")
    
    st.divider()
    st.warning(
        "⚠️ **Medical Disclaimer:** This tool is for educational and portfolio demonstration "
        "purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment."
    )

# --- Main App UI ---
st.title("🫁 Pneumonia Detection AI")
st.markdown("Upload a patient's chest X-ray to instantly process the image through our deep learning pipeline.")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = r"C:\Users\dilee\Desktop\Mdel\pneumonia_resnet50_model.keras"
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model. Please check the model path. Error: {e}")
    st.stop()

# --- File Uploader ---
st.markdown("### 📥 Image Upload")
uploaded_file = st.file_uploader("Drag and drop an X-ray image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.divider()
    
    # Create columns with adjusted widths (left column slightly smaller than right)
    col1, col2 = st.columns([0.4, 0.6], gap="large")
    
    with col1:
        st.markdown("#### 🖼️ Scanned X-Ray")
        # Add a nice border/shadow effect using st.image
        st.image(image, use_container_width=True)
        
    with col2:
        st.markdown("#### 🔬 AI Analysis")
        
        # UX Enhancement: Simulate processing time with a progress bar
        progress_text = "Applying ResNet50 preprocessing and extracting features..."
        progress_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01) # Adds a 1-second visual delay for premium UX
            progress_bar.progress(percent_complete + 1, text=progress_text)
        progress_bar.empty()
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = preprocess_input(img_array)       
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Result Display Box
        result_container = st.container(border=True)
        
        with result_container:
            if prediction > 0.5:
                confidence = prediction * 100
                st.error("🚨 **High Risk: PNEUMONIA DETECTED**")
                st.metric(label="AI Confidence Score", value=f"{confidence:.2f}%")
                st.markdown("The model detected patterns associated with lung consolidation or opacity. **Clinical review is strongly recommended.**")
            else:
                confidence = (1 - prediction) * 100
                st.success("✅ **Low Risk: NORMAL**")
                st.metric(label="AI Confidence Score", value=f"{confidence:.2f}%")
                st.markdown("The lungs appear clear. No distinct signs of pneumonia were detected by the model.")

    # --- Technical Details Expander ---
    st.write("")
    with st.expander("⚙️ View Technical Pipeline Details"):
        st.markdown("""
        * **Input Shape:** `(224, 224, 3)`
        * **Base Architecture:** ResNet50 (pre-trained on ImageNet)
        * **Custom Head:** GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
        * **Preprocessing:** `keras.applications.resnet50.preprocess_input` (zero-centers each color channel with respect to the ImageNet dataset)
        """)