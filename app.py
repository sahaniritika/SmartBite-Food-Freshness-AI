import streamlit as st
from PIL import Image
from utils.cv_analyzer import calculate_decay_index
from models.mobilenet_model import load_model, predict_spoilage

# Page Configuration
st.set_page_config(page_title="SmartBite AI", page_icon="🍎", layout="centered")

st.title("🍎 SmartBite: Food Freshness & Spoilage AI")
st.write("Upload an image or capture via camera to analyze freshness in real-time.")

# Load AI Model
@st.cache_resource
def get_neural_model():
    return load_model()

model = get_neural_model()

# Select Mode
mode = st.radio("Select Input Mode:", ["Image Upload", "Live Webcam"])

# Mode 1: Image File Upload
if mode == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Food Sample", use_container_width=True)
        
        # Calculate Diagnostics
        decay_index = calculate_decay_index(image)
        spoilage_prob = predict_spoilage(image, model)
        
        # Display Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Surface Decay Index", f"{decay_index:.2f}%")
        with col2:
            st.metric("Spoilage Confidence", f"{spoilage_prob:.2f}%")
            
        # Decision Status
        if decay_index > 50.0 or spoilage_prob > 50.0:
            st.error("STATUS: SPOILED / ROTTEN ⚠️")
        else:
            st.success("STATUS: FRESH & SAFE ✅")

# Mode 2: Camera Snapshot Scanner (No Lag / Easy Stop)
elif mode == "Live Webcam":
    st.subheader("📷 Camera Scanner")
    camera_photo = st.camera_input("Take a picture to scan item")
    
    if camera_photo:
        pil_img = Image.open(camera_photo)
        
        # Calculate Diagnostics
        decay_index = calculate_decay_index(pil_img)
        spoilage_prob = predict_spoilage(pil_img, model)
        
        # Display Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Surface Decay Index", f"{decay_index:.2f}%")
        with col2:
            st.metric("Spoilage Confidence", f"{spoilage_prob:.2f}%")
            
        # Decision Status
        if decay_index > 50.0 or spoilage_prob > 50.0:
            st.error("STATUS: SPOILED / ROTTEN ⚠️")
        else:
            st.success("STATUS: FRESH & SAFE ✅")