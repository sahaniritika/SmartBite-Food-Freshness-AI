import streamlit as st
import cv2
from PIL import Image
from utils.cv_analyzer import calculate_decay_index
from models.mobilenet_model import load_model, predict_spoilage

st.set_page_config(page_title="Food Freshness AI", page_icon="🍎")
st.title("🍎 SmartBite: Live Freshness Detector")

@st.cache_resource
def get_neural_model():
    return load_model()

model = get_neural_model()

# User Selectable Input Mode
mode = st.radio("Select Input Mode:", ["Image Upload", "Live Webcam"])

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width="stretch")
        
        decay_index = calculate_decay_index(image)
        spoilage_prob = predict_spoilage(image, model)
        
        st.write(f"**Surface Decay Index:** {decay_index}%")
        st.write(f"**Spoilage Confidence Score:** {spoilage_prob}%")
        
        if decay_index > 50.0 or spoilage_prob > 50.0:
            st.error("STATUS: SPOILED / ROTTEN ⚠️")
        else:
            st.success("STATUS: FRESH & SAFE ✅")

elif mode == "Live Webcam":
    run_webcam = st.checkbox("Start Webcam Feed")
    FRAME_WINDOW = st.image([])
    
    cap = cv2.VideoCapture(0) # 0 = Default Camera
    
    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not accessible.")
            break
            
        # OpenCV BGR format ko RGB me convert karein
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Diagnostics Calculate Karein
        decay_index = calculate_decay_index(pil_img)
        spoilage_prob = predict_spoilage(pil_img, model)
        
        # Real-time Status Overlay Text
        status_text = f"Decay: {decay_index}% | Spoilage: {spoilage_prob}%"
        color = (0, 255, 0) if (decay_index <= 50.0 and spoilage_prob <= 50.0) else (0, 0, 255)
        
        cv2.putText(frame_rgb, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Streamlit Canvas Update
        FRAME_WINDOW.image(frame_rgb)
        
    cap.release()