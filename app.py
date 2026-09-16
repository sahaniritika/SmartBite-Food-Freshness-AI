import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.cv_analyzer import calculate_decay_index
from models.mobilenet_model import load_model, predict_spoilage

# Page Configuration
st.set_page_config(page_title="SmartBite AI", page_icon="🍎", layout="centered")

st.title("🍎 SmartBite: Food Freshness & Spoilage AI")
st.write("Upload an image or capture via live continuous camera feed to analyze freshness in real-time.")

# Load AI Model
@st.cache_resource
def get_neural_model():
    return load_model()

model = get_neural_model()

# Select Input Mode
mode = st.radio("Select Input Mode:", ["Image Upload", "Live Webcam"])

# -------------------------------------------------------------
# MODE 1: Image File Upload
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# MODE 2: Real-Time Live Continuous Webcam Feed
# -------------------------------------------------------------
elif mode == "Live Webcam":
    st.subheader("📹 Real-Time Live Auto-Detector")
    st.write("Live feed me fruit/food dikhayein — screen par hi real-time Freshness status overlay ho jayega.")

    # Control buttons for webcam stream
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_cam = st.button("▶️ Start Live Camera")
    with col_btn2:
        stop_cam = st.button("⏹️ Stop Camera")

    # Session state initialization to prevent freezing
    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False

    if start_cam:
        st.session_state.run_camera = True
    if stop_cam:
        st.session_state.run_camera = False

    FRAME_WINDOW = st.image([])

    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)

        while st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam open nahi ho pa raha hai. Check karein ki camera physically connected hai ya nahi.")
                break

            # Convert BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Continuous AI Analysis
            decay_index = calculate_decay_index(pil_img)
            spoilage_prob = predict_spoilage(pil_img, model)

            # Define Status Overlay Text & Color
            if decay_index > 50.0 or spoilage_prob > 50.0:
                status_text = f"SPOILED | Decay: {decay_index:.1f}%"
                color = (255, 0, 0)  # Red for Spoiled
            else:
                status_text = f"FRESH | Decay: {decay_index:.1f}%"
                color = (0, 255, 0)  # Green for Fresh

            # Write Overlay Text directly onto video frame
            cv2.putText(frame_rgb, status_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Update live image stream on Streamlit UI
            FRAME_WINDOW.image(frame_rgb, use_container_width=True)

        # Properly release webcam when loop stops
        cap.release()