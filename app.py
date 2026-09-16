import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.cv_analyzer import calculate_decay_index
from models.mobilenet_model import load_model, predict_spoilage

# Page Configuration
st.set_page_config(page_title="SmartBite AI", page_icon="🍎", layout="centered")

st.title("🍎 SmartBite: Food Freshness & Spoilage AI")
st.write("Upload an image or use the live scanner to capture and analyze freshness.")

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
        
        # Diagnostics
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
# MODE 2: Live Scanner with Instant Freeze-on-Detect
# -------------------------------------------------------------
elif mode == "Live Webcam":
    st.subheader("📹 Live Camera Scanner")
    st.write("Fruit ko camera ke samne layein aur **Analyze Fruit** button par click karke result freeze karein.")

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_stream = st.button("▶️ Start Camera")
    with col2:
        capture_btn = st.button("📸 Analyze Fruit & Freeze")

    # Session State Variables
    if "scanning" not in st.session_state:
        st.session_state.scanning = False
    if "captured_frame" not in st.session_state:
        st.session_state.captured_frame = None

    if start_stream:
        st.session_state.scanning = True
        st.session_state.captured_frame = None

    FRAME_WINDOW = st.image([])

    # Live Stream Loop (Runs until user clicks 'Analyze')
    if st.session_state.scanning and st.session_state.captured_frame is None:
        cap = cv2.VideoCapture(0)

        while st.session_state.scanning:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Check if user clicked Analyze during live feed
            if capture_btn:
                st.session_state.captured_frame = frame_rgb
                st.session_state.scanning = False  # Auto-Stop Stream Loop
                break

            # Show live preview
            cv2.putText(frame_rgb, "Position fruit and click 'Analyze Fruit'", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            FRAME_WINDOW.image(frame_rgb, use_container_width=True)

        cap.release()

    # Display Frozen Frame & Complete AI Analysis
    if st.session_state.captured_frame is not None:
        frozen_img = st.session_state.captured_frame
        pil_img = Image.fromarray(frozen_img)

        # Run AI Model & Analysis on frozen frame
        decay_index = calculate_decay_index(pil_img)
        spoilage_prob = predict_spoilage(pil_img, model)

        # Draw overlay status on frozen frame
        if decay_index > 50.0 or spoilage_prob > 50.0:
            status_text = f"SPOILED | Decay: {decay_index:.1f}%"
            color = (255, 0, 0)
        else:
            status_text = f"FRESH | Decay: {decay_index:.1f}%"
            color = (0, 255, 0)

        cv2.putText(frozen_img, status_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Display Final Static Image & Results
        FRAME_WINDOW.image(frozen_img, use_container_width=True)

        st.markdown("---")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Surface Decay Index", f"{decay_index:.2f}%")
        with m2:
            st.metric("Spoilage Confidence", f"{spoilage_prob:.2f}%")

        if decay_index > 50.0 or spoilage_prob > 50.0:
            st.error("RESULT: SPOILED / ROTTEN ⚠️")
        else:
            st.success("RESULT: FRESH & SAFE ✅")