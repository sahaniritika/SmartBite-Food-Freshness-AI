import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from utils.cv_analyzer import calculate_decay_index
from models.mobilenet_model import load_model, predict_spoilage

# Page Configuration
st.set_page_config(
    page_title="SmartBite - Food Freshness & Spoilage AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Exact Pixel-Perfect Design Matching the Mockup
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1e293b;
}

#MainMenu, header, footer {visibility: hidden; height: 0;}
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1200px !important;
}

/* App Header / Navbar */
.nav-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem 0;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid #f1f5f9;
}
.brand-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.6rem;
    font-weight: 800;
    color: #1e3a2b;
    text-decoration: none;
}
.brand-logo span {
    font-size: 1.8rem;
}
.nav-right {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.slogan {
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.2;
    text-align: right;
}
.slogan-badge {
    background: #eaf5ee;
    color: #205c38;
    padding: 0.45rem 1rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Main Cards Grid */
.card-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.04);
    height: 100%;
}

.category-tag {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #3f7b53;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 0.6rem;
}

.sub-text {
    font-size: 0.98rem;
    color: #64748b;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

/* Privacy Alert Box */
.privacy-box {
    background: #eef7f0;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-top: 1.5rem;
}
.privacy-icon {
    color: #2e6040;
    font-size: 1.3rem;
    line-height: 1;
}
.privacy-text {
    font-size: 0.82rem;
    color: #274c35;
    line-height: 1.4;
}
.privacy-sub {
    font-size: 0.78rem;
    color: #557963;
}

/* Result Card Styles */
.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
}
.result-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sample-badge {
    font-size: 0.78rem;
    color: #94a3b8;
}

.result-image-container {
    width: 100%;
    height: 250px;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 1.2rem;
    background: #f8fafc;
}
.result-image-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* Fresh Status Banner */
.status-banner-fresh {
    background: #e8f5eb;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.4rem;
}
.status-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.status-icon-fresh {
    width: 38px;
    height: 38px;
    background: #2e7d32;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: bold;
}
.status-info h4 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: #1b5e20;
}
.status-info p {
    margin: 2px 0 0 0;
    font-size: 0.85rem;
    color: #386641;
}
.status-right {
    text-align: right;
    border-left: 1px solid #c8e6c9;
    padding-left: 1.2rem;
}
.status-right .label {
    font-size: 0.75rem;
    color: #52796f;
}
.status-right .score {
    font-size: 1.6rem;
    font-weight: 800;
    color: #1b5e20;
    line-height: 1.1;
}

/* Spoiled Status Banner */
.status-banner-spoiled {
    background: #fef2f2;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.4rem;
}
.status-icon-spoiled {
    width: 38px;
    height: 38px;
    background: #dc2626;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: bold;
}
.status-info-spoiled h4 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: #991b1b;
}
.status-info-spoiled p {
    margin: 2px 0 0 0;
    font-size: 0.85rem;
    color: #b91c1c;
}
.status-right-spoiled {
    text-align: right;
    border-left: 1px solid #fecaca;
    padding-left: 1.2rem;
}
.status-right-spoiled .label {
    font-size: 0.75rem;
    color: #991b1b;
}
.status-right-spoiled .score {
    font-size: 1.6rem;
    font-weight: 800;
    color: #dc2626;
    line-height: 1.1;
}

/* Indicators Table */
.indicator-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #334155;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.indicator-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 0;
    border-bottom: 1px solid #f8fafc;
    font-size: 0.88rem;
}
.ind-left {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #475569;
    font-weight: 500;
}
.ind-right {
    color: #64748b;
}

/* Section Containers */
.info-section {
    background: #ffffff;
    border: 1px solid #f1f5f9;
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-top: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.02);
}

.how-grid {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 3rem;
    align-items: center;
}
.section-heading-box h3 {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 0.4rem 0;
}
.section-heading-box p {
    font-size: 0.95rem;
    color: #64748b;
    margin: 0;
}

.steps-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}
.step-item {
    display: flex;
    gap: 12px;
    flex: 1;
}
.step-num {
    background: #eaf5ee;
    color: #2e6040;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.step-content h5 {
    margin: 0 0 4px 0;
    font-size: 0.98rem;
    font-weight: 700;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 6px;
}
.step-content p {
    margin: 0;
    font-size: 0.82rem;
    color: #64748b;
    line-height: 1.4;
}
.step-arrow {
    color: #cbd5e1;
    font-size: 1.2rem;
    font-weight: bold;
}

/* About Cards */
.about-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}
.about-card {
    background: #f8fafc;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
}
.about-card-icon {
    font-size: 1.8rem;
    margin-bottom: 0.8rem;
}
.about-card h4 {
    margin: 0 0 0.5rem 0;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
}
.about-card p {
    margin: 0;
    font-size: 0.88rem;
    color: #64748b;
    line-height: 1.5;
}

/* FAQ Accordion Styling */
.faq-box {
    margin-top: 1.5rem;
}
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 1.02rem !important;
    color: #1e293b !important;
    background-color: #f8fafc !important;
    border-radius: 10px !important;
}

/* Footer */
.app-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.2rem 0;
    border-top: 1px solid #f1f5f9;
    font-size: 0.85rem;
    color: #94a3b8;
}
.footer-left {
    color: #475569;
}
.footer-left strong {
    color: #0f172a;
}
.footer-right {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #64748b;
}

/* Custom Streamlit File Uploader and Buttons Styling */
div[data-testid="stFileUploader"] {
    background: #fafcfb;
    border: 2px dashed #bbf7d0;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #4ade80;
}
div[data-testid="stFileUploader"] section {
    padding: 0;
}
div[data-testid="stFileUploader"] button {
    background-color: #2e6040 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
}

/* Radio buttons styled as sleek toggle pills */
div[data-testid="stRadio"] > div {
    display: flex;
    gap: 12px;
    margin-bottom: 1.2rem;
}
div[data-testid="stRadio"] label {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    cursor: pointer;
    font-weight: 600;
    color: #475569;
    transition: all 0.2s;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #39674b !important;
    border-color: #39674b !important;
    color: #ffffff !important;
}
div[data-testid="stRadio"] label:has(input:checked) p {
    color: #ffffff !important;
}

/* Navigation Segmented Control */
.nav-tab-container div[data-testid="stRadio"] > div {
    justify-content: center;
    margin-bottom: 1.5rem;
}

/* Buttons */
.stButton > button {
    background: #2e6040;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    border: none;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #234c32;
    color: white;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Load AI Model (Cached)
@st.cache_resource
def get_neural_model():
    return load_model()

model = get_neural_model()

# Header Navigation Bar
st.markdown("""
<div class="nav-container">
    <div class="brand-logo">
        <span>🍎</span> SmartBite
    </div>
    <div class="nav-right">
        <div class="slogan">
            Good Food<br><strong>Better Tomorrow 🍃</strong>
        </div>
        <div class="slogan-badge">
            🍃 For a Healthier You
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Interactive Page Selector
current_tab = st.radio(
    "Navigation Menu",
    ["🏠 Home", "📖 About", "⚙️ How it works", "❓ FAQ"],
    horizontal=True,
    label_visibility="collapsed"
)

# ==============================================================================
# TAB 1: HOME (FOOD FRESHNESS DETECTOR)
# ==============================================================================
if current_tab == "🏠 Home":
    # Main 2-Column Layout
    col_left, col_right = st.columns([1.1, 1], gap="large")

    analyzed_image = None
    decay_index = 0.0
    spoilage_prob = 0.0
    is_fresh = True
    confidence_score = 92
    has_user_input = False

    with col_left:
        st.markdown("""
        <div class="category-tag">Food Freshness Detector</div>
        <div class="main-title">Check food freshness</div>
        <div class="sub-text">Upload an image or use your camera to analyze the freshness and detect possible spoilage.</div>
        """, unsafe_allow_html=True)

        input_mode = st.radio(
            "Select Input Mode:",
            ["🖼️ Upload image", "📷 Live camera"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if input_mode == "🖼️ Upload image":
            uploaded_file = st.file_uploader(
                "Upload an image of fruit or vegetable",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG (Max: 200MB)"
            )

            if uploaded_file is not None:
                analyzed_image = Image.open(uploaded_file)
                has_user_input = True
                
                # Compute real AI predictions
                decay_index = calculate_decay_index(analyzed_image)
                spoilage_prob = predict_spoilage(analyzed_image, model)
                
                is_fresh = (decay_index <= 40.0 and spoilage_prob <= 45.0)
                if is_fresh:
                    confidence_score = int(max(60, min(99, 100 - max(decay_index, spoilage_prob))))
                else:
                    confidence_score = int(max(decay_index, spoilage_prob))
                    
        else:  # Live Camera Mode
            st.write("Point your camera at the food item and capture:")
            camera_photo = st.camera_input("Take a photo to analyze freshness")
            
            if camera_photo is not None:
                analyzed_image = Image.open(camera_photo)
                has_user_input = True
                
                # Compute real AI predictions
                decay_index = calculate_decay_index(analyzed_image)
                spoilage_prob = predict_spoilage(analyzed_image, model)
                
                is_fresh = (decay_index <= 40.0 and spoilage_prob <= 45.0)
                if is_fresh:
                    confidence_score = int(max(60, min(99, 100 - max(decay_index, spoilage_prob))))
                else:
                    confidence_score = int(max(decay_index, spoilage_prob))

        # Privacy Notice
        st.markdown("""
        <div class="privacy-box">
            <div class="privacy-icon">🛡️</div>
            <div>
                <div class="privacy-text"><strong>Your images are processed locally and are not stored permanently.</strong></div>
                <div class="privacy-sub">We respect your privacy.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Helper function to get base64 string from PIL Image
    def get_image_base64(pil_img):
        buffered = io.BytesIO()
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode()

    with col_right:
        # Right column Card Container
        card_title = "Analysis Result" if has_user_input else "Sample Result"
        card_badge = "Real-time AI Analysis" if has_user_input else "This is an example result"
        
        st.markdown(f"""
        <div class="result-header">
            <div class="result-title">📊 {card_title}</div>
            <div class="sample-badge">{card_badge}</div>
        </div>
        """, unsafe_allow_html=True)

        if has_user_input and analyzed_image is not None:
            img_b64 = get_image_base64(analyzed_image)
            img_html = f'<img src="data:image/jpeg;base64,{img_b64}" alt="Analyzed Sample">'
        else:
            sample_apple_url = "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?auto=format&fit=crop&w=800&q=80"
            img_html = f'<img src="{sample_apple_url}" alt="Sample Apple">'

        st.markdown(f"""
        <div class="result-image-container">
            {img_html}
        </div>
        """, unsafe_allow_html=True)

        # Status Banner Display
        if has_user_input:
            if is_fresh:
                st.markdown(f"""
                <div class="status-banner-fresh">
                    <div class="status-left">
                        <div class="status-icon-fresh">✓</div>
                        <div class="status-info">
                            <h4>Fresh</h4>
                            <p>This food looks fresh and safe to consume.</p>
                        </div>
                    </div>
                    <div class="status-right">
                        <div class="label">Confidence</div>
                        <div class="score">{confidence_score}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                color_text = "Natural and vibrant"
                texture_text = "Firm and smooth"
                spoilage_text = f"No significant decay detected ({decay_index:.1f}% surface index)"
            else:
                st.markdown(f"""
                <div class="status-banner-spoiled">
                    <div class="status-left">
                        <div class="status-icon-spoiled">⚠️</div>
                        <div class="status-info-spoiled">
                            <h4>Spoiled / Decay</h4>
                            <p>Surface decay or spoilage detected. Not recommended to eat.</p>
                        </div>
                    </div>
                    <div class="status-right-spoiled">
                        <div class="label">Spoilage Prob</div>
                        <div class="score">{confidence_score}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                color_text = "Browning or discoloration detected"
                texture_text = "Soft or decaying surface texture"
                spoilage_text = f"Decay Index: {decay_index:.1f}% | Spoilage: {spoilage_prob:.1f}%"
        else:
            # Default Sample Result (Matching Mockup)
            st.markdown("""
            <div class="status-banner-fresh">
                <div class="status-left">
                    <div class="status-icon-fresh">✓</div>
                    <div class="status-info">
                        <h4>Fresh</h4>
                        <p>This food looks fresh and safe to consume.</p>
                    </div>
                </div>
                <div class="status-right">
                    <div class="label">Confidence</div>
                    <div class="score">92%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            color_text = "Natural and vibrant"
            texture_text = "Firm and smooth"
            spoilage_text = "No mold or dark spots detected"

        # Key Indicators List
        st.markdown(f"""
        <div>
            <div class="indicator-title">📋 Key Indicators</div>
            <div class="indicator-row">
                <div class="ind-left">🍃 Color</div>
                <div class="ind-right">{color_text}</div>
            </div>
            <div class="indicator-row">
                <div class="ind-left">💧 Texture</div>
                <div class="ind-right">{texture_text}</div>
            </div>
            <div class="indicator-row">
                <div class="ind-left">🛡️ Spoilage / Decay</div>
                <div class="ind-right">{spoilage_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # "How it works" 3-Step Summary at Bottom of Home
    st.markdown("""
    <div class="info-section">
        <div class="how-grid">
            <div class="section-heading-box">
                <h3>How it works</h3>
                <p>Get results in three simple steps.</p>
            </div>
            <div class="steps-container">
                <div class="step-item">
                    <div class="step-num">1</div>
                    <div class="step-content">
                        <h5>🖼️ Capture</h5>
                        <p>Upload an image or use your camera to capture the food item.</p>
                    </div>
                </div>
                <div class="step-arrow">→</div>
                <div class="step-item">
                    <div class="step-num">2</div>
                    <div class="step-content">
                        <h5>🔍 Analyze</h5>
                        <p>Our AI model analyzes visual features to check freshness and detect spoilage.</p>
                    </div>
                </div>
                <div class="step-arrow">→</div>
                <div class="step-item">
                    <div class="step-num">3</div>
                    <div class="step-content">
                        <h5>📄 Decide</h5>
                        <p>Get instant results with key indicators to help you make informed choices.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: ABOUT
# ==============================================================================
elif current_tab == "📖 About":
    st.markdown("""
    <div class="info-section">
        <div class="category-tag">ABOUT SMARTBITE</div>
        <div class="main-title">Empowering Healthier & Smarter Food Choices</div>
        <div class="sub-text">
            SmartBite is an intelligent computer vision application designed to evaluate food freshness, prevent foodborne illness, and combat global household food waste through instant, accessible AI analysis.
        </div>
        
        <div class="about-grid">
            <div class="about-card">
                <div class="about-card-icon">🧠</div>
                <h4>Deep Learning Vision</h4>
                <p>Powered by MobileNetV2 neural networks trained to recognize visual indicators of freshness, ripeness, and decomposition across diverse produce categories.</p>
            </div>
            <div class="about-card">
                <div class="about-card-icon">🔬</div>
                <h4>HSV Pixel Decay Analysis</h4>
                <p>Computer vision algorithms isolate surface decay pixels and color shifts to calculate a precise quantitative decay index across food surfaces.</p>
            </div>
            <div class="about-card">
                <div class="about-card-icon">🌍</div>
                <h4>Zero Food Waste Mission</h4>
                <p>Helping consumers distinguish safe, ripe food from spoiled items to reduce unnecessary food disposal while guaranteeing personal health and safety.</p>
            </div>
            <div class="about-card">
                <div class="about-card-icon">🔒</div>
                <h4>100% Privacy Focused</h4>
                <p>All image evaluation happens locally on device or in temporary memory sessions with zero permanent cloud storage of your private pictures.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 3: HOW IT WORKS
# ==============================================================================
elif current_tab == "⚙️ How it works":
    st.markdown("""
    <div class="info-section">
        <div class="category-tag">HOW IT WORKS</div>
        <div class="main-title">Behind the AI Analysis Pipeline</div>
        <div class="sub-text">SmartBite utilizes a dual-engine architecture combining deep convolutional neural networks with color space computer vision.</div>

        <div class="steps-container" style="margin: 2rem 0; flex-wrap: wrap;">
            <div class="step-item" style="background: #f8fafc; padding: 1.5rem; border-radius: 16px; border: 1px solid #e2e8f0;">
                <div class="step-num">1</div>
                <div class="step-content">
                    <h5>Image Ingestion & Preprocessing</h5>
                    <p>The image is converted to RGB format, normalized to standard ImageNet dimensions (224x224), and transformed into optimized neural tensor inputs.</p>
                </div>
            </div>
            <div class="step-arrow">→</div>
            <div class="step-item" style="background: #f8fafc; padding: 1.5rem; border-radius: 16px; border: 1px solid #e2e8f0;">
                <div class="step-num">2</div>
                <div class="step-content">
                    <h5>Surface Decay Segmentation</h5>
                    <p>Using HSV color space segmentation, we identify dark blemishes, rotting zones, and fungal patterns, computing a surface decay percentage.</p>
                </div>
            </div>
            <div class="step-arrow">→</div>
            <div class="step-item" style="background: #f8fafc; padding: 1.5rem; border-radius: 16px; border: 1px solid #e2e8f0;">
                <div class="step-num">3</div>
                <div class="step-content">
                    <h5>Confidence & Status Decision</h5>
                    <p>Both visual indicators are weighted to produce an actionable safety verdict (Fresh & Safe vs. Spoiled) alongside high-confidence key indicators.</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: FAQ
# ==============================================================================
elif current_tab == "❓ FAQ":
    st.markdown("""
    <div class="info-section">
        <div class="category-tag">FREQUENTLY ASKED QUESTIONS</div>
        <div class="main-title">Got questions? We've got answers.</div>
        <div class="sub-text">Everything you need to know about the SmartBite freshness detector.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🍎 What types of food can I scan with SmartBite?", expanded=True):
        st.write("""
        SmartBite is optimized for fruits and vegetables including **apples, bananas, tomatoes, oranges, strawberries, bell peppers, leafy greens, and bakery goods**. The neural model evaluates color vibrancy, skin texture, and surface blemishes.
        """)

    with st.expander("🔬 How does the AI calculate the Freshness & Spoilage score?"):
        st.write("""
        SmartBite uses a hybrid approach:
        1. **MobileNetV2 Neural Network**: Scans visual texture, structural shape, and feature distributions.
        2. **HSV Color Decomposition**: Pinpoints localized mold, oxidation, and fungal patches to calculate the **Surface Decay Index**.
        """)

    with st.expander("🔒 Are my photos uploaded or stored anywhere?"):
        st.write("""
        **No.** Your privacy is fully preserved. Uploaded pictures and live webcam frames are analyzed directly in temporary RAM and are discarded immediately after diagnosis.
        """)

    with st.expander("📱 Can I use SmartBite on my mobile phone?"):
        st.write("""
        **Yes!** Simply open the SmartBite URL in your mobile browser. You can snap a photo directly using your phone's camera or choose from your photo gallery.
        """)

    with st.expander("⚠️ What should I do if the app flags food as Spoiled?"):
        st.write("""
        If significant surface decay or discoloration is flagged, inspect the item carefully for foul odors or soft spots before consuming. When in doubt, avoid consumption to protect health and safety.
        """)

# Common Footer
st.markdown("""
<div class="app-footer">
    <div class="footer-left">
        <strong>SmartBite</strong> &nbsp;|&nbsp; Food Freshness & Spoilage AI
    </div>
    <div class="footer-right">
        🍃 Good Food Today • Healthier Tomorrow
    </div>
</div>
""", unsafe_allow_html=True)