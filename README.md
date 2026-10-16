<div align="center">

# 🍎 SmartBite: AI Food Freshness & Spoilage Detector

**Real-time AI-powered produce quality assessment, surface decay indexing, and food spoilage diagnostics.**

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[Explore Features](#-key-features) • [Quickstart](#-quickstart--local-setup) • [Architecture](#-system-architecture) • [Deployment](#-deployment-guide) • [Contributing](#-contributing)

</div>

---

## 📌 Overview

**SmartBite** is an intelligent computer vision application created to combat global food waste, promote healthy nutrition, and prevent foodborne illnesses. By merging **deep convolutional neural networks (MobileNetV2)** with **HSV color-space decay segmentation**, SmartBite delivers instant, laboratory-grade produce quality analysis directly from your browser or mobile phone camera.

```
       +-------------------------------------------------------------+
       |                  SmartBite AI Pipeline                      |
       +-------------------------------------------------------------+
                                      |
         [ Image Input ] ------------>+------------> [ Live Camera ]
                                      |
                       +--------------+--------------+
                       |                             |
                       v                             v
           [ MobileNetV2 Deep Vision ]    [ HSV Color Segmentation ]
           - Morphological textures       - Necrotic tissue tracking
           - Spoilage probability         - Surface Decay Index %
                       |                             |
                       +--------------+--------------+
                                      |
                                      v
                      +-------------------------------+
                      |   Weighted Decision Matrix    |
                      |  • Verdict: FRESH or SPOILED  |
                      |  • Confidence Score Rating    |
                      |  • Actionable Safety Tips     |
                      +-------------------------------+
```

---

## 🌟 Key Features

- **🔬 Dual-Engine AI Diagnostics:**
  - **MobileNetV2 Neural Network**: Pretrained transfer-learning architecture that evaluates texture gradients, structural firmness, and biological degradation signatures.
  - **HSV Color Segmentation**: Advanced OpenCV color-space masking isolating oxidized blemishes, mold colonies, and decay patches to compute a precise **Surface Decay Index (%)**.
- **📷 Multi-Modal Food Scanning:**
  - **Image Upload**: Drag-and-drop support for high-resolution images (`.jpg`, `.jpeg`, `.png` up to 200MB).
  - **Live Camera Scanner**: Browser-integrated camera capture with real-time frame evaluation.
- **🎨 Modern Web UI:**
  - Custom design inspired by modern health-tech dashboards with **Plus Jakarta Sans** typography, clean cards, and responsive layout.
  - Comprehensive tabs for **Home**, **About**, **How it works**, and interactive **FAQ**.
- **🛡️ 100% Privacy-First Architecture:**
  - All media is analyzed in volatile memory. Images are never uploaded or permanently stored on external servers.

---

## 📁 Repository Structure

```tree
SmartBite-Food-Freshness-AI/
├── .streamlit/
│   └── config.toml          # Streamlit theme and production server configuration
├── models/
│   └── mobilenet_model.py   # MobileNetV2 model loading, transforms, and inference logic
├── utils/
│   └── cv_analyzer.py       # HSV decay segmentation and pixel analysis routines
├── app.py                   # Main Streamlit web application & modern user interface
├── Dockerfile               # Container deployment configuration
├── .dockerignore            # Docker build ignore patterns
├── .gitignore               # Git exclusions (virtualenvs, cache, OS files)
├── requirements.txt         # Production Python dependencies
└── README.md                # Project documentation
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Frontend / Web Server** | [Streamlit](https://streamlit.io/) | Interactive dashboard and web controls |
| **Deep Learning** | [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/stable/index.html) | MobileNetV2 neural inference pipeline |
| **Computer Vision** | [OpenCV (Headless)](https://opencv.org/) | HSV color conversion and decay masking |
| **Image Processing** | [Pillow (PIL)](https://python-pillow.org/) & [NumPy](https://numpy.org/) | Tensor preparation and matrix manipulation |
| **Containerization** | [Docker](https://www.docker.com/) | Production containerized deployment |

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- **Python 3.9+** (Tested on Python 3.9, 3.10, and 3.11)
- **Git**

### 2. Clone the Repository
```bash
git clone https://github.com/sahaniritika/SmartBite-Food-Freshness-AI.git
cd SmartBite-Food-Freshness-AI
```

### 3. Create and Activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt / PowerShell)
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Application
```bash
streamlit run app.py
```
Open your browser and navigate to:  
👉 **`http://localhost:8501`**

---

## 🐳 Docker Deployment

You can run SmartBite anywhere using Docker:

```bash
# Build the Docker image
docker build -t smartbite-ai .

# Run the container
docker run -p 8501:8501 smartbite-ai
```
Access the application at `http://localhost:8501`.

---

## ☁️ Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended & 100% Free)
1. Fork or push this repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository: `sahaniritika/SmartBite-Food-Freshness-AI`.
5. Select branch: `main`.
6. Set Main file path: `app.py`.
7. Click **"Deploy!"** — Your app will be live with a public URL in seconds!

### Option 2: Render
1. Create a new **Web Service** on [Render](https://render.com/).
2. Connect your GitHub repository.
3. Select **Environment**: `Docker` (or `Python 3`).
4. Set Start Command: `streamlit run app.py --server.port 10000 --server.address 0.0.0.0`.
5. Deploy.

### Option 3: Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **Streamlit** SDK.
3. Push the files or connect your GitHub repository.

---

## 🔍 How It Works

1. **Preprocessing & Tensor Normalization**:
   The input image is resized to $(224 \times 224)$ pixels, converted to standard 3-channel RGB, and normalized using ImageNet distribution metrics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
2. **HSV Decay Extraction**:
   The image is converted to the HSV (Hue-Saturation-Value) color space. An empirical threshold mask $[H: 10\text{--}30, S: 40\text{--}255, V: 20\text{--}200]$ segments necrotic browning, rot, and fungal degradation:
   $$\text{Decay Index (\%)} = \left(\frac{\text{Decayed Pixels}}{\text{Total Surface Pixels}}\right) \times 100$$
3. **Neural Spoilage Confidence**:
   MobileNetV2 processes morphological features through depthwise separable convolutions to generate output probability distributions:
   $$\text{Spoilage Probability} = \sigma(z)_i$$
4. **Safety Verdict**:
   - $\text{Decay} \le 40\%$ and $\text{Spoilage Prob} \le 45\% \implies$ **FRESH & SAFE ✅**
   - $\text{Decay} > 40\%$ or $\text{Spoilage Prob} > 45\% \implies$ **SPOILED / DECAY DETECTED ⚠️**

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve SmartBite:
1. Fork the Project (`gh repo fork sahaniritika/SmartBite-Food-Freshness-AI`).
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

<div align="center">
  <sub>Built with ❤️ for healthier eating and zero food waste.</sub>
</div>
