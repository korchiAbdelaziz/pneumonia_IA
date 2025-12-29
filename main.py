import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image
import os
import numpy as np

# --- 1. CORRECTIF TECHNIQUE ---
class CompatibleDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# --- 2. CONFIG PAGE ---
st.set_page_config(page_title="Pneumo AI", page_icon="📡", layout="centered")

# --- 3. STYLE CSS (Mobile + Cyber + Top Result) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Rajdhani:wght@500;700&display=swap');

    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.88), rgba(0, 0, 0, 0.88)), 
                    url("https://r.jina.ai/i/9e063878772348508e6473c180862086");
        background-attachment: fixed; background-size: cover;
        font-family: 'Rajdhani', sans-serif;
    }

    .cyber-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        text-align: center;
        background: linear-gradient(90deg, #00fbff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }

    /* NOTIFICATION RESULTAT EN HAUT */
    .notification {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 20px;
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        animation: glow 1.5s infinite alternate;
    }
    .notif-pos { background: rgba(255, 75, 75, 0.2); border: 2px solid #ff4b4b; color: #ff4b4b; }
    .notif-neg { background: rgba(0, 255, 136, 0.2); border: 2px solid #00ff88; color: #00ff88; }

    /* PETITE IMAGE CIRCULAIRE */
    .small-preview img {
        width: 120px !important;
        height: 120px !important;
        object-fit: cover;
        border-radius: 50%;
        border: 2px solid #00fbff;
        margin: 0 auto 20px auto;
        display: block;
    }

    .main-container {
        background: rgba(10, 20, 30, 0.8);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 251, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
    }

    @keyframes glow {
        from { box-shadow: 0 0 5px rgba(0, 251, 255, 0.2); }
        to { box-shadow: 0 0 15px rgba(0, 251, 255, 0.6); }
    }
</style>
""", unsafe_allow_html=True)

# --- 4. CHARGEMENT RESSOURCES ---
@st.cache_resource
def load_resources():
    model_path = "./model/pneumonia_classifier.h5"
    model = load_model(model_path, compile=False, custom_objects={'DepthwiseConv2D': CompatibleDepthwiseConv2D})
    with open("./model/labels.txt") as f:
        labels = [line.strip().split(" ")[1] for line in f.readlines()]
    return model, labels

model, class_names = load_resources()

# --- 5. UI PRINCIPALE ---
st.markdown('<h1 class="cyber-title">AI NEURAL SCAN</h1>', unsafe_allow_html=True)

container = st.container()

# On place l'uploader en bas dans le code, mais on affichera les résultats au-dessus grâce au container
file = st.file_uploader("📤 Scan New Radiology", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    
    # Analyse
    from util import classify
    label, score = classify(img, model, class_names)
    
    is_pneumonia = "PNEUMONIA" in label.upper()
    notif_class = "notif-pos" if is_pneumonia else "notif-neg"
    icon = "⚠️" if is_pneumonia else "✅"

    # AFFICHAGE DANS LE CONTAINER (Haut de page)
    with container:
        # 1. Notification de résultat
        st.markdown(f"""
            <div class="notification {notif_class}">
                <div style="font-size: 0.7rem; letter-spacing: 2px;">SCAN COMPLETE</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{icon} {label.upper()}</div>
                <div style="font-size: 1rem;">CONFIDENCE: {score*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. Petite image preview
        st.markdown('<div class="small-preview">', unsafe_allow_html=True)
        st.image(img)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr style="border-color: rgba(0,251,255,0.1)">', unsafe_allow_html=True)

else:
    with container:
        st.info("System ready. Please upload a radiology file below.")

# Footer
st.markdown("<p style='text-align:center; font-size:10px; opacity:0.3; margin-top:50px;'>BIO-DIGITAL UNIT v2.0</p>", unsafe_allow_html=True)