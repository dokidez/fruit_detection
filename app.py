import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Fruit Classifier AI", 
    page_icon="🍎", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS - GLASSMORPHISM & DARK FUTURISTIC UI
# ==========================================
def inject_custom_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        /* --- Global Styles & Static Dark Background --- */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #181538 100%);
            font-family: 'Inter', sans-serif;
            color: #e0e2e6; /* Force base app color */
        }

        /* --- Hide Streamlit Defaults --- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* --- AGGRESSIVE Text Visibility Overrides --- */
        .stApp p, .stApp span, .stApp label, .stApp li, .stApp td, .stApp th, 
        .stApp strong, .stApp em, .stApp blockquote {
            color: #e0e2e6 !important;
        }
        /* Fix bold text to stand out more */
        .stApp strong {
            color: #ffffff !important;
        }

        /* --- Typography --- */
        h1, h2, h3, h4, h5, h6 {
            background: linear-gradient(90deg, #e0c3fc 0%, #8ec5fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }

        /* --- Glassmorphism Cards (Columns) --- */
        [data-testid="stColumn"] {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 2rem 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            margin: 1rem 0.5rem;
            transition: all 0.3s ease;
        }
        [data-testid="stColumn"]:hover {
            border: 1px solid rgba(255, 255, 255, 0.25);
            box-shadow: 0 12px 40px 0 rgba(142, 197, 252, 0.1);
            transform: translateY(-2px);
        }

        /* --- Neumorphism Metric Cards --- */
        [data-testid="stMetric"] {
            background: rgba(40, 40, 60, 0.6);
            border-radius: 16px;
            padding: 1.2rem;
            box-shadow: 6px 6px 12px #0e0c20, -6px -6px 12px #201d3e;
            border: 1px solid rgba(255,255,255,0.05);
            margin-top: 1rem;
        }
        [data-testid="stMetricLabel"] {
            color: #a0a0b0 !important;
            font-size: 0.9rem !important;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }

        /* --- File Uploader --- */
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.8); /* Lighter frosted glass background so black text is readable */
            border: 2px dashed rgba(142, 197, 252, 0.6);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(8px);
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploader"]:hover {
            border: 2px dashed rgba(142, 197, 252, 0.9);
            background: rgba(255, 255, 255, 0.95);
        }
        /* FIX: Make placeholder text black */
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] span {
            color: #000000 !important; 
        }
        /* Ensure the Browse button text stays white */
        .st-emotion-cache-1eriv1k {
            background: linear-gradient(135deg, #302b63 0%, #24243e 100%) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            color: white !important;
            border-radius: 12px !important;
            transition: all 0.2s ease !important;
        }
        .st-emotion-cache-1eriv1k:hover {
            border: 1px solid rgba(255,255,255,0.5) !important;
            box-shadow: 0 0 15px rgba(142, 197, 252, 0.3) !important;
        }

        /* --- Progress Bars --- */
        .stProgress > div > div > div {
            background-image: linear-gradient(90deg, #e0c3fc 0%, #8ec5fc 100%);
            box-shadow: 0 0 10px rgba(142, 197, 252, 0.5);
            border-radius: 20px;
        }
        .stProgress > div > div > div > div {
            border-radius: 20px;
        }
        .stProgress > div > div {
            background-color: rgba(255, 255, 255, 0.1);
        }

        /* --- Success Alert --- */
        [data-testid="stSuccess"] {
            background: rgba(78, 236, 147, 0.1);
            border: 1px solid rgba(78, 236, 147, 0.3);
            border-radius: 12px;
            backdrop-filter: blur(5px);
        }
        [data-testid="stSuccess"] p, [data-testid="stSuccess"] span, [data-testid="stSuccess"] strong {
            color: #4eec93 !important;
        }

        /* --- Top Container/Block constraints --- */
        .block-container {
            padding: 3rem 5rem;
            max-width: 1200px;
        }
        
        /* --- Image styling & Centering --- */
        .stImage {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        /* --- Spinner --- */
        .stSpinner > div {
            border-top-color: #8ec5fc !important;
            border-bottom-color: #e0c3fc !important;
        }
        .stSpinner p {
            color: #e0e2e6 !important;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==========================================
# CONFIGURATION & CACHING
# ==========================================
MODEL_PATH = "mobilenetv2_fruit_model.keras"
CLASS_NAMES_PATH = "class_names.pkl"
IMG_SIZE = (128, 128)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        return None
    return joblib.load(CLASS_NAMES_PATH)

# ==========================================
# LOAD MODEL
# ==========================================
with st.spinner('⏳ Initializing AI Core...'):
    model = load_model()
    class_names = load_class_names()

if model is None:
    st.error(f"🚨 Model file not found! Make sure '{MODEL_PATH}' is in the same folder as app.py")
    st.stop()

if class_names is None:
    st.error(f"🚨 Class names file not found! Make sure '{CLASS_NAMES_PATH}' is in the same folder as app.py")
    st.stop()

# ==========================================
# MAIN APP UI
# ==========================================
st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>🍎 Fruit Classifier AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0a0b0;'>Upload an image of a fruit, and the MobileNetV2 neural network will classify it in real-time.</p><br>", unsafe_allow_html=True)

def predict_image(image, model, class_names):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    
    # Handle PNGs with 4 channels (RGBA) -> convert to 3 channels (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]
    
    return predicted_class, confidence, predictions[0]

uploaded_file = st.file_uploader("Drag & drop an image here or click to browse", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='text-align: center;'>Input Scan</h3>", unsafe_allow_html=True)
        st.image(image, width=280) 

    with st.spinner('⚡ Processing Neural Network Inference...'):
        predicted_class, confidence, all_predictions = predict_image(image, model, class_names)

    with col2:
        st.markdown("<h3 style='text-align: center;'>Analysis Result</h3>", unsafe_allow_html=True)
        st.success(f"🎯 Predicted: **{predicted_class}**")
        st.metric(label="Confidence Level", value=f"{confidence * 100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Probability Distribution</h3>", unsafe_allow_html=True)
    
    top_3_indices = np.argsort(all_predictions)[-3:][::-1]
    
    # Custom layout for top 3 predictions
    pred_cols = st.columns(3)
    for idx, i in enumerate(top_3_indices):
        class_name = class_names[i]
        prob = all_predictions[i] * 100
        
        with pred_cols[idx]:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1); text-align: center; backdrop-filter: blur(5px);">
                <h2 style="margin:0; color:#e0c3fc;">{class_name}</h2>
                <p style="font-size: 1.5rem; font-weight: 700; color: white; margin: 10px 0;">{prob:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob / 100)