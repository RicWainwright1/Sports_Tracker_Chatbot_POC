import streamlit as st
import base64
import os
from pathlib import Path

def get_base64_encoded_font(font_file):
    """Function to encode font file to base64"""
    font_path = Path("assets") / font_file
    if not font_path.exists():
        st.error(f"Font file {font_file} not found in assets directory!")
        return ""
        
    with open(font_path, "rb") as f:
        font_data = f.read()
    return base64.b64encode(font_data).decode()

def inject_font_css():
    """Inject custom fonts directly into Streamlit"""
    # Check if assets directory exists
    if not Path("assets").exists():
        st.error("Assets directory not found!")
        return
        
    # Get base64 encoded fonts
    mont_regular = get_base64_encoded_font("Mont-Regular.woff")
    mont_heavy = get_base64_encoded_font("Mont-Heavy.woff")
    mont_light = get_base64_encoded_font("Mont-Light.woff")
    
    # Create CSS with embedded fonts
    css = f"""
    @font-face {{
        font-family: 'Mont';
        src: url(data:font/woff;base64,{mont_regular}) format('woff');
        font-weight: normal;
        font-style: normal;
    }}
    
    @font-face {{
        font-family: 'Mont';
        src: url(data:font/woff;base64,{mont_heavy}) format('woff');
        font-weight: bold;
        font-style: normal;
    }}
    
    @font-face {{
        font-family: 'Mont';
        src: url(data:font/woff;base64,{mont_light}) format('woff');
        font-weight: 300;
        font-style: normal;
    }}
    
    * {{
        font-family: 'Mont', sans-serif !important;
    }}
    
    body {{
        font-family: 'Mont', sans-serif;
    }}
    
    .stApp {{
        font-family: 'Mont', sans-serif !important;
    }}
    
    /* Style specific Streamlit elements */
    .stButton button, .stMarkdown, .stText {{
        font-family: 'Mont', sans-serif !important;
    }}
    
    h1, h2, h3, h4, h5, h6, p {{
        font-family: 'Mont', sans-serif !important;
    }}
    
    .stTitle, .stHeader {{
        font-family: 'Mont', sans-serif !important;
    }}
    """
    
    # Inject CSS with the font
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Add this line at the beginning of your app
inject_font_css()

# Then continue with your existing code...
# st.set_page_config(page_title="🎙️ TIF Sports Expert", layout="centered")
# st.title("🎙️ TIF Sports Expert")
# ...etc.