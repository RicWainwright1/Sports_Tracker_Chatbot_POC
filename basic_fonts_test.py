import streamlit as st
from pathlib import Path
import base64

# This MUST be the very first Streamlit command
st.set_page_config(page_title="Font Test", layout="centered")

# Function to get base64 encoded font
def get_base64_font(font_path):
    font_path = Path("assets") / font_path
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load font files
try:
    mont_regular = get_base64_font("Mont-Regular.woff")
    mont_heavy = get_base64_font("Mont-Heavy.woff")
    mont_light = get_base64_font("Mont-Light.woff")
    
    # More aggressive CSS with !important flags and targeting many elements
    css = f"""
    @font-face {{
        font-family: 'Mont';
        src: url(data:font/woff;base64,{mont_regular}) format('woff');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }}
    
    @font-face {{
        font-family: 'Mont';
        src: url(data:font/woff;base64,{mont_heavy}) format('woff');
        font-weight: 700;
        font-style: normal;
        font-display: swap;
    }}
    
    @font-face {{
        font-family: 'Mont';
        src: url(data:font/woff;base64,{mont_light}) format('woff');
        font-weight: 300;
        font-style: normal;
        font-display: swap;
    }}
    
    html, body, div, span, applet, object, iframe, h1, h2, h3, h4, h5, h6,
    p, blockquote, pre, a, abbr, acronym, address, big, cite, code, del,
    dfn, em, font, img, ins, kbd, q, s, samp, small, strike, strong, sub,
    sup, tt, var, dl, dt, dd, ol, ul, li, fieldset, form, label, legend,
    table, caption, tbody, tfoot, thead, tr, th, td, 
    .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, 
    .StreamlitApp, .stApp, .stMarkdown, .stText, .stTitle,
    button, input, textarea, .stButton, .stDownloadButton, .stFileUploader {{
        font-family: 'Mont', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    
    * {{
        font-family: 'Mont', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    
    /* Force body and main container */
    body {{
        font-family: 'Mont', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    
    /* Specifically target Streamlit elements */
    .element-container, .stText, .css-10trblm {{
        font-family: 'Mont', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    
    /* Override any inline styles */
    [style*="font-family"] {{
        font-family: 'Mont', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    """
    
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.success("Font CSS injected with more aggressive targeting")
except Exception as e:
    st.error(f"Error loading fonts: {e}")


