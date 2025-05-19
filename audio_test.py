import streamlit as st
import os
import base64
import io
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up page
st.set_page_config(page_title="Audio Test", layout="centered")
st.title("ElevenLabs Audio Test")

# Get API key
elevenlabs_api_key = os.environ.get("ELEVEN_API_KEY")
if not elevenlabs_api_key:
    st.error("ELEVEN_API_KEY not found in environment variables")
    st.stop()

st.success(f"API Key found: {elevenlabs_api_key[:4]}***")

# Simple sidebar settings
with st.sidebar:
    st.title("Voice Settings")
    
    # Voice selection
    voice_options = {
        "Adam (Male British)": "pNInz6obpgDQGcFmaJgB",
        "Rachel (Female American)": "21m00Tcm4TlvDq8ikWAM",
        "Antoni (Male American)": "ErXwobaYiN019PkySvjV",
        "Bella (Female British)": "EXAVITQu4vr4xnSDxMaL",
        "Sam (Male American)": "yoZ06aMxZJJ28mfd3POQ",
        "Elli (Female American)": "MF3mGyEYCl7XYWbV9V6O"
    }
    
    selected_voice = st.selectbox(
        "Voice",
        options=list(voice_options.keys()),
        index=0
    )
    
    voice_id = voice_options[selected_voice]
    
    # Audio speed
    speed = st.slider(
        "Speech Speed",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

# Test input and button
text_input = st.text_area("Enter text to convert to speech", value="Hello! This is a test of the ElevenLabs text-to-speech API.")

def generate_audio(text, voice_id, speed):
    try:
        # Create client
        client = ElevenLabs(api_key=elevenlabs_api_key)
        
        # Generate audio
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
                "speaking_rate": float(speed)
            }
        )
        
        # Convert generator to bytes
        if hasattr(audio_generator, 'read'):
            # If it's a file-like object
            audio_bytes = audio_generator.read()
        elif hasattr(audio_generator, '__iter__'):
            # If it's a generator
            buffer = io.BytesIO()
            for chunk in audio_generator:
                buffer.write(chunk)
            buffer.seek(0)
            audio_bytes = buffer.read()
        else:
            # Assume it's already bytes
            audio_bytes = audio_generator
        
        st.write(f"Type of audio data: {type(audio_bytes)}")
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        # Create HTML5 audio element
        audio_html = f"""
        <audio controls autoplay="true" style="width:100%;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        
        return audio_html, audio_bytes
    
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

# Method 1: Using markdown with unsafe_allow_html
if st.button("Generate Audio (Method 1 - HTML via markdown)"):
    with st.spinner("Generating audio..."):
        audio_html, _ = generate_audio(text_input, voice_id, speed)
        if audio_html:
            st.success("Audio generated!")
            st.markdown(audio_html, unsafe_allow_html=True)

# Method 2: Different HTML approach
if st.button("Generate Audio (Method 2 - HTML via container)"):
    with st.spinner("Generating audio..."):
        audio_html, _ = generate_audio(text_input, voice_id, speed)
        if audio_html:
            st.success("Audio generated!")
            # Create a container for the audio
            audio_container = st.container()
            # Use the container to display the HTML
            audio_container.markdown(audio_html, unsafe_allow_html=True)

# Method 3: Using Streamlit's native audio component
if st.button("Generate Audio (Method 3 - Native Streamlit audio)"):
    with st.spinner("Generating audio..."):
        try:
            _, audio_bytes = generate_audio(text_input, voice_id, speed)
            if audio_bytes:
                st.success("Audio generated using Streamlit native audio component!")
                st.audio(audio_bytes, format="audio/mp3", start_time=0)
            
        except Exception as e:
            st.error(f"Error in Method 3: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Alternative method 4: Use the elevenlabs package with a stream
if st.button("Generate Audio (Method 4 - Alternative API)"):
    with st.spinner("Generating audio..."):
        try:
            # Using a different approach with the elevenlabs API
            import time
            import elevenlabs
            
            st.write("Using elevenlabs library directly...")
            
            # Set API key
            elevenlabs.set_api_key(elevenlabs_api_key)
            
            # Get the audio
            audio_data = elevenlabs.generate(
                text=text_input,
                voice=voice_id,
                model="eleven_multilingual_v2"
            )
            
            # Save to a temporary file
            audio_file_path = "temp_audio.mp3"
            with open(audio_file_path, "wb") as f:
                f.write(audio_data)
            
            # Use Streamlit's audio player
            st.success("Audio generated with alternative API!")
            st.audio(audio_file_path, format="audio/mp3", start_time=0)
            
        except Exception as e:
            st.error(f"Error in Method 4: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

st.divider()
st.caption("This is a simple test of the ElevenLabs audio functionality.") 