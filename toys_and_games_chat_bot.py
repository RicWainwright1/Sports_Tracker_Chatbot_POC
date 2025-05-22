import streamlit as st
import dotenv
from dotenv import load_dotenv
import os
import openai
import numpy as np
import base64
from pathlib import Path
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
import anthropic
from langsmith import Client
from langsmith.run_helpers import traceable

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone

import io
import base64
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import time
import uuid
import tempfile

# Initialize LangSmith client
langsmith_client = Client(
    api_url=st.secrets["LANGSMITH_API_URL"],
    api_key=st.secrets["LANGSMITH_API_KEY"]
)

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGSMITH_API_URL"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "toys-and-games-chatbot"  # Project name for organizing traces

# First Streamlit command must be set_page_config - keep this at the top
st.set_page_config(page_title="🎙️ TIF Toys & Games Expert", layout="centered")

# Create sidebar with settings
with st.sidebar:
    st.title("Settings")
    
    # Coach Mode toggle
    if "coach_mode" not in st.session_state:
        st.session_state.coach_mode = True
    
    st.session_state.coach_mode = st.toggle(
        "Coach Mode", 
        value=st.session_state.coach_mode,
        # adds some help text
        help="Turn on/off the enthusiastic toys expert commentator personality"
    )
    
    # Answer length options
    if "answer_length" not in st.session_state:
        st.session_state.answer_length = "Short"
    
    st.session_state.answer_length = st.radio(
        "Answer Length",
        options=["Short", "Medium", "Long"],
        index=["Short", "Medium", "Long"].index(st.session_state.answer_length),
        horizontal=True
    )

    # Define your valid models
    valid_models = ["gpt-4o", "gpt-3.5-turbo", "gpt-4.1", "o4-mini", "claude-3-7-sonnet", "grok-3-latest"]

    if "model" not in st.session_state:
        # Default model
        st.session_state.model = "o4-mini"

    # Add this safety check
    if st.session_state.model not in valid_models:
        print(f"Invalid model value: {st.session_state.model}, resetting to default")
        st.session_state.model = valid_models[0]  # Reset to first option if invalid

    st.session_state.model = st.selectbox(
        "Model",
        options=valid_models,
        index=valid_models.index(st.session_state.model),  # Now this will be safe
        help="Select the model to use for generating responses"
    )

    st.divider()
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    
    st.session_state.show_debug = st.toggle(
        "Show Debug Info", 
        value=st.session_state.show_debug,
        help="Toggle to show/hide debugging information"
    )
    

    st.divider()
    st.subheader("Voice Settings")
    
    # Auto-audio toggle
    if "auto_audio" not in st.session_state:
        st.session_state.auto_audio = False
    
    st.session_state.auto_audio = st.toggle(
        "Automatic Audio",
        value=st.session_state.auto_audio,
        help="Automatically generate audio for each response"
    )
    
    # Voice selection
    if "voice_id" not in st.session_state:
        st.session_state.voice_id = "JBFqnCBsd6RMkjVDRZzb"  # Default voice ID
    
    # Dictionary of voice options (name: voice_id)
    voice_options = {
        "Ric (TIF)": "3K8NPIuCNYW9SOTxx9xD",
        "Adam (TIF)": "1y9auxdbw3o2w1M1KZ8S",
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
        index=0,
        help="Select a voice for audio playback"
    )
    
    # Update the voice ID when selection changes
    st.session_state.voice_id = voice_options[selected_voice]
    
    # Audio speed control
    if "audio_speed" not in st.session_state:
        st.session_state.audio_speed = 1.0
    
    st.session_state.audio_speed = st.slider(
        "Speech Speed",
        min_value=0.5,
        max_value=2.0,
        value=st.session_state.audio_speed,
        step=0.1,
        help="Adjust the speed of speech playback"
    )

    # Show current settings
    st.caption(f"Current settings: {'Coach Mode ON' if st.session_state.coach_mode else 'Coach Mode OFF'}, {st.session_state.answer_length} answers, {st.session_state.model} model")

# User context configuration - this will come from the DB
USER_CONTEXT = {
    "job_title": "Product Development Manager",
    "company": "Hasbro",
    "interests": ["skill development", "Kids toys"],
    "audience": "teenagers",
    "expertise_level": "intermediate"  # beginner, intermediate, expert
}

# Function to get base64 encoded font
def get_base64_font(font_path):
    font_path = Path("assets") / font_path
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
def get_audio_player_simple(text, voice_id=None, speed=1.0):
    """Simplified audio player that focuses on basic functionality"""
    try:
        # Get API key with better visibility
        # elevenlabs_api_key = os.environ.get("ELEVEN_API_KEY")
        elevenlabs_api_key = st.secrets["ELEVEN_API_KEY"]
        if not elevenlabs_api_key:
            print("⚠️ No ElevenLabs API key found!")
            return "<p>Error: No API key found</p>"
        
        print(f"API key found (starts with {elevenlabs_api_key[:4]}***)")
        
        # Create client
        client = ElevenLabs(api_key=elevenlabs_api_key)
        
        # Generate audio (with simplified parameters)
        audio_generator = client.text_to_speech.convert(
            text=text[:200],  # Limit text length for testing
            voice_id=voice_id or "JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
        )
        
        # Handle different return types (generator, file-like, or bytes)
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
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        # Very simple audio element
        return f"""
        <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support audio playback.
        </audio>
        """
    except Exception as e:
        print(f"❌ Audio generation error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return f"<p>Error generating audio: {str(e)}</p>"

# Replace your get_audio_player function with this simplified version
def get_audio_player(text, voice_id=None, speed=1.0):
    """
    Generate audio from text using ElevenLabs and return an HTML audio player
    with more robust handling
    """
    print(f"\n=== GENERATING AUDIO FOR TEXT (length: {len(text)}) ===")
    print(f"=== VOICE ID: {voice_id} ===")
    print(f"=== SPEED: {speed} ===")

    try:
        # Get API key
        elevenlabs_api_key = os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            error_message = "ELEVEN_API_KEY not found in environment variables"
            print(f"ERROR: {error_message}")
            return f"<p style='color:red;'>Error: {error_message}</p>"

        # Validate input parameters
        if voice_id is None:
            voice_id = st.session_state.get("voice_id", "JBFqnCBsd6RMkjVDRZzb")
        if not speed and "audio_speed" in st.session_state:
            speed = st.session_state.audio_speed

        print(f"Using ElevenLabs with: voice_id={voice_id}, API key length={len(elevenlabs_api_key) if elevenlabs_api_key else 0}")

        # Create client
        client = ElevenLabs(api_key=elevenlabs_api_key)

        # First, test that we can access ElevenLabs API
        try:
            voice_list = client.voices.get_all()
            print(f"Successfully connected to ElevenLabs API. Found {len(voice_list.voices)} voices.")
        except Exception as e:
            print(f"⚠️ ElevenLabs API connection test failed: {e}")
            return f"<p style='color:orange;'>Warning: Could not connect to ElevenLabs API: {str(e)}</p>"

        # Limit text length if too long
        if len(text) > 500:
            print(f"Text too long ({len(text)} chars), truncating to 500 chars")
            text = text[:497] + "..."

        print("=== CALLING ELEVENLABS API ===")
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
        
        # Handle different return types (generator, file-like, or bytes)
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

        print(f"=== RECEIVED AUDIO DATA: {len(audio_bytes)} bytes ===")

        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        print(f"=== CONVERTED TO BASE64: {len(audio_base64)} chars ===")

        # Very simple audio player to minimize potential issues
        audio_player = f"""
        <audio controls style="width:100%; height:50px;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """

        print("=== AUDIO PLAYER HTML GENERATED ===")
        return audio_player

    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR GENERATING AUDIO: {error_msg}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return f"<p style='color:red;'>Error generating audio: {error_msg}</p>"
    



def inject_font_css():
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
    except Exception as e:
        st.error(f"Error loading fonts: {e}")

def extract_and_print_gathered_info(conversation_history):
        """
        Extracts gathered information from conversation history and prints it to the terminal
        in the requested format.
        """
        # Initialize data containers
        age = []
        gender = []
        location = []
        date_from = None
        date_to = None
        
        # Extract information from conversation history
        for message in conversation_history:
            if message["role"] == "user":
                content = message["content"].lower()
                
                # Extract age
                age_keywords = ["age", "year", "old", "years"]
                if any(keyword in content for keyword in age_keywords):
                    # Look for numbers in the message
                    import re
                    numbers = re.findall(r'\b\d+\b', content)
                    if numbers:
                        for num in numbers:
                            if int(num) < 100:  # Reasonable age filter
                                age.append(num)
                
                # Extract gender
                if "boy" in content or "male" in content or "m" in content:
                    gender.append("boy")
                if "girl" in content or "female" in content or "f" in content:
                    gender.append("girl")
                    
                # Extract location
                location_keywords = ["location", "country", "city", "place", "region", "area"]
                if any(keyword in content for keyword in location_keywords):
                    # Simple extraction - take words after location indicators
                    words = content.split()
                    for i, word in enumerate(words):
                        if word in location_keywords and i+1 < len(words):
                            potential_location = words[i+1].strip(",.:;")
                            if len(potential_location) > 2:  # Avoid short words
                                location.append(potential_location)
                
                # Extract date range
                date_keywords = ["date", "from", "to", "between", "range", "period"]
                if any(keyword in content for keyword in date_keywords):
                    # Look for date patterns YYYY-MM-DD
                    import re
                    dates = re.findall(r'\d{4}-\d{2}-\d{2}', content)
                    if len(dates) >= 2:
                        date_from = dates[0]
                        date_to = dates[1]
                    elif len(dates) == 1:
                        if "from" in content:
                            date_from = dates[0]
                        elif "to" in content:
                            date_to = dates[0]
        
        # Print the gathered information in the requested format
        print("\n----- GATHERED USER INFORMATION -----")
        print(f"age: {','.join(age) if age else 'Not provided'}")
        print(f"gender: {','.join(gender) if gender else 'Not provided'}")
        print(f"location: {','.join(location) if location else 'Not provided'}")
        print(f"date from: {date_from if date_from else 'Not provided'}")
        print(f"date to: {date_to if date_to else 'Not provided'}")
        print("-------------------------------------\n")



# Now inject the fonts (after set_page_config)
inject_font_css()

# Load .env variables
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["OPENAI_API_KEY"]


class PineconeVectorDB:
    def __init__(self, api_key, environment, index_name, dimension):
        # Create a Pinecone client using the new syntax
        from pinecone import Pinecone, PodSpec  # or ServerlessSpec if you're using serverless
        
        # Initialize the Pinecone client
        self.pc = Pinecone(
            api_key=api_key
        )
        
        self.index_name = index_name
        self.embeddings = OpenAIEmbeddings()

        # Check if index exists and create it if not
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            # Use the right spec based on your needs (PodSpec or ServerlessSpec)
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=environment
                )
            )

        # Get the index
        self.index = self.pc.Index(index_name)

    # Rest of your search method remains the same
    def search(self, query, top_k=3):
        """Search the vector database for similar vectors to the query"""
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Query Pinecone with the new API
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Debug the metadata structure
            if results.matches and len(results.matches) > 0:
                print("Sample metadata structure:")
                print(results.matches[0].metadata)
                print("Available metadata keys:")
                print(list(results.matches[0].metadata.keys() if hasattr(results.matches[0], 'metadata') else []))
            
            # Format results - use 'Insight' field instead of 'text'
            formatted_results = []
            for match in results.matches:
                metadata = getattr(match, 'metadata', {}) or {}
                # Try to get the Insight field first, fall back to other fields if needed
                content = metadata.get('Insight')
                
                # If no Insight field, try other common fields
                if not content:
                    content = metadata.get('insight')  # Try lowercase version
                
                # If still no content, try to create a summary from available fields
                if not content:
                    # Create a summary from multiple fields if Insight is not available
                    keyword = metadata.get('Keyword', metadata.get('keyword', ''))
                    region = metadata.get('Region', metadata.get('region', ''))
                    dimension = metadata.get('Dimension', metadata.get('dimension', ''))
                    
                    if keyword or region or dimension:
                        parts = []
                        if keyword:
                            parts.append(f"Keyword: {keyword}")
                        if region:
                            parts.append(f"Region: {region}")
                        if dimension:
                            parts.append(f"Dimension: {dimension}")
                        content = " | ".join(parts)
                    else:
                        content = "No content available"
                
                formatted_results.append({
                    'id': match.id,
                    'content': content,
                    'similarity': match.score
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching vector database: {e}")
            return []

class DatabaseConnector:
    def __init__(self):
        self.database = {
            "players": {
                "lebron_james": {"sport": "basketball", "team": "Los Angeles Lakers", "career_points": 38652},
                "lionel_messi": {"sport": "soccer", "team": "Inter Miami CF", "career_goals": 819},
                "serena_williams": {"sport": "tennis", "retired": True, "grand_slam_titles": 23}
            },
            "teams": {
                "manchester_united": {"sport": "soccer", "league": "Premier League", "championships": 20},
                "new_york_yankees": {"sport": "baseball", "league": "MLB", "world_series": 27},
                "golden_state_warriors": {"sport": "basketball", "league": "NBA", "championships": 7}
            },
            "events": {
                "super_bowl_lviii": {"winner": "Kansas City Chiefs", "year": 2024, "mvp": "Patrick Mahomes"},
                "olympics_2024": {"host": "Paris", "total_events": 329},
                "wimbledon_2023": {"men_winner": "Carlos Alcaraz", "women_winner": "Marketa Vondrousova"},
                "Ric Wainwright Competition": {"winner": "John Doe", "year": 2023, "prize": "Gold Medals"},
            }
        }

    def query(self, query_text: str) -> Dict[str, Any]:
        results = {}
        query_lower = query_text.lower()
        
        # More flexible matching for events, players and teams
        for category_name, category_data in self.database.items():
            for item_id, item_data in category_data.items():
                # Convert to lowercase for case-insensitive matching
                item_name = item_id.replace("_", " ").lower()
                
                # Check if query contains the item name OR if item name contains the query
                if item_name in query_lower or any(word in item_name for word in query_lower.split() if len(word) > 3):
                    # Use the original case format for the key in results
                    original_name = item_id.replace("_", " ")
                    results[original_name] = item_data
        
        # No fallbacks anymore - if no results, generate_response will handle the empty results
        return results

# ---- MAIN BOT ----

class SportsCommentatorBot:
    def __init__(self, model="o4-mini", effort="medium", user_context=USER_CONTEXT):
        self.model = model
        self.reasoning = {
            "effort": effort
        }
        
        # Use a try-except block to handle Pinecone API issues
        try:
            print("Attempting to connect to Pinecone...")
            # Get API key from environment variables
            pinecone_api_key = st.secrets["PINECONE_API_KEY"]
            pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]
            
            if pinecone_api_key:
                self.vector_db = PineconeVectorDB(
                    api_key=pinecone_api_key,
                    environment=pinecone_environment,
                    index_name="tandg",
                    dimension=1536
                )
                print("Successfully connected to Pinecone.")
            else:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
                
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            print("Using mock vector database instead.")
            # Create a simple mock vector database if Pinecone connection fails
            self.vector_db = type('MockVectorDB', (), {
                'search': lambda query, **kwargs: [
                    {
                        'id': '1',
                        'content': 'Kids aged 6-12 show increased interest in educational toys that combine play with STEM learning.',
                        'similarity': 0.95
                    },
                    {
                        'id': '2',
                        'content': 'Eco-friendly and sustainable toys are growing in popularity among environmentally conscious parents.',
                        'similarity': 0.92
                    },
                    {
                        'id': '3',
                        'content': 'Board games have seen a resurgence, with family game nights becoming more common post-pandemic.',
                        'similarity': 0.88
                    }
                ]
            })()
            
        self.db_connector = DatabaseConnector()
        self.user_context = user_context

    @traceable(run_type="chain", name="determine_action")
    def determine_action(self, query: str) -> str:
        print(f"\n=== DETERMINE ACTION CALLED FOR QUERY: '{query}' ===\n")
        
        # Quick check for known database entities before calling the LLM
        query_lower = query.lower()
        
        # Check if query contains any player, team or event names from database
        for category_data in self.db_connector.database.values():
            for item_id in category_data.keys():
                item_name = item_id.replace("_", " ").lower()
                if item_name in query_lower:
                    print(f"=== FOUND DATABASE MATCH: '{item_name}' ===")
                    print(f"=== BYPASSING LLM, SETTING ACTION TO: 'query_database' ===\n")
                    return "query_database"
        
        # If no direct match was found in database, use LLM to determine action
        system_prompt = """
        You are a decision-making engine for a toys and games expert.
        
        CONVERSATIONAL APPROACH:
        - Consider the user's query in the context of a natural conversation
        - Think about what would be most helpful in a real dialogue
        - Consider whether the user is asking for specific information or exploring a topic
        
        Determine the appropriate action based on the user's query:
        "search_vector_db" if we likely have insights ready in our vector database
        "query_database" if we need to look up specific data
        "request_more_info" if the query is too vague or we need clarification
        
        Return ONLY one of these three options with no additional text.
        """
        
        # Prepare messages for all models
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Handle Claude model differently
        if self.model == "claude-3-7-sonnet":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                
                # Call Claude API - REMOVED trace wrapper
                response = client.messages.create(
                    model="claude-3-7-sonnet-20240620",
                    max_tokens=100,
                    messages=messages
                )
                
                action = response.content[0].text.strip().lower()
            except Exception as e:
                print(f"Error calling Claude API: {e}")
                # Fallback to OpenAI if Claude fails - REMOVED trace wrapper
                completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=1
                )
                action = completion.choices[0].message.content.strip().lower()
        elif self.model == "grok-3-latest":
            try:
                import requests
                import json
                
                # Get X.AI API key from environment
                xai_api_key = st.secrets["XAI_API_KEY"]
                if not xai_api_key:
                    raise ValueError("XAI_API_KEY not found in environment variables")
                
                # Prepare the request
                url = "https://api.x.ai/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {xai_api_key}"
                }
                
                # Prepare the payload
                payload = {
                    "messages": messages,
                    "model": "grok-3-latest",
                    "stream": False,
                    "temperature": 1
                }
                
                # Make the API call - REMOVED trace wrapper
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                response.raise_for_status()
                result = response.json()
                
                action = result["choices"][0]["message"]["content"].strip().lower()
                
            except Exception as e:
                print(f"Error calling X.AI API: {e}")
                # Fallback to OpenAI if X.AI fails - REMOVED trace wrapper
                completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=1
                )
                action = completion.choices[0].message.content.strip().lower()
        else:
            # Use OpenAI for other models - REMOVED trace wrapper
            completion = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1
            )
            action = completion.choices[0].message.content.strip().lower()
            
        final_action = action if action in ["search_vector_db", "query_database", "request_more_info"] else "request_more_info"
        
        print(f"=== RAW ACTION RETURNED: '{action}' ===")
        print(f"=== FINAL ACTION DECISION: '{final_action}' ===\n")
        
        return final_action

    @traceable(run_type="chain", name="generate_response")
    def generate_response(self, query: str, conversation_history: List[Dict[str, str]]) -> tuple:
        # Call determine_action and log the result
        print("\n=== GENERATE RESPONSE STARTING ===")
        action = self.determine_action(query)
        print(f"=== ACTION RECEIVED IN GENERATE_RESPONSE: '{action}' ===")
        
        # Get the current run ID from LangSmith
        current_run_id = None
        try:
            current_run = langsmith_client.get_current_run()
            if current_run:
                current_run_id = current_run.id
        except Exception as e:
            print(f"Error getting current run ID: {e}")
        
        insights = db_results = None
        need_more_info = False
        info_gathering_mode = False

        # Get max words based on answer length setting
        max_words = 50  # Default
        if st.session_state.answer_length == "Short":
            max_words = 30
        elif st.session_state.answer_length == "Medium":
            max_words = 100
        elif st.session_state.answer_length == "Long":
            max_words = 500

        # Create debug info that will be displayed in Streamlit
        debug_info = f"**🔍 DEBUG INFO:**\n\n"
        debug_info += f"**User context:** {self.user_context['job_title']}, {self.user_context['company']}\n\n"
        debug_info += f"**Settings:** {'Coach Mode ON' if st.session_state.coach_mode else 'Coach Mode OFF'}, {st.session_state.answer_length} answers ({max_words} words max)\n\n"
        debug_info += f"**Action determined:** {action}\n\n"

        # Use coach mode setting to determine the tone and style
        coach_mode = st.session_state.coach_mode

        # Process based on the action
        print(f"=== PROCESSING ACTION: '{action}' ===")
        
        if action == "search_vector_db":
            print("=== EXECUTING VECTOR DB SEARCH ===")
            # REMOVED trace wrapper
            insights = self.vector_db.search(query)
            debug_info += f"**Vector DB Search Results:**\n"
            if insights:
                for i, insight in enumerate(insights):
                    debug_info += f"- {insight['content']} (similarity: {insight['similarity']:.2f})\n"
            else:
                debug_info += "No relevant insights found in vector DB.\n"
                info_gathering_mode = True
        elif action == "query_database":
            print("=== EXECUTING DATABASE QUERY ===")
            # REMOVED trace wrapper
            db_results = self.db_connector.query(query)
            debug_info += f"**Database Query Results:**\n"
            if db_results:
                for k, v in db_results.items():
                    debug_info += f"- {k}: {v}\n"
            else:
                debug_info += "No database results found.\n"
                info_gathering_mode = True
        else:
            print("=== HANDLING REQUEST FOR MORE INFO ===")
            need_more_info = True
            info_gathering_mode = True
            debug_info += "**Need more information:** Query is too vague\n"
        
        debug_info += "\n---\n\n"
        print("=== DEBUG INFO PREPARED ===")

        # Prepare context for the LLM
        context = ""
        if insights:
            context += "RELEVANT INSIGHTS:\n"
            for insight in insights:
                context += f"- {insight['content']}\n"
        if db_results:
            context += "DATABASE LOOKUP RESULTS:\n"
            for k, v in db_results.items():
                context += f"- {k}: {v}\n"
        if info_gathering_mode:
            context += "NO_RESULTS_FOUND: Begin information gathering conversation to learn more about user's query needs.\n"
            context += "INFORMATION_NEEDED: age, gender, location, date range\n"
        if need_more_info:
            context += "NEED MORE INFORMATION: The query lacks sufficient detail to provide a specific response.\n"
        
        # Choose system prompt based on whether we have results or need to gather info
        if info_gathering_mode:
            system_prompt = f"""
            You are an enthusiastic Toys and Games expert that provides insights on kids, parents and families stats.
            
            USER CONTEXT:
            - Job Title: {self.user_context['job_title']}
            - Company: {self.user_context['company']}
            - Audience: {self.user_context['audience']}
            - Expertise Level: {self.user_context['expertise_level']}
            
            CONVERSATIONAL APPROACH:
            - Engage in a natural dialogue that feels like a conversation with a knowledgeable friend
            - Show genuine curiosity about the user's interests and needs
            - Build on previous questions to create a coherent learning journey
            - Acknowledge what you've learned from previous interactions
            - Use follow-up questions that demonstrate you're listening and want to understand better
            
            TAILOR YOUR RESPONSE to be relevant for someone in this role, using appropriate terminology,
            focusing on aspects that would interest them, and matching their level of expertise.
            
            IMPORTANT FORMATTING RULES:
            - Start your response with a bold title using markdown format: **TITLE**
            - Keep your ENTIRE response (including title) to 50 WORDS MAXIMUM
            - Make every word count - be concise but informative
            
            NO RESULTS WERE FOUND for this query. Start a conversation to gather more information - explain to the user that no instant insights are available so we'll find some together. 
            Ask for the following pieces of information: age, gender, location, date range and what specific answers they're interested in.
            Frame your question in an enthusiastic, conversational style that shows you're genuinely interested in helping them.
            """
            if coach_mode:
                system_prompt += """
                Frame your question in an enthusiastic commentator style.
                Always maintain an enthusiastic tone and British English style.
                Use phrases like "I'd love to know more about..." or "Tell me about..." to make it conversational.
                """
        
        else:
            system_prompt = f"""
            You are a{'  enthusiastic Toys and Games expert' if coach_mode else 'n AI'} assistant.
            
            USER CONTEXT:
            - Job Title: {self.user_context['job_title']}
            - Company: {self.user_context['company']}
            - Audience: {self.user_context['audience']}
            - Expertise Level: {self.user_context['expertise_level']}
            
            CONVERSATIONAL APPROACH:
            - Engage in a natural dialogue that feels like a conversation with a knowledgeable friend
            - Show genuine curiosity about the user's interests and needs
            - Build on previous questions to create a coherent learning journey
            - Acknowledge what you've learned from previous interactions
            - Use follow-up questions that demonstrate you're listening and want to understand better
            - Connect new information to what you've already discussed to create a narrative
            
            TAILOR YOUR RESPONSE to be relevant for someone in this role, using appropriate terminology,
            focusing on aspects that would interest them, and matching their level of expertise.
            
            IMPORTANT FORMATTING RULES:
            - Start your response with a bold title using markdown format: **TITLE**
            - Keep your ENTIRE response (including title) to 100 WORDS MAXIMUM
            - Make every word count - be concise but informative
            
            Respond with the energy and vocabulary of a professional sports announcer.
            If you have context data, use it to craft your response but don't repeat it verbatim.
            If the query is too vague, briefly ask for specific details.
            
            """
            if coach_mode:
                system_prompt += """
                Respond with the energy and vocabulary of a professional sports announcer.
                Always maintain an enthusiastic tone and British English style.
                Use phrases like "That's fascinating!" or "I'm curious about..." to make it conversational.
                End with a follow-up question that builds on what you've just discussed.
                """
            else:
                system_prompt += """
                Provide clear, professional responses focusing on accuracy and relevance.
                If you have context data, use it to craft your response but don't repeat it verbatim.
                If the query is too vague, briefly ask for specific details.
                End with a thoughtful follow-up question that encourages further exploration.
                """

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nUSER QUERY: {query}\n\nRemember to start with a bold title and keep your total response under {max_words} words. Make your response conversational and engaging, as if we're having a natural dialogue. End with a thoughtful follow-up question that builds on what we've discussed."
                    if context else query
        })

        print("=== GENERATING LLM RESPONSE ===")
        
        # Handle Claude model differently
        if self.model == "claude-3-7-sonnet":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                
                # Convert messages to Claude format
                claude_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        claude_messages.append({"role": "system", "content": msg["content"]})
                    elif msg["role"] == "user":
                        claude_messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        claude_messages.append({"role": "assistant", "content": msg["content"]})
                
                # Call Claude API - REMOVED trace wrapper
                response = client.messages.create(
                    model="claude-3-7-sonnet-20240620",
                    max_tokens=1000,
                    messages=claude_messages
                )
                
                reply = response.content[0].text
            except Exception as e:
                print(f"Error calling Claude API: {e}")
                # Fallback to OpenAI if Claude fails - REMOVED trace wrapper
                completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=1
                )
                reply = completion.choices[0].message.content
        elif self.model == "grok-3-latest":
            try:
                import requests
                import json
                
                # Get X.AI API key from environment
                xai_api_key = st.secrets["XAI_API_KEY"]
                if not xai_api_key:
                    raise ValueError("XAI_API_KEY not found in environment variables")
                
                # Prepare the request
                url = "https://api.x.ai/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {xai_api_key}"
                }
                
                # Convert messages to Grok format
                grok_messages = []
                for msg in messages:
                    grok_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                # Prepare the payload
                payload = {
                    "messages": grok_messages,
                    "model": "grok-3-latest",
                    "stream": False,
                    "temperature": 1
                }
                
                # Make the API call - REMOVED trace wrapper
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                response.raise_for_status()
                result = response.json()
                
                reply = result["choices"][0]["message"]["content"]
                
            except Exception as e:
                print(f"Error calling X.AI API: {e}")
                # Fallback to OpenAI if X.AI fails - REMOVED trace wrapper
                completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=1
                )
                reply = completion.choices[0].message.content
        else:
            # Use OpenAI for other models - REMOVED trace wrapper
            completion = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1
            )
            reply = completion.choices[0].message.content

        print(f"=== LLM RESPONSE GENERATED ===")
        print(f"Raw LLM reply: {reply}")  # Print the raw reply to see if it's empty
        print(f"Reply length: {len(reply) if reply else 0}")  # Check if it's empty

        # Ensure the reply isn't None or empty
        if not reply or len(reply.strip()) == 0:
            print("WARNING: Empty reply from LLM, using fallback response")
            reply = "no reply from LLM"

        # At the very end of the function, check what's being returned
        print(f"Debug info length: {len(debug_info)}")
        print(f"Combined response first 100 chars: {(debug_info + reply)[:100]}")

        # Return both debug info and reply as a tuple
        return (debug_info, reply, current_run_id)

# ---- STREAMLIT CHAT UI ----

# Update the bot if model changes and bot exists
if "bot" in st.session_state and hasattr(st.session_state, "bot") and hasattr(st.session_state.bot, "model") and st.session_state.bot.model != st.session_state.model:
    st.session_state.bot = SportsCommentatorBot(model=st.session_state.model)
    st.rerun()  # Rerun the app to reflect changes immediately

st.title("TIF Toys & Games Expert")
st.caption("Sharper insights. Smarter decisions. For every organisation...")

if "audio_states" not in st.session_state:
    st.session_state.audio_states = {}

if "history" not in st.session_state:
    st.session_state.history = []

if "bot" not in st.session_state:
    st.session_state.bot = SportsCommentatorBot(model=st.session_state.model)

# Display prior messages only once and maintain scroll state
for idx, msg in enumerate(st.session_state.history):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:  # assistant message
        with st.chat_message("assistant", avatar="tif_shield_small.png"): 
            st.markdown(msg["content"])

# Welcome message only once
if not st.session_state.history:
    with st.chat_message("assistant", avatar="tif_shield_small.png"):
        welcome_message = """
        <div style="font-size: 0.9rem;">
        <strong>Hey! I'm your personal Toys & Games expert powered by TIF.</strong><br>
        Whether you're breaking down stats, comparing trends, or prepping for your next big presentation, I'm here to give you smart, fast, and accurate insights. Ask me anything — let's get you game-ready!<br><br>
        </div>
        """
        st.markdown(welcome_message, unsafe_allow_html=True)

user_input = st.chat_input("What's your Toys & Games question?")
st.caption("TIF experts can make mistakes. Please verify any critical information.")


if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="tif_shield_small.png"):
        with st.spinner("Getting the insights..."):
            # Get debug info, reply, and run_id from generate_response
            result = st.session_state.bot.generate_response(user_input, st.session_state.history)
            
            # Handle the tuple unpacking correctly
            if len(result) == 3:
                debug_info, reply, run_id = result
            else:
                debug_info, reply = result
                run_id = None
            
            # Display debug info followed by reply
            if st.session_state.get("show_debug", True):
                st.markdown(debug_info + reply)
            else:
                st.markdown(reply)
            
            # Add feedback buttons ONLY for the current response
            st.markdown("""
                <style>
                    /* Remove borders and padding from button containers */
                    div[data-testid="column"] {
                        border: none !important;
                        padding: 0 !important;
                        margin: 0 !important;
                    }
                    
                    /* Style the buttons to be compact and inline */
                    div[data-testid="stButton"] button {
                        padding: 0.25rem 0.5rem !important;
                        font-size: 1rem !important;
                        min-height: 2rem !important;
                        width: auto !important;
                        margin: 0.1rem !important;
                        border: none !important;
                        background: transparent !important;
                        box-shadow: none !important;
                    }
                    
                    /* Hover effects */
                    div[data-testid="stButton"] button:hover {
                        background: rgba(0,0,0,0.05) !important;
                        transform: scale(1.1);
                    }
                    
                    /* Remove column gaps */
                    .row-widget.stHorizontal {
                        gap: 0 !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Create columns with minimal spacing
            col1, col2, col3, col4 = st.columns([1, 1, 1, 10])
            
            with col1:
                current_thumbs_up = st.button("👍", key=f"current_thumbs_up")
                if current_thumbs_up:
                    try:
                        # Log positive feedback in LangSmith
                        if run_id:
                            langsmith_client.create_feedback(
                                run_id=run_id,
                                key="user_feedback",
                                score=1.0,
                                comment="User provided positive feedback"
                            )
                            print(f"✅ Positive feedback logged for run_id: {run_id}")
                        else:
                            print("⚠️ No run_id available for feedback")
                    except Exception as e:
                        print(f"❌ Error logging positive feedback to LangSmith: {e}")
                        import traceback
                        print(traceback.format_exc())
                    st.toast("Thanks for your positive feedback!")
            
            with col2:
                current_thumbs_down = st.button("👎", key=f"current_thumbs_down")
                if current_thumbs_down:
                    try:
                        # Log negative feedback in LangSmith
                        if run_id:
                            langsmith_client.create_feedback(
                                run_id=run_id,
                                key="user_feedback",
                                score=0.0,
                                comment="User provided negative feedback"
                            )
                            print(f"✅ Negative feedback logged for run_id: {run_id}")
                        else:
                            print("⚠️ No run_id available for feedback")
                    except Exception as e:
                        print(f"❌ Error logging negative feedback to LangSmith: {e}")
                        import traceback
                        print(traceback.format_exc())
                    st.toast("Thanks for your feedback! We'll use it to improve.")
            
            with col3:
                current_copy_btn = st.button("📋", key=f"current_copy")
                if current_copy_btn:
                    st.toast("Copied to clipboard!")
            
            # Auto-generate audio if enabled
            if st.session_state.get("auto_audio", False):
                try:
                    with st.spinner("Generating audio..."):
                        # Get clean text for speech (remove markdown)
                        speech_text = reply.replace('**', '').replace('#', '').strip()
                        
                        # Get voice settings from session state
                        voice_id = st.session_state.get("voice_id", "JBFqnCBsd6RMkjVDRZzb")
                        
                        # Get ElevenLabs API key - FIXED to use st.secrets
                        elevenlabs_api_key = st.secrets["ELEVEN_API_KEY"]
                        if not elevenlabs_api_key:
                            st.error("ElevenLabs API key not found. Please add it to your environment.")
                        else:
                            # Create the ElevenLabs client
                            client = ElevenLabs(api_key=elevenlabs_api_key)
                            
                            # Generate audio
                            audio_generator = client.text_to_speech.convert(
                                text=speech_text,
                                voice_id=voice_id,
                                model_id="eleven_multilingual_v2",
                                output_format="mp3_44100_128"
                            )
                            
                            # Handle different return types
                            if hasattr(audio_generator, 'read'):
                                # File-like object
                                audio_bytes = audio_generator.read()
                            elif hasattr(audio_generator, '__iter__'):
                                # Generator
                                buffer = io.BytesIO()
                                for chunk in audio_generator:
                                    buffer.write(chunk)
                                buffer.seek(0)
                                audio_bytes = buffer.read()
                            else:
                                # Bytes
                                audio_bytes = audio_generator
                            
                            # Save to a temporary file
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                            temp_filename = temp_file.name
                            temp_file.close()
                            
                            with open(temp_filename, 'wb') as f:
                                f.write(audio_bytes)
                            
                            # Display the audio player
                            st.audio(temp_filename, format="audio/mp3")
                            
                except Exception as e:
                    st.error(f"Error generating audio: {str(e)}")
                    if st.session_state.get("show_debug", False):
                        import traceback
                        st.code(traceback.format_exc())
    
    # Store the actual reply in conversation history with run_id
    st.session_state.history.append({
        "role": "assistant", 
        "content": reply,
        "run_id": run_id  # Store the run_id with the message
    })
    
    # Process conversation history to extract information
    extract_and_print_gathered_info(st.session_state.history)