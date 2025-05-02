import streamlit as st
from dotenv import load_dotenv
import os
import openai
import numpy as np
import base64
from pathlib import Path
from typing import List, Dict, Any

# First Streamlit command must be set_page_config - keep this at the top
st.set_page_config(page_title="🎙️ TIF Sports Expert", layout="centered")

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
        help="Turn on/off the enthusiastic sports commentator personality"
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

    if "model" not in st.session_state:
        # Default model
        st.session_state.model = "gpt-4o"
    
    st.session_state.model = st.selectbox(
        "Model",
        options=["gpt-4o", "gpt-3.5-turbo", "gpt-4.1"],
        index=["gpt-4o", "gpt-3.5-turbo", "gpt-4.1"].index(st.session_state.model),
        help="Select the model to use for generating responses"
    )
    
    # Show current settings
    st.caption(f"Current settings: {'Coach Mode ON' if st.session_state.coach_mode else 'Coach Mode OFF'}, {st.session_state.answer_length} answers, {st.session_state.model} model")

# User context configuration - this will come from the DB
USER_CONTEXT = {
    "job_title": "Product Development Manager",
    "company": "Hasbro",
    "interests": ["youth athletics", "skill development", "Kids toys"],
    "audience": "teenagers",
    "expertise_level": "intermediate"  # beginner, intermediate, expert
}

# Function to get base64 encoded font
def get_base64_font(font_path):
    font_path = Path("assets") / font_path
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
# Add image loader function
def load_bottom_images():
    try:
        # Left image
        import os
        import base64
        
        assets_dir = os.path.join(os.getcwd(), "assets")
        # Left image
        left_image_path = os.path.join(assets_dir, "coachiq.png")
        if os.path.exists(left_image_path):
            with open(left_image_path, "rb") as img_file:
                left_img_data = base64.b64encode(img_file.read()).decode()
                left_html = f"""
                <div style="position: fixed; bottom: 0px; left: 0px; margin: 0; padding: 0; z-index: 9999;">
                    <img src="data:image/png;base64,{left_img_data}" 
                         style="border-radius: 8px; box-shadow: 0 0px 0px rgba(0,0,0,0.2); width: 200px;">
                </div>
                """
        else:
            left_html = """
            <div style="position: fixed; bottom: 0px; left: 0px; margin: 0; padding: 0; z-index: 9999;">
                <img src="https://via.placeholder.com/120x60/4CAF50/FFFFFF?text=Coach+IQ" 
                     style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); width: 120px;">
            </div>
            """
        
        # Right image
        right_image_path = os.path.join(assets_dir, "jumpman.png")
        if os.path.exists(right_image_path):
            with open(right_image_path, "rb") as img_file:
                right_img_data = base64.b64encode(img_file.read()).decode()
                right_html = f"""
                <div style="position: fixed; bottom: 350px; right: 40px; z-index: 9999;">
                    <img src="data:image/png;base64,{right_img_data}" 
                         style="border-radius: 8px; width: 175px;">
                </div>
                """
        else:
            right_html = """
            <div style="position: fixed; bottom: 200px; right: 0px; z-index: 9999;">
                <img src="https://via.placeholder.com/150x150?text=Coach+IQ+Jumping" 
                     style="border-radius: 8px; width: 100px;">
            </div>
            """
        
        # Return both HTML snippets
        return left_html, right_html
        
    except Exception as e:
        print(f"Error loading images: {e}")
        import traceback
        traceback.print_exc()
        return "", ""

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
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---- Made up classes ----

class MockVectorDB:
    def __init__(self):
        self.documents = [
            {
                "content": "Michael Jordan averaged 30.1 points per game during his NBA career.",
                "embedding": np.random.rand(1536),
                "tags": ["basketball", "nba", "stats", "michael jordan"]
            },
            {
                "content": "The 2022 World Cup was won by Argentina, beating France in penalties.",
                "embedding": np.random.rand(1536),
                "tags": ["football", "world cup", "argentina"]
            },
            {
                "content": "Serena Williams has won 23 Grand Slam singles titles in her career.",
                "embedding": np.random.rand(1536),
                "tags": ["tennis", "serena williams", "grand slam"]
            },
            {
                "content": "Rory McIlroy won the 2025 Masters Tounament.",
                "embedding": np.random.rand(1536),
                "tags": ["Golf", "Rory McIlroy", "Masters Tournament"]
            },
            {
                "content": "Rory McIlroy won the 2022 PGA Championship.",
                "embedding": np.random.rand(1536),
                "tags": ["Golf", "Rory McIlroy", "PGA Championship"]
            },
            
        ]

    def _get_embedding(self, text: str) -> np.ndarray:
        return np.random.rand(1536)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query: str, threshold: float = 0.85, max_results: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self._get_embedding(query)
        results = []
        
        # Use query terms to boost exact matches
        query_terms = set(query.lower().split())
        
        for doc in self.documents:
            # Base similarity from embedding comparison
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            
            # Boost similarity for exact term matches in content
            content_lower = doc["content"].lower()
            term_matches = sum(1 for term in query_terms if term in content_lower)
            match_boost = min(term_matches * 0.05, 0.15)  
            
            # Apply name-specific boosting
            doc_name_matches = [tag for tag in doc["tags"] if any(name in query.lower() for name in tag.split())]
            name_boost = 0.2 if doc_name_matches else 0
            
            # Final adjusted similarity
            adjusted_similarity = min(similarity + match_boost + name_boost, 0.99)
            
            if adjusted_similarity > threshold:
                results.append({
                    "content": doc["content"],
                    "similarity": adjusted_similarity,
                    "tags": doc["tags"]
                })
                
        # Sort by similarity score
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:max_results]

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
    def __init__(self, model="gpt-4o", user_context=USER_CONTEXT):
        self.model = model
        self.vector_db = MockVectorDB()
        self.db_connector = DatabaseConnector()
        self.user_context = user_context
    
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
        You are a decision-making engine for a sports commentator AI.
        Determine the appropriate action based on the user's query:
        "search_vector_db" if we likely have insights ready in our vector database
        "query_database" if we need to look up specific data
        "request_more_info" if the query is too vague or we need clarification
        Return ONLY one of these three options with no additional text.
        """
        completion = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        action = completion.choices[0].message.content.strip().lower()
        final_action = action if action in ["search_vector_db", "query_database", "request_more_info"] else "request_more_info"
        
        print(f"=== RAW ACTION RETURNED: '{action}' ===")
        print(f"=== FINAL ACTION DECISION: '{final_action}' ===\n")
        
        return final_action
        
        action = completion.choices[0].message.content.strip().lower()
        return action if action in ["search_vector_db", "query_database", "request_more_info"] else "request_more_info"
    
    def generate_response(self, query: str, conversation_history: List[Dict[str, str]]) -> tuple:
        # Call determine_action and log the result
        print("\n=== GENERATE RESPONSE STARTING ===")
        action = self.determine_action(query)
        print(f"=== ACTION RECEIVED IN GENERATE_RESPONSE: '{action}' ===")
        
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
            You are an enthusiastic sports commentator AI assistant that provides insights on kids, parents and families stats.
            
            USER CONTEXT:
            - Job Title: {self.user_context['job_title']}
            - Company: {self.user_context['company']}
            - Audience: {self.user_context['audience']}
            - Expertise Level: {self.user_context['expertise_level']}
            
            TAILOR YOUR RESPONSE to be relevant for someone in this role, using appropriate terminology,
            focusing on aspects that would interest them, and matching their level of expertise.
            
            IMPORTANT FORMATTING RULES:
            - Start your response with a bold title using markdown format: **TITLE**
            - Keep your ENTIRE response (including title) to 50 WORDS MAXIMUM
            - Make every word count - be concise but informative
            
            NO RESULTS WERE FOUND for this query. You need to start a conversation to gather more information - apologies and explain to the user that no instant insights are avaliable so we will find some instead. 
            Ask for the following pieces of information from: age, gender, location, date range and the answers your interested in.
            Frame your question in an enthusiastic sports commentator style.
            """
            if coach_mode:
                system_prompt += """
                Frame your question in an enthusiastic sports commentator style.
                Always maintain a sports commentator's enthusiastic tone and British English style.
                """
           
        else:
            system_prompt = f"""
            You are a{'n enthusiastic sports commentator' if coach_mode else 'n AI'} AI assistant.
            
            USER CONTEXT:
            - Job Title: {self.user_context['job_title']}
            - Compnay: {self.user_context['company']}
            - Audience: {self.user_context['audience']}
            - Expertise Level: {self.user_context['expertise_level']}
            
            TAILOR YOUR RESPONSE to be relevant for someone in this role, using appropriate terminology,
            focusing on aspects that would interest them, and matching their level of expertise.
            
            IMPORTANT FORMATTING RULES:
            - Start your response with a bold title using markdown format: **TITLE**
            - Keep your ENTIRE response (including title) to 50 WORDS MAXIMUM
            - Make every word count - be concise but informative
            
            Respond with the energy and vocabulary of a professional sports announcer.
            If you have context data, use it to craft your response but don't repeat it verbatim.
            If the query is too vague, briefly ask for specific details.
            
            """
            if coach_mode:
                system_prompt += """
                Respond with the energy and vocabulary of a professional sports announcer.
                Always maintain a sports commentator's enthusiastic tone and British English style.
                """
            else:
                system_prompt += """
                Provide clear, professional responses focusing on accuracy and relevance.
                If you have context data, use it to craft your response but don't repeat it verbatim.
                If the query is too vague, briefly ask for specific details.
                """

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nUSER QUERY: {query}\n\nRemember to start with a bold title and keep your total response under {max_words} words."
                    if context else query
        })

        print("=== GENERATING LLM RESPONSE ===")
        completion = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        reply = completion.choices[0].message.content
        print(f"=== LLM RESPONSE GENERATED ===")
        
        # Return both debug info and reply as a tuple
        return (debug_info, reply)

# ---- STREAMLIT CHAT UI ----

# Update the bot if model changes and bot exists
if "bot" in st.session_state and hasattr(st.session_state, "bot") and hasattr(st.session_state.bot, "model") and st.session_state.bot.model != st.session_state.model:
    st.session_state.bot = SportsCommentatorBot(model=st.session_state.model)
    st.rerun()  # Rerun the app to reflect changes immediately

st.title("Coach IQ - TIF Sports Expert")
st.caption("Sharper insights. Smarter decisions. For every game...")

if "history" not in st.session_state:
    st.session_state.history = []

if "bot" not in st.session_state:
    st.session_state.bot = SportsCommentatorBot(model=st.session_state.model)

# Display prior messages only once and maintain scroll state
for msg in st.session_state.history:
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
        <strong>Hey champ! I'm CoachIQ — your personal sports expert powered by TIF.</strong><br>
        Whether you're breaking down stats, comparing trends, or prepping for your next big presentation, I'm here to give you smart, fast, and accurate insights. Ask me anything — let's get you game-ready!<br><br>
        </div>
        """
        st.markdown(welcome_message, unsafe_allow_html=True)

user_input = st.chat_input("What’s your sports question?")
st.caption("Coach IQ can make mistakes. Please verify any critical information.")


if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="tif_shield_small.png"):
        with st.spinner("Getting the insights..."):
            # Get both debug info and reply from generate_response
            debug_info, reply = st.session_state.bot.generate_response(user_input, st.session_state.history)
            # Display debug info followed by reply
            st.markdown(debug_info + reply)
    
    # Only store the actual reply in conversation history, not the debug info
    st.session_state.history.append({"role": "assistant", "content": reply})
    
    # Process conversation history to extract information
    extract_and_print_gathered_info(st.session_state.history)

# Initialize persistent image state
if "images_loaded" not in st.session_state:
    st.session_state.images_loaded = False
    
# Load images only once
if not st.session_state.images_loaded:
    left_html, right_html = load_bottom_images()
    st.session_state.left_image_html = left_html
    st.session_state.right_image_html = right_html
    st.session_state.images_loaded = True

# Always display the images using the stored HTML
if hasattr(st.session_state, 'left_image_html'):
    st.markdown(st.session_state.left_image_html, unsafe_allow_html=True)
if hasattr(st.session_state, 'right_image_html'):
    st.markdown(st.session_state.right_image_html, unsafe_allow_html=True)
