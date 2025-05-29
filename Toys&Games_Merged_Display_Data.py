import streamlit as st
import dotenv
from dotenv import load_dotenv
import os
import openai
import numpy as np
import base64
from pathlib import Path
from typing import List, Dict, Any
import pinecone
import anthropic
from langsmith import Client
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
import pandas as pd
import plotly.express as px

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
os.environ["LANGCHAIN_PROJECT"] = "toys-and-games-chatbot"

# Load the survey data for chart creation
@st.cache_data
def load_survey_data():
    """Load the children's toy survey data"""
    try:
        return pd.read_csv("Children_Toy_Survey_Dummy_Data.csv")
    except FileNotFoundError:
        print("Warning: Children_Toy_Survey_Dummy_Data.csv not found")
        return pd.DataFrame()

# Custom color palette for charts
CUSTOM_COLORS = ["#2A317F", "#004FB9", "#39A8E0"]

def test_langsmith_connection():
    """Test LangSmith connection at startup"""
    try:
        print("=== TESTING LANGSMITH CONNECTION ===")
        
        test_run_id = str(uuid.uuid4())
        print(f"Testing feedback creation with dummy run_id: {test_run_id}")
        
        feedback_result = langsmith_client.create_feedback(
            run_id=test_run_id,
            key="test_connection",
            score=1,
            comment="Startup connection test"
        )
        
        print(f"✅ LangSmith connection successful: {feedback_result}")
        return True
        
    except Exception as e:
        print(f"❌ LangSmith connection failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

# Only test LangSmith connection once per session
if "langsmith_connection_tested" not in st.session_state:
    print("=== STARTUP: Testing LangSmith Connection ===")
    langsmith_connection_ok = test_langsmith_connection()
    st.session_state.langsmith_connection_ok = langsmith_connection_ok
    st.session_state.langsmith_connection_tested = True
else:
    langsmith_connection_ok = st.session_state.langsmith_connection_ok
print(f"LangSmith connection status: {'✅ OK' if langsmith_connection_ok else '❌ Failed'}")

# First Streamlit command must be set_page_config
st.set_page_config(page_title="🎙️ TIF Toys & Games Expert", layout="centered")

# Create sidebar with settings
with st.sidebar:
    st.title("Settings")
    
    if "coach_mode" not in st.session_state:
        st.session_state.coach_mode = True
    
    st.session_state.coach_mode = st.toggle(
        "Coach Mode", 
        value=st.session_state.coach_mode,
        help="Turn on/off the enthusiastic toys expert commentator personality"
    )
    
    if "answer_length" not in st.session_state:
        st.session_state.answer_length = "Short"
    
    st.session_state.answer_length = st.radio(
        "Answer Length",
        options=["Short", "Medium", "Long"],
        index=["Short", "Medium", "Long"].index(st.session_state.answer_length),
        horizontal=True
    )

    valid_models = ["gpt-4o", "gpt-3.5-turbo", "gpt-4.1", "o4-mini", "claude-3-7-sonnet", "grok-3-latest"]

    if "model" not in st.session_state:
        st.session_state.model = "o4-mini"

    if st.session_state.model not in valid_models:
        print(f"Invalid model value: {st.session_state.model}, resetting to default")
        st.session_state.model = valid_models[0]

    st.session_state.model = st.selectbox(
        "Model",
        options=valid_models,
        index=valid_models.index(st.session_state.model),
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

# User context configuration
USER_CONTEXT = {
    "job_title": "Product Development Manager",
    "company": "Hasbro",
    "interests": ["skill development", "Kids toys"],
    "audience": "teenagers",
    "expertise_level": "intermediate"
}

def determine_chart_type_and_layout(data, column, question_code=None, question_type=None):
    """Determine the best chart type based on data characteristics"""
    if data.empty:
        return None, 1
    
    answers = data[column].value_counts().reset_index()
    answers.columns = ['answer', 'count']

    count = len(answers)
    longest_length = max(answers['answer'].astype(str).apply(len))

    # Special case override
    if question_code == 'FEEL005':
        return "horizontalBar", 2

    if longest_length > 55:
        return "table", 1

    if count <= 5:
        if question_type in ["single_choice", "exclusive"]:
            return "horizontalBar", 2
        else:
            return "bar", 1
    elif count <= 10:
        return "horizontalBar", 2

    return "bar", 2

def create_chart_from_survey_data(query, chart_column=None, age_filter=None, gender_filter=None):
    """Create a chart based on the CSV survey data, filtered by query context"""
    df = load_survey_data()
    
    if df.empty or not chart_column or chart_column not in df.columns:
        return None
    
    # Apply filters based on query context or metadata
    df_filtered = df.copy()
    
    # Apply age filter if provided
    if age_filter:
        if isinstance(age_filter, str):
            if '-' in age_filter:
                try:
                    age_range = [int(x) for x in age_filter.split('-')]
                    df_filtered = df_filtered[(df_filtered['Age'] >= age_range[0]) & (df_filtered['Age'] <= age_range[1])]
                except:
                    pass
            else:
                try:
                    age_val = int(age_filter)
                    df_filtered = df_filtered[df_filtered['Age'] == age_val]
                except:
                    pass
    
    # Apply gender filter if provided
    if gender_filter and gender_filter.lower() in ['male', 'female']:
        df_filtered = df_filtered[df_filtered['Gender'].str.lower() == gender_filter.lower()]
    
    # If no data after filtering, use original
    if df_filtered.empty:
        df_filtered = df
    
    # Get chart type
    chart_type, chart_size = determine_chart_type_and_layout(df_filtered, chart_column)
    
    if chart_type == "table" or chart_type is None or df_filtered.empty:
        return None
    
    # Create value counts
    value_counts = df_filtered[chart_column].value_counts().reset_index()
    value_counts.columns = [chart_column, 'Count']
    
    # Limit to top 10 for readability
    if len(value_counts) > 10:
        value_counts = value_counts.head(10)
    
    # Create the chart
    title = f"Survey Data: {chart_column}"
    if age_filter:
        title += f" (Age: {age_filter})"
    if gender_filter:
        title += f" (Gender: {gender_filter})"
    
    if chart_type == "bar":
        fig = px.bar(
            value_counts, 
            x=chart_column, 
            y='Count', 
            title=title,
            color=chart_column,
            color_discrete_sequence=CUSTOM_COLORS
        )
        fig.update_xaxes(tickangle=45)
    elif chart_type == "horizontalBar":
        fig = px.bar(
            value_counts, 
            x='Count', 
            y=chart_column, 
            orientation='h',
            title=title,
            color=chart_column,
            color_discrete_sequence=CUSTOM_COLORS
        )
    else:
        return None
    
    # Customize chart appearance
    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        title_font_size=14
    )
    
    return fig

def extract_chart_context_from_query(query, metadata_list=None):
    """Extract age, gender, and chart column suggestions from query and metadata"""
    query_lower = query.lower()
    
    # Extract age from query
    age_filter = None
    import re
    age_patterns = [
        r'(\d+)[-\s]?(?:to|-)[-\s]?(\d+)[-\s]?(?:year|yr)',  # "5-8 year olds"
        r'(\d+)[-\s]?(?:year|yr)[-\s]?old',  # "5 year olds"
        r'age[s]?\s+(\d+)',  # "ages 5"
        r'(\d+)\s+(?:and|to)\s+(\d+)',  # "5 and 8"
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, query_lower)
        if match:
            if len(match.groups()) == 2:
                age_filter = f"{match.group(1)}-{match.group(2)}"
            else:
                age_filter = match.group(1)
            break
    
    # Extract gender from query
    gender_filter = None
    if any(word in query_lower for word in ['boy', 'boys', 'male']):
        gender_filter = 'Male'
    elif any(word in query_lower for word in ['girl', 'girls', 'female']):
        gender_filter = 'Female'
    
    # Suggest chart column based on query content
    chart_column = None
    column_keywords = {
        'Favorite Toy': ['favorite', 'preferred', 'popular', 'toy', 'like'],
        'Gender': ['gender', 'boy', 'girl', 'male', 'female'],
        'Age': ['age', 'year', 'old'],
        'Play Frequency': ['play', 'often', 'frequency', 'how much'],
        'Purchase Channels': ['buy', 'purchase', 'store', 'where', 'shop'],
        'Digital vs. Physical': ['digital', 'physical', 'screen', 'online'],
        'Character Connection': ['character', 'favorite character'],
        'Budget Allocation (£/yr)': ['budget', 'spend', 'money', 'cost', 'price'],
        'Play Duration': ['duration', 'long', 'time'],
        'Sustainability Importance': ['sustainability', 'environment', 'eco'],
    }
    
    for column, keywords in column_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            chart_column = column
            break
    
    # Default to most interesting columns if no specific match
    if not chart_column:
        chart_column = 'Favorite Toy'  # Most common default
    
    return age_filter, gender_filter, chart_column

class PineconeVectorDB:
    def __init__(self, api_key, environment, index_name, dimension):
        from pinecone import Pinecone, PodSpec
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.embeddings = OpenAIEmbeddings()

        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=environment
                )
            )

        self.index = self.pc.Index(index_name)

    def search_with_metadata(self, query, top_k=5):
        """Enhanced search that returns both content and metadata for charting"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            formatted_results = []
            
            for match in results.matches:
                metadata = getattr(match, 'metadata', {}) or {}
                
                # Get content
                content = metadata.get('Insight') or metadata.get('insight')
                if not content:
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
                    'similarity': match.score,
                    'metadata': metadata
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
        
        for category_name, category_data in self.database.items():
            for item_id, item_data in category_data.items():
                item_name = item_id.replace("_", " ").lower()
                
                if item_name in query_lower or any(word in item_name for word in query_lower.split() if len(word) > 3):
                    original_name = item_id.replace("_", " ")
                    results[original_name] = item_data
        
        return results

class SportsCommentatorBot:
    def __init__(self, model="o4-mini", effort="medium", user_context=USER_CONTEXT):
        self.model = model
        self.reasoning = {"effort": effort}
        
        try:
            print("Attempting to connect to Pinecone...")
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
            self.vector_db = type('MockVectorDB', (), {
                'search_with_metadata': lambda self, query, **kwargs: [{
                    'id': '1',
                    'content': 'Kids aged 6-12 show increased interest in educational toys that combine play with STEM learning.',
                    'similarity': 0.95,
                    'metadata': {'Age': '6-12', 'Gender': 'Mixed', 'Category': 'Educational'}
                }]
            })()
            
        self.db_connector = DatabaseConnector()
        self.user_context = user_context

    @traceable(run_type="chain", name="determine_action")
    def determine_action(self, query: str) -> str:
        print(f"\n=== DETERMINE ACTION CALLED FOR QUERY: '{query}' ===\n")
        
        query_lower = query.lower()
        
        for category_data in self.db_connector.database.values():
            for item_id in category_data.keys():
                item_name = item_id.replace("_", " ").lower()
                if item_name in query_lower:
                    print(f"=== FOUND DATABASE MATCH: '{item_name}' ===")
                    print(f"=== BYPASSING LLM, SETTING ACTION TO: 'query_database' ===\n")
                    return "query_database"
        
        system_prompt = """
        You are a decision-making engine for a toys and games expert.
        
        Determine the appropriate action based on the user's query:
        "search_vector_db" if we likely have insights ready in our vector database
        "query_database" if we need to look up specific data
        "request_more_info" if the query is too vague or we need clarification
        
        Return ONLY one of these three options with no additional text.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            if self.model == "claude-3-7-sonnet":
                import anthropic
                client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                response = client.messages.create(
                    model="claude-3-7-sonnet-20240620",
                    max_tokens=100,
                    messages=messages
                )
                action = response.content[0].text.strip().lower()
            else:
                completion = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1
                )
                action = completion.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"Error in determine_action: {e}")
            action = "search_vector_db"  # Default fallback
            
        final_action = action if action in ["search_vector_db", "query_database", "request_more_info"] else "search_vector_db"
        
        print(f"=== FINAL ACTION DECISION: '{final_action}' ===\n")
        return final_action

    @traceable(run_type="chain", name="generate_response")
    def generate_response(self, query: str, conversation_history: List[Dict[str, str]]) -> tuple:
        current_run_id = None
        try:
            run_tree = get_current_run_tree()
            if run_tree and hasattr(run_tree, 'id'):
                current_run_id = str(run_tree.id)
                print(f"✅ Captured run_id: {current_run_id}")
            else:
                print("⚠️ No run tree found or no ID attribute")
        except Exception as e:
            print(f"⚠️ Could not get run ID: {e}")
            current_run_id = str(uuid.uuid4())
            print(f"✅ Generated fallback run_id: {current_run_id}")
        
        print("\n=== GENERATE RESPONSE STARTING ===")
        action = self.determine_action(query)
        print(f"=== ACTION RECEIVED: '{action}' ===")
        
        insights = db_results = None
        chart_figure = None
        need_more_info = False
        info_gathering_mode = False

        # Get max words based on answer length setting
        max_words = {"Short": 30, "Medium": 100, "Long": 500}[st.session_state.answer_length]

        debug_info = f"**🔍 DEBUG INFO:**\n\n"
        debug_info += f"**User context:** {self.user_context['job_title']}, {self.user_context['company']}\n\n"
        debug_info += f"**Settings:** {'Coach Mode ON' if st.session_state.coach_mode else 'Coach Mode OFF'}, {st.session_state.answer_length} answers ({max_words} words max)\n\n"
        debug_info += f"**Action determined:** {action}\n\n"

        coach_mode = st.session_state.coach_mode

        print(f"=== PROCESSING ACTION: '{action}' ===")
        
        if action == "search_vector_db":
            print("=== EXECUTING VECTOR DB SEARCH ===")
            insights = self.vector_db.search_with_metadata(query)
            
            debug_info += f"**Vector DB Search Results:**\n"
            if insights:
                for i, insight in enumerate(insights):
                    debug_info += f"- {insight['content']} (similarity: {insight['similarity']:.2f})\n"
                
                # Extract context for chart creation from query and top insight
                age_filter, gender_filter, chart_column = extract_chart_context_from_query(
                    query, [insight.get('metadata', {}) for insight in insights[:3]]
                )
                
                # Create chart from survey data
                chart_figure = create_chart_from_survey_data(
                    query, chart_column, age_filter, gender_filter
                )
                
                if chart_figure:
                    debug_info += f"**Chart created:** {chart_column}"
                    if age_filter:
                        debug_info += f" (Age: {age_filter})"
                    if gender_filter:
                        debug_info += f" (Gender: {gender_filter})"
                    debug_info += "\n"
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
        
        # System prompt based on context
        if info_gathering_mode:
            system_prompt = f"""
            You are an enthusiastic Toys and Games expert that provides insights on kids, parents and families stats.
            
            USER CONTEXT:
            - Job Title: {self.user_context['job_title']}
            - Company: {self.user_context['company']}
            - Audience: {self.user_context['audience']}
            - Expertise Level: {self.user_context['expertise_level']}
            
            IMPORTANT FORMATTING RULES:
            - Start your response with a bold title using markdown format: **TITLE**
            - Keep your ENTIRE response (including title) to {max_words} WORDS MAXIMUM
            - Make every word count - be concise but informative
            
            NO RESULTS WERE FOUND for this query. Start a conversation to gather more information.
            Ask for: age, gender, location, date range and what specific answers they're interested in.
            Be enthusiastic and encourage exploration with phrases like:
            - "I can find more insights for your audience if you want"
            - "Tell me what you're most curious about"
            - "What specific trends are you looking to understand?"
            Frame your question in an enthusiastic, conversational style that encourages continued searching.
            """
            if coach_mode:
                system_prompt += """
                Frame your question in an enthusiastic commentator style.
                Always maintain an enthusiastic tone and British English style.
                Use phrases like "I'd love to help you discover..." or "Let's dive deeper into..." to make it conversational.
                """
        else:
            system_prompt = f"""
            You are a{'n enthusiastic Toys and Games expert' if coach_mode else 'n AI'} assistant.
            
            USER CONTEXT:
            - Job Title: {self.user_context['job_title']}
            - Company: {self.user_context['company']}
            - Audience: {self.user_context['audience']}
            - Expertise Level: {self.user_context['expertise_level']}
            
            IMPORTANT FORMATTING RULES:
            - Start your response with a bold title using markdown format: **TITLE**
            - Keep your ENTIRE response (including title) to {max_words} WORDS MAXIMUM
            - Make every word count - be concise but informative
            
            CONVERSATIONAL APPROACH:
            - Present the insight first, then encourage further exploration
            - End with engaging phrases like:
              * "I can find more insights for your audience if you want - tell me what you're looking for!"
              * "Want to explore this further? Ask me about specific age groups or toy categories!"
              * "There's so much more data to discover - what trends interest you most?"
            - Make it feel like an ongoing conversation, not a one-off answer
            
            Tailor your response to be relevant for someone in this role, using appropriate terminology,
            focusing on aspects that would interest them, and matching their level of expertise.
            Present the key insight clearly, then encourage continued exploration with specific suggestions.
            """
            if coach_mode:
                system_prompt += """
                Respond with the energy and vocabulary of a professional sports announcer.
                Always maintain an enthusiastic tone and British English style.
                Use phrases like "That's fascinating!" and "But wait, there's more!" 
                End with enthusiastic encouragement like "What else can I help you discover about your market?"
                Make every response feel like the beginning of an exciting journey of discovery.
                """

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nUSER QUERY: {query}\n\nRemember to start with a bold title and keep your total response under {max_words} words."
        })

        print("=== GENERATING LLM RESPONSE ===")
        
        try:
            if self.model == "claude-3-7-sonnet":
                import anthropic
                client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                
                claude_messages = []
                for msg in messages:
                    claude_messages.append({"role": msg["role"], "content": msg["content"]})
                
                response = client.messages.create(
                    model="claude-3-7-sonnet-20240620",
                    max_tokens=1000,
                    messages=claude_messages
                )
                reply = response.content[0].text
            else:
                completion = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1
                )
                reply = completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            reply = "I apologize, but I'm having trouble generating a response right now. Please try again."

        if not reply or len(reply.strip()) == 0:
            print("WARNING: Empty reply from LLM, using fallback response")
            reply = "I'm sorry, but I couldn't generate a proper response. Please try rephrasing your question."

        print(f"=== LLM RESPONSE GENERATED ===")
        
        return (debug_info, reply, current_run_id, chart_figure)

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize session state
if "audio_states" not in st.session_state:
    st.session_state.audio_states = {}

if "history" not in st.session_state:
    st.session_state.history = []

if "bot" not in st.session_state:
    st.session_state.bot = SportsCommentatorBot(model=st.session_state.model)

# Update bot if model changes
if "bot" in st.session_state and hasattr(st.session_state, "bot") and hasattr(st.session_state.bot, "model") and st.session_state.bot.model != st.session_state.model:
    st.session_state.bot = SportsCommentatorBot(model=st.session_state.model)
    st.rerun()

st.title("TIF Toys & Games Expert")
st.caption("Sharper insights. Smarter decisions. For every organisation...")

# Display chat history
for idx, msg in enumerate(st.session_state.history):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:  # assistant message
        with st.chat_message("assistant", avatar="tif_shield_small.png"): 
            st.markdown(msg["content"])
            # Display chart if available
            if "chart" in msg and msg["chart"] is not None:
                st.plotly_chart(msg["chart"], use_container_width=True, key=f"chart_history_{idx}")

# Welcome message
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

# Handle user input
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="tif_shield_small.png"):
        with st.spinner("Getting the insights..."):
            # Get enhanced response with chart
            result = st.session_state.bot.generate_response(user_input, st.session_state.history)
            
            if len(result) == 4:
                debug_info, reply, run_id, chart_figure = result
            else:
                debug_info, reply = result[:2]
                run_id = None
                chart_figure = None
            
            # Display response
            if st.session_state.get("show_debug", False):
                st.markdown(debug_info + reply)
            else:
                st.markdown(reply)
            
            # Display chart if available
            if chart_figure is not None:
                st.plotly_chart(chart_figure, use_container_width=True, key=f"chart_new_{int(time.time() * 1000)}")
            
            # Initialize feedback logging
            if "feedback_log" not in st.session_state:
                st.session_state.feedback_log = []

            # Feedback buttons
            button_timestamp = str(int(time.time() * 1000))
            msg_count = len(st.session_state.history)

            col1, col2, col3, col4 = st.columns([1, 1, 1, 10])

            with col1:
                if st.button("👍", key=f"up_{msg_count}_{button_timestamp}"):
                    try:
                        if run_id:
                            feedback_result = langsmith_client.create_feedback(
                                run_id=run_id,
                                key="user_feedback_positive",
                                score=1,
                                comment="User clicked thumbs up - positive feedback"
                            )
                            st.toast("✅ Thank you for your positive feedback!")
                        else:
                            st.toast("⚠️ No run ID available for feedback")
                    except Exception as e:
                        st.toast(f"❌ Feedback system error: {str(e)}")

            with col2:
                if st.button("👎", key=f"down_{msg_count}_{button_timestamp}"):
                    try:
                        if run_id:
                            feedback_result = langsmith_client.create_feedback(
                                run_id=run_id,
                                key="user_feedback_negative",
                                score=0,
                                comment="User clicked thumbs down - negative feedback"
                            )
                            st.toast("✅ Thank you for your feedback! We'll improve.")
                        else:
                            st.toast("⚠️ No run ID available for feedback")
                    except Exception as e:
                        st.toast(f"❌ Could not record feedback: {str(e)}")

            with col3:
                if st.button("📋", key=f"copy_{msg_count}_{button_timestamp}"):
                    st.toast("📋 Response copied!", icon="📋")

    # Store the response in conversation history with chart
    st.session_state.history.append({
        "role": "assistant", 
        "content": reply,
        "run_id": run_id,
        "chart": chart_figure
    })

def extract_and_print_gathered_info(conversation_history):
    """Extract gathered information from conversation history and print it to the terminal"""
    age = []
    gender = []
    location = []
    date_from = None
    date_to = None
    
    for message in conversation_history:
        if message["role"] == "user":
            content = message["content"].lower()
            
            # Extract age
            age_keywords = ["age", "year", "old", "years"]
            if any(keyword in content for keyword in age_keywords):
                import re
                numbers = re.findall(r'\b\d+\b', content)
                if numbers:
                    for num in numbers:
                        if int(num) < 100:
                            age.append(num)
            
            # Extract gender
            if "boy" in content or "male" in content or "m" in content:
                gender.append("boy")
            if "girl" in content or "female" in content or "f" in content:
                gender.append("girl")
                
            # Extract location
            location_keywords = ["location", "country", "city", "place", "region", "area"]
            if any(keyword in content for keyword in location_keywords):
                words = content.split()
                for i, word in enumerate(words):
                    if word in location_keywords and i+1 < len(words):
                        potential_location = words[i+1].strip(",.:;")
                        if len(potential_location) > 2:
                            location.append(potential_location)
            
            # Extract date range
            date_keywords = ["date", "from", "to", "between", "range", "period"]
            if any(keyword in content for keyword in date_keywords):
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
    
    print("\n----- GATHERED USER INFORMATION -----")
    print(f"age: {','.join(age) if age else 'Not provided'}")
    print(f"gender: {','.join(gender) if gender else 'Not provided'}")
    print(f"location: {','.join(location) if location else 'Not provided'}")
    print(f"date from: {date_from if date_from else 'Not provided'}")
    print(f"date to: {date_to if date_to else 'Not provided'}")
    print("-------------------------------------\n")

# Process conversation history
extract_and_print_gathered_info(st.session_state.history)