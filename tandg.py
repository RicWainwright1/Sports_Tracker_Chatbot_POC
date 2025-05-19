import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Get values
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tandg")  # Default to 'tandg' if not set

# Initialize embedding
embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone with new API pattern
pc = Pinecone(api_key=PINECONE_API_KEY)

# Print available indexes for debugging
print(f"Available Pinecone indexes: {pc.list_indexes().names()}")
print(f"Using index: {PINECONE_INDEX_NAME}")

# Get the Pinecone index
index = pc.Index(PINECONE_INDEX_NAME)

# Now create the LangChain wrapper around the index
pinecone_index = PineconeVectorStore(
    index=index,
    embedding=embedding,
    text_key="text"
)

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

# --- Utility Functions ---
def query_vector_db(query: str, metadata_filters: dict = {}) -> list:
    return pinecone_index.similarity_search_with_score(query, k=5, filter=metadata_filters)

def has_valid_insights(results, threshold=0.85) -> bool:
    return any(score >= threshold for _, score in results)

def build_discussion_prompt(user_question: str, insights: list, metadata: list) -> str:
    insights_text = "\n".join(
        f"- {insight} (from Q1 2025 Toys and Games Tracker Data)" for insight in insights
    )
    return f"""
You are a neutral, insightful discussion agent focused on the toys and games industry.

You use quantifiable knowledge pulled from a Pinecone vector database that contains survey-driven insights. Each time you answer, support your ideas with the insights retrieved, referencing the keyword and region if possible.

Speak conversationally — you're here to help someone go on a journey to understand trends in the market. Avoid being definitive. Sit on the fence. Do not go against the insights from the vector DB.

If the user asks follow-up questions like \"tell me more\", \"compare X with Y\", or \"how has it performed over time\", use your insight history to help expand the discussion.

Here's what you know for this query:
{insights_text}

The user asked: {user_question}
"""

def build_category_prompt(user_question: str) -> str:
    return f"""
You are a toys and games industry expert discussion agent.

You don't have specific data for this question, but you should reason at the category level. Think broadly about common behaviours, typical trends, or market logic when discussing topics.

Speak in a non-committal way, always leaving room for uncertainty and possibility. Do not fabricate specific stats.

The user asked: {user_question}
"""

def get_openai_response(prompt: str) -> str:
    chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["input"], template="{input}"))
    return chain.run({"input": prompt})

# --- Streamlit UI ---
st.title("Toys & Games Industry Chatbot")

query = st.text_input("Ask me about Toys & Games")

if query:
    results = query_vector_db(query)

    if has_valid_insights(results):
        insights = [res[0].page_content for res in results if res[1] >= 0.85]
        metadata = [res[0].metadata for res in results if res[1] >= 0.85]
        agent_prompt = build_discussion_prompt(query, insights, metadata)
    else:
        agent_prompt = build_category_prompt(query)

    response = get_openai_response(agent_prompt)
    st.markdown(response)
