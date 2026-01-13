import streamlit as st
import logging
import os
from src.rag_engine import RAGEngine
from src.chatbot_agent import AutoStreamAgent
from langgraph.graph import END

# Try to load API key from file or environment
try:
    from api_key import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="AutoStream AI", page_icon="🎥")
st.title("🎥 AutoStream AI Assistant")

if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set GOOGLE_API_KEY environment variable or create api_key.py.")
    st.info("Check README.md for instructions on how to set up your API key.")
    st.stop()

# Initialize Objects in Session State
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(
        md_path="source_of_truth.md",
        json_path="source_of_truth.json",
        db_path="chroma_db"
    )

if "agent" not in st.session_state:
    st.session_state.agent = AutoStreamAgent(
        api_key=GEMINI_API_KEY, 
        rag_engine=st.session_state.rag_engine
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        'messages': [],
        'intent': None,
        'lead_captured': False,
        'step': 'greetings',
        'user_input': None,
        'agent_response': None,
        'lead_state': None,
        'is_streamlit': True,
        'asked_name': False,
        'asked_email': False,
        'asked_platform': False
    }
    # Initial Greeting
    st.session_state.agent_state = st.session_state.agent.graph.invoke(st.session_state.agent_state)
    if st.session_state.agent_state.get('agent_response'):
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.agent_state['agent_response']})

# Display Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Update state
    st.session_state.agent_state['user_input'] = prompt
    st.session_state.agent_state['messages'].append({'role': 'user', 'content': prompt})
    
    # Decide flow
    ls = st.session_state.agent_state.get('lead_state')
    is_mid_qual = (
        st.session_state.agent_state.get('asked_name') or 
        st.session_state.agent_state.get('asked_email') or 
        st.session_state.agent_state.get('asked_platform')
    )
    
    st.session_state.agent_state['step'] = 'lead_qual' if is_mid_qual else 'intent'

    # Run Graph
    with st.spinner("Thinking..."):
        while st.session_state.agent_state.get('step') not in ['await_user', END]:
            st.session_state.agent_state = st.session_state.agent.graph.invoke(st.session_state.agent_state)
            if st.session_state.agent_state.get('agent_response'):
                content = st.session_state.agent_state['agent_response']
                st.session_state.messages.append({"role": "assistant", "content": content})
                with st.chat_message("assistant"):
                    st.markdown(content)
                st.session_state.agent_state['agent_response'] = None
