import streamlit as st
import logging
import os
from src.rag_engine import RAGEngine
from src.chatbot_agent import AutoStreamAgent
from langgraph.graph import END

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="AutoStream AI | Automated Video Editing",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more polished look
st.markdown("""
    <style>
    /* Better chat message styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* Fix sidebar visibility and button styling */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #FF4B4B;
        color: white;
    }
    /* Ensure dashboard section stands out */
    .dashboard-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CREDENTIALS ---
try:
    from api_key import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- LOGGING ---
logging.basicConfig(level=logging.WARNING)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/video-editing.png", width=80)
    st.title("AutoStream AI")
    st.markdown("---")
    
    st.subheader("About AutoStream")
    st.info("""
        AutoStream provides automated video editing tools for creators. 
        Powered by **ServiceHive** and part of the **Inflx** platform.
    """)
    
    st.subheader("Quick Actions")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.agent_state = None
        st.rerun()
    
    st.markdown("---")
    st.subheader("Sample Questions")
    st.write("- What services do you provide?")
    st.write("- Tell me about your pricing plans.")
    st.write("- I want to purchase the Pro plan.")
    st.write("- How does the AI captioning work?")
    
    st.markdown("---")
    st.caption("Powered by LangGraph & Gemini 2.0 Flash")

# --- MAIN INTERFACE ---
st.title("🎥 AutoStream Assistant")
st.markdown("Welcome! I'm here to help you automate your video workflow and convert engagement into leads.")

if not GEMINI_API_KEY:
    st.error("⚠️ API Key Missing")
    st.info("Please set the `GOOGLE_API_KEY` environment variable or create an `api_key.py` file.")
    st.stop()

# --- INITIALIZATION ---
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

# Initialize agent state if needed
if st.session_state.get("agent_state") is None:
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
    # Get initial greeting
    st.session_state.agent_state = st.session_state.agent.graph.invoke(st.session_state.agent_state)
    if st.session_state.agent_state.get('agent_response'):
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.agent_state['agent_response']})

# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- DASHBOARD / GET STARTED (Shown when chat is new) ---
if len(st.session_state.messages) <= 1:
    st.markdown("---")
    st.subheader("How can I assist you?")
    st.write("""
        I'm trained on AutoStream's official documentation to help you with:
        - ℹ️ **Product Information**: Learn what AutoStream does and how it can help you.
        - 💰 **Pricing & Plans**: Get details on our Basic and Pro subscription plans.
        - 🚀 **Getting Started**: I can guide you through the sign-up and purchase process.
        - 🛠️ **Features**: Ask about AI captions, 4K exports, and automated editing.
    """)
    
    st.info("Try clicking one of the common questions below to see me in action!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 What services do you provide?"):
            st.session_state.user_query_trigger = "What services do you provide?"
        if st.button("💳 Tell me about pricing plans."):
            st.session_state.user_query_trigger = "Tell me about your pricing plans."
    with col2:
        if st.button("🔥 I want to purchase the Pro plan."):
            st.session_state.user_query_trigger = "I want to purchase the Pro plan."
        if st.button("❓ How does AI captioning work?"):
            st.session_state.user_query_trigger = "How does AI captioning work?"

# --- USER INPUT HANDLING ---
# Check if a button was clicked
prompt = st.chat_input("Ask about services, pricing, or say hello...")
if st.session_state.get("user_query_trigger"):
    prompt = st.session_state.user_query_trigger
    del st.session_state["user_query_trigger"]

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare agent state for next run
    st.session_state.agent_state['user_input'] = prompt
    st.session_state.agent_state['messages'].append({'role': 'user', 'content': prompt})
    
    # Force flow logic
    is_mid_qual = (
        st.session_state.agent_state.get('asked_name') or 
        st.session_state.agent_state.get('asked_email') or 
        st.session_state.agent_state.get('asked_platform')
    )
    st.session_state.agent_state['step'] = 'lead_qual' if is_mid_qual else 'intent'

    # Execute Graph
    with st.spinner("Processing..."):
        # The loop handles nodes that don't wait for input
        while st.session_state.agent_state.get('step') not in ['await_user', END]:
            st.session_state.agent_state = st.session_state.agent.graph.invoke(st.session_state.agent_state)
            
            # Display any response generated by the node
            if st.session_state.agent_state.get('agent_response'):
                content = st.session_state.agent_state['agent_response']
                st.session_state.messages.append({"role": "assistant", "content": content})
                with st.chat_message("assistant"):
                    st.markdown(content)
                # Clear response field so it doesn't duplicate in the next iteration
                st.session_state.agent_state['agent_response'] = None
