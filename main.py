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

def main():
    if not GEMINI_API_KEY:
        print("Error: Gemini API key not found. Please set GOOGLE_API_KEY environment variable or create api_key.py.")
        return

    # Initialize Engine and Agent
    rag_engine = RAGEngine(
        md_path="source_of_truth.md",
        json_path="source_of_truth.json",
        db_path="chroma_db"
    )
    
    agent = AutoStreamAgent(api_key=GEMINI_API_KEY, rag_engine=rag_engine)
    
    # Initial State
    state = {
        'messages': [],
        'intent': None,
        'lead_captured': False,
        'step': 'greetings',
        'user_input': None,
        'agent_response': None,
        'lead_state': None,
        'is_streamlit': False
    }
    
    print("--- AutoStream Chatbot (Terminal Mode) ---")
    
    while state.get('step') != END:
        if state.get('step') == 'await_user':
            user_input = input("User: ")
            state['user_input'] = user_input
            state['messages'].append({'role': 'user', 'content': user_input})
            state['step'] = 'intent'
        
        state = agent.graph.invoke(state)
        
        if state.get('agent_response'):
            print(f"Agent: {state['agent_response']}")

if __name__ == "__main__":
    main()
