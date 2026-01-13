import logging
from typing import Dict, Any, TypedDict, Optional, List
from enum import Enum
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class Intent(Enum):
    GREETING = "greeting"
    INQUIRY = "inquiry"
    HIGH_INTENT = "high_intent"
    UNKNOWN = "unknown"

class AgentState(TypedDict):
    messages: List[dict]
    intent: Optional[Any]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    step: str
    user_input: Optional[str]
    agent_response: Optional[str]
    lead_state: Optional[Any]
    asked_name: Optional[bool]
    asked_email: Optional[bool]
    asked_platform: Optional[bool]
    is_streamlit: Optional[bool]

class AutoStreamAgent:
    def __init__(self, api_key, rag_engine):
        self.api_key = api_key
        self.rag_engine = rag_engine
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0
        )
        
        # Build components
        self._setup_chains()
        self.graph = self._build_graph()

    def _setup_chains(self):
        # Intent Classification Chain
        intent_prompt = PromptTemplate.from_template("""
Analyze the following user message and classify its intent.
Categories:
- greeting: Saying hello, "who are you", "what is this".
- inquiry: Asking about services, pricing, features, plans, or how it works.
- high_intent: Expressing a clear desire to buy, purchase, sign up, or get started.
- info_update: Providing a single piece of information like a name, email, or platform name (e.g., "Shivangi", "abc@gmail.com", "YouTube").

User Message: "{user_input}"

Classification (output only the category name):
""")
        self.intent_chain = intent_prompt | self.llm | StrOutputParser()

        # RAG Chain
        rag_prompt = PromptTemplate.from_template("""
You are a helpful and human-friendly assistant for AutoStream, a SaaS product by ServiceHive. 
AutoStream provides automated video editing tools for content creators and is part of the Inflx platform.

Use the following pieces of retrieved context to answer the user's question. 
Answer strictly based on the context provided. 
If the question is about who you are, what you do, or what services you provide, use the context to explain AutoStream's services and mission.
If the context mentions specific plans like "Basic Plan" or "Pro Plan", make sure to provide all relevant details found in the context for those plans.

Context:
{context}

Question: {question}

Answer:
""")
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

    def identify_intent(self, user_input: str) -> Intent:
        try:
            prediction = self.intent_chain.invoke({"user_input": user_input}).strip().lower()
            if "high_intent" in prediction: return Intent.HIGH_INTENT
            if "inquiry" in prediction: return Intent.INQUIRY
            if "greeting" in prediction: return Intent.GREETING
            return Intent.UNKNOWN
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return Intent.UNKNOWN

    def _build_graph(self):
        sg = StateGraph(AgentState)
        
        # Add Nodes
        sg.add_node('greetings', self.greeting_node)
        sg.add_node('intent', self.intent_node)
        sg.add_node('rag', self.rag_node)
        sg.add_node('lead_qual', self.lead_qual_node)
        sg.add_node('lead_capture', self.lead_capture_node)
        sg.add_node('fallback', self.fallback_node)
        
        # Conditional Edges
        sg.add_conditional_edges(START, self.start_router)
        sg.add_edge('greetings', END)
        sg.add_conditional_edges('intent', self.edge_fn)
        sg.add_edge('rag', END)
        sg.add_conditional_edges('lead_qual', self.lead_qual_transition)
        sg.add_edge('lead_capture', END)
        sg.add_edge('fallback', END)
        
        return sg.compile()

    # Nodes
    def greeting_node(self, state: AgentState):
        user_input = state.get('user_input', '')
        if user_input:
            context = "\n---\n".join(self.rag_engine.retrieve(user_input))
            state['agent_response'] = self.rag_chain.invoke({"context": context, "question": user_input})
        else:
            state['agent_response'] = "Hi! I'm your AutoStream assistant. How can I help you today?"
        state['step'] = 'await_user'
        return state

    def intent_node(self, state: AgentState):
        state['intent'] = self.identify_intent(state.get('user_input', ''))
        state['agent_response'] = None
        return state

    def rag_node(self, state: AgentState):
        user_input = state.get('user_input', '')
        context = "\n---\n".join(self.rag_engine.retrieve(user_input))
        state['agent_response'] = self.rag_chain.invoke({"context": context, "question": user_input})
        state['step'] = 'await_user'
        return state

    def lead_qual_node(self, state: AgentState):
        if not isinstance(state.get('lead_state'), dict):
            state['lead_state'] = {"name": None, "email": None, "platform": None}
        
        ls = state['lead_state']
        user_input = state.get('user_input', '')
        is_streamlit = state.get('is_streamlit', False)

        if not ls.get("name"):
            if is_streamlit:
                if not state.get('asked_name'):
                    state.update({'agent_response': "May I have your name?", 'asked_name': True, 'step': 'await_user'})
                    return state
                ls["name"], state['asked_name'] = user_input, False
            else:
                ls["name"] = input("Name: ")

        if ls.get("name") and not ls.get("email"):
            if is_streamlit:
                if not state.get('asked_email'):
                    state.update({'agent_response': "What is your email address?", 'asked_email': True, 'step': 'await_user'})
                    return state
                ls["email"], state['asked_email'] = user_input, False
            else:
                ls["email"] = input("Email: ")

        if ls.get("name") and ls.get("email") and not ls.get("platform"):
            if is_streamlit:
                if not state.get('asked_platform'):
                    state.update({'agent_response': "Which platform do you use?", 'asked_platform': True, 'step': 'await_user'})
                    return state
                ls["platform"], state['asked_platform'] = user_input, False
            else:
                ls["platform"] = input("Platform: ")

        state['step'] = 'lead_capture' if all(ls.values()) else 'await_user'
        return state

    def lead_capture_node(self, state: AgentState):
        ls = state['lead_state']
        state['agent_response'] = f"✅ Thank you {ls['name']}! We'll reach out to {ls['email']} soon."
        state['step'] = END
        return state

    def fallback_node(self, state: AgentState):
        state.update({'agent_response': "I'm here to help with product info or sign-up!", 'step': 'await_user'})
        return state

    # Routers
    def start_router(self, state: AgentState):
        if state.get('asked_name') or state.get('asked_email') or state.get('asked_platform'):
            return 'lead_qual'
        ls = state.get('lead_state')
        if isinstance(ls, dict) and ls.get("name") and not all(ls.values()):
            return 'lead_qual'
        return 'intent' if state.get('step') == 'intent' else 'greetings'

    def edge_fn(self, state: AgentState):
        intent = state.get('intent', Intent.UNKNOWN)
        if intent == Intent.GREETING: return 'greetings'
        if intent == Intent.INQUIRY: return 'rag'
        if intent == Intent.HIGH_INTENT: return 'lead_qual'
        return 'fallback'

    def lead_qual_transition(self, state: AgentState):
        return 'lead_capture' if state.get('step') == 'lead_capture' else END
