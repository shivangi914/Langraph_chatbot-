# AutoStream AI Chatbot

AutoStream is a sophisticated conversational AI agent designed for a fictional SaaS company. It provides automated video editing tools for content creators and is powered by `LangGraph`, `LangChain`, and `Google Gemini`.

## Overview
This chatbot uses a `RAG (Retrieval-Augmented Generation)` pipeline to answer user questions strictly based on a provided knowledge base. It features semantic intent detection to distinguish between general greetings, product inquiries, and high-intent purchase requests.

---

## Project Structure

- `src/rag_engine.py`: The core RAG component. It handles loading `Markdown` and `JSON` knowledge files, chunking the content, managing the `ChromaDB` vector store, and retrieving relevant context for queries.
- `src/chatbot_agent.py`: The brain of the bot. It defines the `LangGraph` state machine, LLM-powered semantic intent classification, and the logic for the "Lead Qualification" flow (collecting name, email, etc.).
- `streamlit_app.py`: The modern web interface built with `Streamlit`. It provides a clean chat UI and manages the session state for the agent.
- `main.py`: A terminal-based entry point for those who prefer interacting with the bot via the command line.
- `api_key.py`: A configuration file used to store your `Google Gemini API Key`.
- `source_of_truth.md` / `.json`: The knowledge base files that the agent uses to answer questions about AutoStream's services, pricing, and policies.
- `requirements.txt`: Lists all the Python libraries required to run the project.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher.
- A Google Gemini API Key (obtain one from Google AI Studio).

### 2. Clone and Install
Open your terminal in the project root and run:
```bash
# Install required libraries
pip install -r requirements.txt
```

### 3. Configure API Key
To keep your credentials safe, the `api_key.py` file is ignored by Git. To set up your key:
1. Copy the provided template: `cp api_key_template.py api_key.py` (or manually copy and rename it).
2. Open `api_key.py` and replace `"YOUR_API_KEY_HERE"` with your actual Google Gemini API key.

Alternatively, you can set an environment variable named `GOOGLE_API_KEY`.

---

## How to Run

### Option 1: Web UI (Recommended)
Launch the interactive Streamlit interface:
```bash
streamlit run streamlit_app.py
```

### Option 2: Terminal Mode
Run the chatbot directly in your console:
```bash
python main.py
```

---

## Features
- `Semantic Intent Detection`: Uses LLM reasoning to understand what you want, rather than simple word matching.
- `Strict RAG`: The agent only answers based on the `source_of_truth` files, ensuring accurate product information.
- `Stateful Lead Capture`: A guided flow that collects user information (Name, Email, Platform) before triggering a mock lead-capture API.
- `Optimized Storage`: Uses a persistent ChromaDB instance to avoid redundant data processing.

---

Built using LangGraph & Gemini.
