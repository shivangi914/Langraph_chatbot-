import os
import json
import logging
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, md_path, json_path, db_path, model_name='all-MiniLM-L6-v2'):
        self.md_path = md_path
        self.json_path = json_path
        self.db_path = db_path
        self.model_name = model_name
        self._model = None
        self._collection = None

    def get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def get_collection(self):
        if self._collection is None:
            chroma_client = PersistentClient(path=self.db_path)
            # Check if collection exists and refresh if needed
            try:
                # For this setup, we refresh every time the engine is initialized 
                # to ensure data consistency, but we could make it smarter.
                chroma_client.delete_collection(name="knowledge_base")
            except:
                pass
            self._collection = chroma_client.create_collection(name="knowledge_base")
            self._load_initial_data()
        return self._collection

    def _load_initial_data(self):
        if not os.path.exists(self.md_path) or not os.path.exists(self.json_path):
            logger.warning("Knowledge base files not found.")
            return

        with open(self.md_path, encoding="utf-8") as f:
            md_content = f.read()
        with open(self.json_path, encoding="utf-8") as f:
            json_content = json.load(f)

        def chunk_markdown(md_text):
            lines = md_text.splitlines()
            chunks = []
            current_chunk = []
            for line in lines:
                if line.startswith("# ") or line.startswith("## "):
                    if current_chunk:
                        chunks.append("\n".join(current_chunk).strip())
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            if current_chunk:
                chunks.append("\n".join(current_chunk).strip())
            return [c for c in chunks if c]

        def chunk_json(json_obj):
            chunks = []
            for section, data in json_obj.items():
                if isinstance(data, list):
                    chunks.append(f"{section}: " + ", ".join(data))
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict):
                            details = ", ".join([f"{key}: {val}" for key, val in v.items()])
                            chunks.append(f"{section} - {k}: {details}")
                        else:
                            chunks.append(f"{section} - {k}: {v}")
                else:
                    chunks.append(f"{section}: {data}")
            return chunks

        all_chunks = chunk_markdown(md_content) + chunk_json(json_content)
        model = self.get_model()
        embeddings = model.encode(all_chunks)
        
        for i, chunk in enumerate(all_chunks):
            self._collection.add(documents=[chunk], embeddings=[embeddings[i]], ids=[f"chunk_{i}"])
        logger.info(f"Loaded {len(all_chunks)} chunks into Chroma.")

    def retrieve(self, query, top_k=3):
        model = self.get_model()
        collection = self.get_collection()
        query_emb = model.encode([query])[0]
        results = collection.query(query_embeddings=[query_emb], n_results=top_k)
        return results['documents'][0]
