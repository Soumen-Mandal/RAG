"""
rag.py
Production-ready RAG core module
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import chromadb
import vertexai
from dotenv import load_dotenv
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer
from vertexai.generative_models import GenerativeModel

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================================
# Data Loader
# ============================================================================

class DataLoader:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def process_all_pdfs(self):
        all_docs = []
        pdf_files = Path(self.directory_path).glob("*.pdf")

        for pdf in pdf_files:
            loader = PyMuPDFLoader(str(pdf))
            docs = loader.load()

            for d in docs:
                d.metadata["source_file"] = pdf.name
                d.metadata["file_type"] = "pdf"

            all_docs.extend(docs)

        return all_docs


# ============================================================================
# Document Splitter
# ============================================================================

class DocumentSplitter:
    def __init__(self, documents, chunk_size=1000, chunk_overlap=200):
        self.documents = documents
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(self.documents)


# ============================================================================
# Embeddings
# ============================================================================

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)


# ============================================================================
# Vector Store
# ============================================================================

class VectorStore:
    def __init__(self, persist_directory: str, collection_name="pdf_documents"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, chunks, embeddings):
        ids, docs, metas, embs = [], [], [], []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            docs.append(chunk.page_content)
            metas.append(chunk.metadata)
            embs.append(emb.tolist())

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs
        )


# ============================================================================
# Retriever
# ============================================================================

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingManager):
        self.vs = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int):
        q_emb = self.embedder.embed([query])[0]

        res = self.vs.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k
        )

        docs = []
        if res["documents"]:
            for i, (doc, meta, dist) in enumerate(
                zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
            ):
                docs.append({
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist,
                    "rank": i + 1
                })

        return docs


# ============================================================================
# Gemini
# ============================================================================

class GeminiRAG:
    def __init__(self):
        key = os.getenv("VERTEX_AI_KEY_PATH")
        project = os.getenv("VERTEX_AI_PROJECT_ID")
        location = os.getenv("VERTEX_AI_LOCATION", "us-central1")

        creds = service_account.Credentials.from_service_account_file(key)
        vertexai.init(project=project, location=location, credentials=creds)

        self.model = GenerativeModel("gemini-2.5-flash")

    def generate(self, query: str, docs: List[Dict[str, Any]]):
        context = "\n\n".join(
            f"[{d['rank']}] {d['content']}" for d in docs
        )

        prompt = f"""
Use the context below to answer the question.
If the answer is not in the context, say so.

Context:
{context}

Question:
{query}

Answer:
"""

        resp = self.model.generate_content(prompt)
        return resp.text


# ============================================================================
# RAG PIPELINE (PUBLIC API)
# ============================================================================

class RAGPipeline:
    def __init__(self, pdf_dir: str, vector_dir: str):
        self.embedder = EmbeddingManager()
        self.vs = VectorStore(vector_dir)
        self.retriever = RAGRetriever(self.vs, self.embedder)
        self.llm = GeminiRAG()
        self.pdf_dir = pdf_dir

    def index(self, chunk_size=1000, chunk_overlap=200):
        docs = DataLoader(self.pdf_dir).process_all_pdfs()
        chunks = DocumentSplitter(docs, chunk_size, chunk_overlap).split()
        embeddings = self.embedder.embed([c.page_content for c in chunks])
        self.vs.add(chunks, embeddings)

        return {
            "documents": len(docs),
            "chunks": len(chunks)
        }

    def query(self, question: str, top_k=5):
        docs = self.retriever.retrieve(question, top_k)
        answer = self.llm.generate(question, docs)

        return {
            "query": question,
            "answer": answer,
            "num_sources": len(docs),
            "sources": docs
        }


# ============================================================================
# Singleton helpers (used by API)
# ============================================================================

_rag = None

def init_rag(pdf_dir: str, vector_dir: str):
    global _rag
    _rag = RAGPipeline(pdf_dir, vector_dir)

def index_docs(chunk_size=1000, chunk_overlap=200):
    return _rag.index(chunk_size, chunk_overlap)

def query_rag(question: str, top_k=5):
    return _rag.query(question, top_k)

def doc_count():
    return _rag.vs.collection.count()

def clear_docs():
    _rag.vs.client.delete_collection(_rag.vs.collection_name)
