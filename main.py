# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import shutil
import uvicorn

# ⚠️ IMPORTANT: import notebook functions
import rag   # document.ipynb

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
PDF_UPLOAD_DIR = "data/pdf_files"
VECTOR_DIR = "data/vector_store"

os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# API Models
# ----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    query: str
    num_sources: int
    sources: List[Dict[str, Any]]

class IndexRequest(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200

# ----------------------------------------------------------------------------
# App Init
# ----------------------------------------------------------------------------
app = FastAPI(
    title="RAG API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    try:
        rag.init_rag(PDF_UPLOAD_DIR, VECTOR_DIR)
        print("✅ RAG initialized")
    except Exception as e:
        raise RuntimeError(f"RAG init failed: {e}")

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": rag.doc_count()
    }

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    path = os.path.join(PDF_UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"status": "uploaded", "file": file.filename}

@app.post("/index")
def index_docs(req: IndexRequest):
    try:
        return rag.index_docs(req.chunk_size, req.chunk_overlap)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/query", response_model=QueryResponse)
def query_docs(req: QueryRequest):
    try:
        return rag.query_rag(req.question, req.top_k)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/clear")
def clear_docs():
    rag.clear_docs()
    return {"status": "cleared"}

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
