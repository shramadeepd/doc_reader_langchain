
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import shutil
import os
import json
import asyncio
from datetime import datetime
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentStatus(BaseModel):
    id: str
    filename: str
    status: str
    upload_time: str
    processing_status: str
    error: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None

class SearchResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence_score: float

class Config:
    UPLOAD_DIR = Path("uploaded_documents")
    VECTOR_DB_DIR = Path("vector_stores")
    METADATA_FILE = Path("document_metadata.json")
    ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MODEL_PATH = "llama-2-7b-chat.gguf"  # Download this file separately
    
    @classmethod
    def initialize(cls):
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(exist_ok=True)
        if not cls.METADATA_FILE.exists():
            cls.METADATA_FILE.write_text("{}")

app = FastAPI(title="Document Processing API")
Config.initialize()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qa_template = """
Context: {context}

Question: {question}

Answer the question based on the context provided. If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Answer:"""

PROMPT = PromptTemplate(
    template=qa_template,
    input_variables=["context", "question"]
)
class DocumentManager:
    def __init__(self):
        self.vector_stores: Dict[str, Chroma] = {}
        self.llm = CTransformers(
            model=Config.MODEL_PATH,
            model_type="llama",
            config={'max_new_tokens': 256, 'temperature': 0.1}
        )
        self.load_existing_documents()

    def load_existing_documents(self):
        if Config.METADATA_FILE.exists():
            metadata = json.loads(Config.METADATA_FILE.read_text())
            for doc_id, info in metadata.items():
                if info['processing_status'] == 'completed':
                    try:
                        self.load_vector_store(doc_id)
                    except Exception as e:
                        logger.error(f"Failed to load vector store for {doc_id}: {e}")

    def load_vector_store(self, doc_id: str):
        vector_store_path = Config.VECTOR_DB_DIR / doc_id
        if vector_store_path.exists():
            self.vector_stores[doc_id] = Chroma(
                persist_directory=str(vector_store_path),
                embedding_function=embeddings
            )
    async def process_document(self, file_path: Path, doc_id: str):
        try:
            self.update_document_status(doc_id, "processing")
            documents = self.load_document(file_path)
            vector_store = Chroma.from_documents(
                documents,
                embedding=embeddings,
                persist_directory=str(Config.VECTOR_DB_DIR / doc_id)
            )
            self.vector_stores[doc_id] = vector_store
            self.update_document_status(doc_id, "completed") 
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            self.update_document_status(doc_id, "failed", str(e))
            raise

    def load_document(self, file_path: Path):
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return text_splitter.split_documents(documents)

    def update_document_status(self, doc_id: str, status: str, error: str = None):
        metadata = json.loads(Config.METADATA_FILE.read_text())
        if doc_id in metadata:
            metadata[doc_id]['processing_status'] = status
            metadata[doc_id]['error'] = error
            Config.METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    def get_qa_chain(self, vector_store):
        """Create QA chain for a vector store"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}
        )
doc_manager = DocumentManager()

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> DocumentStatus:
    try:
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type. Allowed: {Config.ALLOWED_EXTENSIONS}")
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = Config.UPLOAD_DIR / f"{doc_id}{file_extension}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        metadata = {
            "id": doc_id,
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "processing_status": "pending",
            "file_path": str(file_path)
        }
        current_metadata = json.loads(Config.METADATA_FILE.read_text())
        current_metadata[doc_id] = metadata
        Config.METADATA_FILE.write_text(json.dumps(current_metadata, indent=2))
        background_tasks.add_task(doc_manager.process_document, file_path, doc_id)
        return DocumentStatus(**metadata)

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(query: SearchQuery):
    try:
        stores = []
        if query.doc_ids:
            stores = [doc_manager.vector_stores[doc_id] for doc_id in query.doc_ids 
                     if doc_id in doc_manager.vector_stores]
        else:
            stores = list(doc_manager.vector_stores.values())

        if not stores:
            raise HTTPException(400, "No processed documents available for search")
        results = []
        for store in stores:
            qa_chain = doc_manager.get_qa_chain(store)
            result = qa_chain({"query": query.query})
            if result["result"]:
                results.append(result)
        if not results:
            return SearchResponse(
                answer="I don't have enough information to answer this question.",
                sources=[],
                confidence_score=0.0
            )
        best_result = results[0]
        return SearchResponse(
            answer=best_result["result"],
            sources=[str(doc.metadata.get("source", "Unknown")) for doc in 
                    best_result.get("source_documents", [])],
            confidence_score=0.90
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")

@app.get("/documents")
async def list_documents() -> List[DocumentStatus]:
    try:
        metadata = json.loads(Config.METADATA_FILE.read_text())
        return [DocumentStatus(**doc_info) for doc_info in metadata.values()]
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(500, f"Failed to list documents: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        metadata = json.loads(Config.METADATA_FILE.read_text())
        if doc_id not in metadata:
            raise HTTPException(404, "Document not found")
        file_path = Path(metadata[doc_id]["file_path"])
        vector_store_path = Config.VECTOR_DB_DIR / doc_id
        if file_path.exists():
            file_path.unlink()
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
        if doc_id in doc_manager.vector_stores:
            del doc_manager.vector_stores[doc_id]
        del metadata[doc_id]
        Config.METADATA_FILE.write_text(json.dumps(metadata, indent=2))
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(500, f"Failed to delete document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
