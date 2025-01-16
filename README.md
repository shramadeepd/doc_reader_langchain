# doc_reader_langchain
## Prerequisites

- Python 3.8+
- Git

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd doc_reader_langchain
```

### 2. Set Up the Document Processing API (FastAPI)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn python-multipart langchain chromadb unstructured 
pip install python-docx pypdf sentence-transformers ctransformers

# Download Llama 2 model
# Option 1: Using wget
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O llama-2-7b-chat.gguf

# Option 2: Manual download
# Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# Download: llama-2-7b-chat.Q4_K_M.gguf
# Rename to: llama-2-7b-chat.gguf

# Create necessary directories
mkdir uploaded_documents vector_stores
```

## Running the Application

### 1. Start the Document Processing API

```bash
cd doc_reader_langchain
python api.py
```

## API Documentation

### FastAPI Document Processor Endpoints

- `POST /upload`: Upload documents
  ```bash
  curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
  ```

- `POST /search`: Search documents
  ```bash
  curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{"query": "your search query"}'
  ```

- `GET /documents`: List all documents
  ```bash
  curl http://localhost:8000/documents
  ```
## Integration Guide

### 1. Frontend to Node.js Backend Integration

Update `frontend/src/services/api.js`:

```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
```

### 2. Node.js Backend to FastAPI Integration

Update `backend/src/services/documentService.js`:

```javascript
const axios = require('axios');

const fastApiClient = axios.create({
  baseURL: process.env.FASTAPI_URL,
});

const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return await fastApiClient.post('/upload', formData);
};

const searchDocuments = async (query) => {
  return await fastApiClient.post('/search', { query });
};

module.exports = {
  uploadDocument,
  searchDocuments,
};
```

## Development Workflow

1. Start all services in development mode
2. Make changes to the code
3. Frontend changes will hot-reload
4. Backend changes will auto-restart the server
5. FastAPI changes require manual restart

## Common Issues and Solutions

1. **CORS Issues**
   - Ensure CORS is properly configured in both FastAPI and Node.js
   - Check browser console for CORS errors
   - Verify API URLs in environment files

2. **File Upload Errors**
   - Check file size limits
   - Verify supported file types
   - Ensure proper directory permissions

3. **Search Not Working**
   - Verify document processing completed successfully
   - Check vector store initialization
   - Confirm model files are present

## Deployment

### Development Environment
- Use provided scripts with hot-reload
- MongoDB running locally
- All services on localhost

### Production Environment
FastAPI:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```
