# 📄 AI-Driven Document Search & Knowledge Retrieval System  
### Advanced Retrieval-Augmented Generation (RAG) Framework

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system engineered for document-grounded question answering using **Hybrid Retrieval (BM25 + Dense Embeddings)** and **LLM-powered contextual response generation**.

This application enables semantic document search, contextual reasoning, and hallucination-controlled responses strictly derived from uploaded PDFs.

---

## 🧠 Project Overview

Large Language Models (LLMs) often generate responses without grounding in verified sources.  
This project addresses that limitation by implementing:

- Hybrid Sparse + Dense Retrieval  
- Persistent Vector Storage  
- Context-Aware Prompt Engineering  
- Controlled LLM Inference  

The chatbot generates answers **only from the selected document**, ensuring reliability and traceability.

---


---

## 🔬 Core Components

### 1️⃣ Document Ingestion Pipeline
- PDF parsing using **PyPDF**
- Recursive text chunking
- Metadata enrichment
- Embedding generation
- Persistent storage in **ChromaDB**

---

### 2️⃣ Embedding System
- **Sentence-Transformers (HuggingFace Embeddings)**
- Dense semantic vector representations
- Cosine similarity-based nearest neighbor search

---

### 3️⃣ Hybrid Retrieval Strategy

#### 🔹 BM25 (Sparse Retrieval)
- TF-IDF based lexical scoring
- Keyword relevance ranking
- Efficient exact phrase matching

#### 🔹 Dense Retrieval
- Semantic similarity search
- Context-aware embedding comparison
- Improved recall for meaning-based queries

#### 🔹 Fusion Strategy
Combines sparse and dense retrieval to achieve:
- Higher precision
- Better contextual recall
- Reduced retrieval bias

---

### 4️⃣ Vector Database
- **ChromaDB**
- Persistent local vector store
- Fast similarity search
- Document-level namespace isolation

---

### 5️⃣ LLM Integration
- **Groq API**
- Retrieval-Augmented prompt injection
- Context-grounded answer synthesis
- Hallucination mitigation through strict context restriction

---

### 6️⃣ Context Restriction Mechanism
- Active PDF selection
- Prevents cross-document contamination
- Ensures document-scoped responses

---

## ⚙️ Tech Stack

### Frontend
- Streamlit

### AI / NLP
- LangChain
- LangChain Community
- Sentence-Transformers
- BM25 Retrieval
- Retrieval-Augmented Generation (RAG)

### Vector Storage
- ChromaDB

### Document Processing
- PyPDF

### Environment & Utilities
- Python 3.12
- python-dotenv
- Virtual Environment (venv)

---

## 🚀 Key Features

- ✅ Hybrid Retrieval (BM25 + Dense Embeddings)
- ✅ Multi-PDF Support
- ✅ Active Document Isolation
- ✅ Persistent Vector Indexing
- ✅ Context-Grounded Generation
- ✅ Hallucination Reduction
- ✅ Developer-Friendly Modular Architecture
- ✅ Fast LLM Responses via Groq

---

---

## 📊 Improvements Over Basic RAG Systems

| Basic RAG | This Implementation |
|-----------|--------------------|
| Single retrieval | Hybrid BM25 + Dense |
| Single document | Multi-document isolation |
| Weak ranking | Retrieval fusion strategy |
| Generic responses | Context-controlled generation |
| No persistence | Persistent vector storage |

---

## 🧪 Use Cases

- Research Paper QA Systems
- Academic Study Assistants
- Enterprise Knowledge Bases
- Legal Document Analysis
- Technical Documentation Search

---

## 📈 Performance Characteristics

- Low-latency retrieval
- No GPU dependency
- Scalable embedding storage
- Deterministic context selection
- Reduced hallucination rate

---

## 🔐 Security & Privacy

- Local vector persistence
- API keys managed via `.env`
- Sensitive files ignored via `.gitignore`
- No external document storage

---

### ⚡ Core System Capabilities Enabled
- Hybrid Semantic + Lexical Search
- Context-Aware Prompt Engineering
- Hallucination Reduction via Retrieval Grounding
- Persistent Knowledge Indexing
- Multi-Document Query Handling
- GPU-free Local Deployment

## 👩‍💻 Author

- Abhinaya Nalivela
Computer Science Student | AI Enthusiast

