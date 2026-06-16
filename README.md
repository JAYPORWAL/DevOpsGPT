# 🚀 DevOpsGPT - AI-Powered DevOps Learning & Interview Assistant

DevOpsGPT is a production-ready AI-powered learning platform built using **Python, Streamlit, Google Gemini 2.5 Flash, FAISS, and Pydantic**.

The application helps learners understand DevOps concepts, prepare for interviews, practice quizzes, perform document-based Q&A using RAG, and track learning progress.

---

## ✨ Features

### 📚 Learning Hub

Generate complete study packs for:

* Docker
* Kubernetes
* Jenkins
* Terraform
* AWS
* Linux
* Git
* CI/CD

Each study pack includes:

* Concept explanation
* Architecture diagrams (Mermaid)
* Learning roadmap
* Hands-on labs
* Capstone projects
* Interview preparation
* MCQ quizzes

---

### 🎯 Interview Preparation

Generate:

* Beginner questions
* Intermediate questions
* Advanced questions

Evaluate candidate answers using AI.

Provides:

* Score
* Strengths
* Weaknesses
* Improved answer suggestions

---

### 🤖 AI Mentor Chat

Interactive DevOps mentor powered by:

* Gemini 2.5 Flash

Supports:

* Follow-up questions
* Technical discussions
* Learning guidance

---

### 🔍 RAG (Retrieval Augmented Generation)

Upload documents:

* TXT
* PDF
* DOCX

Features:

* Document ingestion
* Chunking
* Embedding generation
* FAISS vector search
* Context-aware question answering

---

### 📊 Progress Tracking

Tracks:

* Topics studied
* Quiz attempts
* Scores
* Learning history

Stored locally using JSON persistence.

---

## 🏗️ Tech Stack

### Backend

* Python 3.12
* Streamlit
* Google Gemini 2.5 Flash
* Pydantic v2

### AI / LLM

* Gemini 2.5 Flash
* Gemini Embeddings

### RAG

* LangChain
* FAISS
* Recursive Text Splitter

### Storage

* JSON
* Local File Cache

---

## 📂 Project Structure

```text
DevOpsGPT/
│
├── app.py
├── requirements.txt
├── .env.example
│
├── models/
│
├── services/
│   ├── gemini_service.py
│   ├── rag_service.py
│   └── storage_service.py
│
├── prompts/
│
├── utils/
│
├── static/
│
├── data/
│
└── README.md
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/DevOpsGPT.git

cd DevOpsGPT
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```powershell
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

### Run Application

```bash
streamlit run app.py
```

---

## 🧪 Tested Components

### Verified

* Learning Hub
* Quiz Generator
* Interview Evaluator
* Mentor Chat
* RAG Pipeline
* FAISS Retrieval
* JSON Persistence
* Streamlit UI

---

## 🔥 AI Models Used

### LLM

```text
gemini-2.5-flash
```

### Embeddings

```text
gemini-embedding-2
```

### Vector Store

```text
FAISS
```

---

## 📈 Future Improvements

* User Authentication
* Cloud Database
* Team Learning Dashboard
* Multi-user Support
* Docker Deployment
* Kubernetes Deployment
* PDF Report Export
* CI/CD Integration

---

## 👨‍💻 Author

Built as part of an AI & LLM Engineering learning journey focused on:

* Prompt Engineering
* RAG
* Vector Databases
* LLM APIs
* DevOps Education

---
