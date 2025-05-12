# PDF-Powered RAG Chatbot Streamlit App

A simple retrieval-augmented generation (RAG) chatbot built with Streamlit, LangChain, and Google Gemini. Upload one or more PDF documents, ask questions, and get answers with page-level citations.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)


---

## Features

- Upload multiple PDF documents via a web UI
- Splits PDF pages into overlapping text chunks
- Embeds chunks using Sentence-Transformers (`all-MiniLM-L6-v2`)
- Builds a FAISS index in-memory (no external database required)
- Uses Google Gemini (via LangChain) to answer user queries
- Returns answers with source document names and page numbers

---

## Prerequisites

- Python 3.8 or higher
- A Google Cloud API key with access to Gemini (set in `.env`)

---

## Installation

1. **Clone the repository**

   ```
   git clone https://github.com/yourusername/pdf-rag-streamlit.git
   cd pdf-rag-streamlit

2. **Create & activate a virtual environment**

    ```
    python3 -m venv venv
    source venv/bin/activate          # macOS/Linux
    .\\venv\\Scripts\\activate        # Windows

3. **Run the Streamlit app**

    ```
    streamlit run streamlit_app.py
