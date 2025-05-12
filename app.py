import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# 1) Load your Gemini API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=api_key
)
# 3) Streamlit UI
st.title("📄🔍 PDF-Powered RAG Chatbot")

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("⌛ Processing PDFs…"):
        # 4) Load & split
        docs = []
        for uf in uploaded_files:
            reader = PdfReader(uf)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                docs.append(Document(
                    page_content=text,
                    metadata={"source": uf.name, "page": i + 1}
                ))

        # 2) Split into chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # 5) Embed & index
        embed_model = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embed_model)

        # 6) Build QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

    # 7) Ask questions
    query = st.text_input("❓ Ask a question about your PDFs")
    if query:
        with st.spinner("🧠 Thinking…"):
            resp = qa_chain({"query": query})

        st.subheader("📝 Answer")
        st.write(resp["result"])

        st.subheader("📚 Sources")
        for doc in resp["source_documents"]:
            src = doc.metadata.get("source", "unknown")
            pg  = doc.metadata.get("page", "unknown")
            st.write(f"- {src} — page {pg}")
