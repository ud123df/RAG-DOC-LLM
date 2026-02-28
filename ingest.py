import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def ingest_pdf(file_path):
    print("📄 Loading PDF...")

    loader = PyMuPDFLoader("C:/Users/UDIT/Desktop/Udit_Sharma_resume.pdf")
    documents = loader.load()

    print("✂ Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Clean empty or invalid chunks
    clean_chunks = []

    for chunk in chunks:
     if chunk.page_content and isinstance(chunk.page_content, str):
        text = chunk.page_content.strip()
        if len(text) > 20:   # remove tiny garbage chunks
            chunk.page_content = text
            clean_chunks.append(chunk)
    print(f"Clean chunks: {len(clean_chunks)}")

    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("💾 Saving to vector DB...")
    vectorstore = Chroma.from_documents(
        documents=clean_chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )

    vectorstore.persist()

    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest_pdf("C:/Users/UDIT/Desktop/Udit_Sharma_resume.pdf")   # <-- IMPORTANT