from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
load_dotenv()
def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(BASE_DIR,"..", "vector_db")

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs = {"k":3})

    llm = ChatGroq(
        model = "llama-3.1-8b-instant",
        temperature=0.1
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever=retriever,
        return_source_documents = True
    )

    return qa_chain