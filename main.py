from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import get_rag_chain
import os 
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
qa_chain = get_rag_chain()

class Query(BaseModel):
    question : str

@app.post("/ask")
def ask_question(query: Query):
    result = qa_chain(query.question)

    return {
        "answer":result['result'],
        "source":[doc.metadata for doc in result['source_documents']]
    }    

