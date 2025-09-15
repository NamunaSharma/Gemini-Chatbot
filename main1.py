

from fastapi import Depends, FastAPI,Request,File,UploadFile
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from fastapi import Depends
import time
from sqlalchemy import text
from fastapi import Request, Depends
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import numpy as np

import time
import logging
from sqlalchemy import text
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from numpy.linalg import norm


from sklearn.feature_extraction.text import TfidfVectorizer



app = FastAPI()
import google.generativeai as genai
from dotenv import load_dotenv
import os

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from typing import List
# genai.configure(api_key="AIzaSyD1J0vqebuEq6W1SCXnJh4rCErVD13yn4s")


from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from sqlalchemy import create_engine, Column, Text
from sqlalchemy.orm import sessionmaker, declarative_base


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 
DATABASE_URL = "postgresql://postgres:password@localhost:5434/postgres"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

filepath = "chat.txt"
texts=''
with open(filepath, 'r', encoding="utf-8") as f:
        content = f.read().strip()
        texts=content


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
class FileData(Base):
    __tablename__ = "file_data"  # double underscores
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)

class ChunkEmbedding(Base):
    __tablename__ = "chunk_embedding"
    id = Column(Integer, primary_key=True, index=True)
    chunk_text = Column(Text)
    embedding = Column(Text)
Base.metadata.create_all(bind=engine) 

from pydantic import BaseModel

# Define a Pydantic model
class FileDataCreate(BaseModel):
    name: str
    description: str

class FileDataResponse(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        from_attributes = True


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")




app.add_middleware(
    CORSMiddleware,
    allow_origins= ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def get_chunk():
    db = SessionLocal()
    items = db.query(FileData.description).first()  

    if not items:
        print("No description found")
        db.close()
        return
    # for item in items:
    #     # Check if this file's chunks already exist
    #     existing_chunks = db.query(ChunkEmbedding).filter(ChunkEmbedding.chunk_text.like(f"%{item.chunks[:20]}%")).first()
    #     if existing_chunks:
    #         print(f"Chunks for '{item.name}' already exist. Skipping...")
    #         continue

    # description = items[0] 
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=800,
    #     chunk_overlap=50,
    #     separators=["\n\n", "\n", " ", ""]
    # )
    # chunks = text_splitter.split_text(description)

    # print(f"Created {len(chunks)} chunks with recursive splitting.")
    # for chunk in chunks:
    #     vector = embeddings.embed_query(chunk)
    #     print(vector[:1])
    #     db.execute(
    #         text("INSERT INTO chunk_embedding (chunk_text, embedding) VALUES (:chunk, :vector)"),
    #         {"chunk": chunk, "vector": vector}
    #     )
    # db.commit()
    # db.close()
    # print("Chunking and embedding process completed.")


# Query Expansion Function
def expand_query(prompt: str):
    expansion_dict = {
        "glasses": ["spectacles", "eyewear"],
        "sunglasses": ["shades", "sun glasses"],
        "vision": ["eyesight", "eye health"],
        "lens": ["optical lens", "eyepiece"],
        "frame": ["eyeglass frame", "mounting"]
    }
    terms = prompt.split()
    expanded_terms = []
    for term in terms:
        expanded_terms.append(term)
        if term.lower() in expansion_dict:
            expanded_terms.extend(expansion_dict[term.lower()])
    return " ".join(expanded_terms)

store = {}
@app.post("/api/chat/")
async def chat(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    prompt = data.get("prompt", "")
    filename = data.get("filename", "")
    retrieval_method = data.get("retrieval_method", "dense")

    if not prompt:
        return {"error": "Prompt is required."}

    # --- Start timer ---
    start_time = time.time()

    # --- Expand query if needed ---
    if retrieval_method == "expanded":
        prompt_expanded = expand_query(prompt)
    else:
        prompt_expanded = prompt

    # --- Get embedding for query ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    user_vector = embeddings.embed_query(prompt_expanded)

    # --- Dense Retrieval with pgvector ---
    if retrieval_method in ["dense", "expanded", "hybrid"]:
        query_vec_str = "[" + ",".join(str(x) for x in user_vector) + "]"

        dense_results = db.execute(
            text("""
                SELECT chunk_text, 1 - (embedding <=> :query_vec) AS score
                FROM chunk_embedding
                ORDER BY embedding <=> :query_vec
                LIMIT 10
            """),
            {"query_vec": query_vec_str}
        ).fetchall()



        dense_chunks = [row[0] for row in dense_results]
        dense_scores = [row[1] for row in dense_results]
    else:
        dense_chunks, dense_scores = [], []

    # --- Sparse Retrieval (TF-IDF in Python) ---
    if retrieval_method in ["sparse", "hybrid", "expanded"]:
        items = db.query(ChunkEmbedding.chunk_text).all()
        corpus = [i[0] for i in items]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec_sparse = vectorizer.transform([prompt_expanded])
        sparse_scores = sk_cosine(tfidf_matrix, query_vec_sparse).flatten()

        top_idx = np.argsort(-sparse_scores)[:10]
        sparse_chunks = [corpus[i] for i in top_idx]
        sparse_scores = sparse_scores[top_idx]
    else:
        sparse_chunks, sparse_scores = [], []

    # --- Merge results ---
    if retrieval_method == "dense":
        top_chunks = dense_chunks[:3]
    elif retrieval_method == "sparse":
        top_chunks = sparse_chunks[:3]
    else:  # hybrid/expanded
        combined = list(set(dense_chunks[:5] + sparse_chunks[:5]))
        top_chunks = combined[:3]

    rag_context = "\n\n".join(top_chunks)

    # --- End timer ---
    end_time = time.time()

    return {
        "retrieval_method": retrieval_method,
        "expanded_query": prompt_expanded,
        "top_chunks": top_chunks,
        "retrieval_time_seconds": end_time - start_time,
        "dense_scores": dense_scores,
        "sparse_scores": sparse_scores.tolist() if len(sparse_scores) else []
    }
@app.post("/api/content/", response_model=FileDataResponse)
async def create_item(item: FileDataCreate, db: Session = Depends(get_db)):
    db_item = FileData(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/api/prompts/")
def get_files(db: Session = Depends(get_db)):
    items = db.query(FileData).all()
    return [{"id": item.id, "name": item.name, "description": item.description} for item in items]
