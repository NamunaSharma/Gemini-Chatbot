

from fastapi import Depends, FastAPI,Request,File,UploadFile
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from fastapi import Depends
import logging
from sqlalchemy import text
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from numpy.linalg import norm


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

app = FastAPI()


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

@app.on_event("startup")
def get_chunk():
    db = SessionLocal()
    items = db.query(FileData.description).first()  # fetch first description

    if not items:
        print("No description found")
        db.close()
        return

    description = items[0] 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(description)

    print(f"Created {len(chunks)} chunks with recursive splitting.")
    for chunk in chunks:
        vector = embeddings.embed_query(chunk)
        print(vector[:1])
        db.execute(
            text("INSERT INTO chunk_embedding (chunk_text, embedding) VALUES (:chunk, :vector)"),
            {"chunk": chunk, "vector": vector}
        )
    db.commit()
    db.close()
    print("Chunking and embedding process completed.")


@app.post("/userquery/")
def user_query(request:Request):
    db= SessionLocal()
    user_query = request.query_params['query']
    user_vector = embeddings.embed_query(user_query)
    items = db.query(ChunkEmbedding.chunk_text, ChunkEmbedding.embedding).all()  # fetch all embedding and the chunks
    def cosine_similarity_np(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        magnitude_vec1 = norm(vec1)
        magnitude_vec2 = norm(vec2)
        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0  # Handle cases where a vector is zero
        return dot_product / (magnitude_vec1 * magnitude_vec2)
    similarities = []

    for chunk_text, embedding in items:
        embedding = np.fromstring(embedding.strip("[]"), sep=',')
        similarity = cosine_similarity_np(user_vector, embedding)
        similarities.append((similarity, chunk_text))
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_chunks = similarities[:3]
    response_chunks = [chunk for _, chunk in top_chunks]
    response_text = "\n\n".join(response_chunks)
   
    print(f"Cosine Similarity (NumPy): {similarity}")
    db.close()
    return {"response": response_text}







@app.post("/api/content/", response_model=FileDataResponse)
async def create_item(item: FileDataCreate, db: Session = Depends(get_db)):
    db_item = FileData(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


