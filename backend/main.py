

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

filepath = "data.txt"
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
        # db.execute(
        #     text("INSERT INTO chunk_embedding (chunk_text, embedding) VALUES (:chunk, :vector)"),
        #     {"chunk": chunk, "vector": vector}
        # )
    db.commit()
    db.close()
    print("Chunking and embedding process completed.")

store = {}

@app.post("/api/chat/")
async def chat(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    prompt = data.get("prompt", "")
    filename = data.get("filename", "")
    
    if not prompt:
        return {"error": "Prompt is required."}

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    user_vector = embeddings.embed_query(prompt)
    items = db.query(ChunkEmbedding.chunk_text, ChunkEmbedding.embedding).all()
    import numpy as np
    from numpy.linalg import norm

    def cosine_similarity(vec1, vec2):
        dot = np.dot(vec1, vec2)
        return dot / (norm(vec1) * norm(vec2)) if norm(vec1) and norm(vec2) else 0

    similarities = []
    for chunk_text, emb_str in items:
        emb_vec = np.fromstring(emb_str.strip("[]"), sep=",")
        score = cosine_similarity(user_vector, emb_vec)
        similarities.append((score, chunk_text))

    similarities.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for _, chunk in similarities[:3]]
    rag_context = "\n\n".join(top_chunks)

    # Step 4: If additional instructions from filename, include them
    file_instruction = ""
    if filename:
        item = db.query(FileData).filter(FileData.name == filename).first()
        if item:
            file_instruction = item.description

    # Final system instruction
    system_instruction = f"{texts}\n\n{rag_context}"

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ])

    chain = prompt_template | model

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="message",
        history_messages_key="history",
    )

    response = with_message_history.invoke(
        {"message": prompt},
        config={"configurable": {"session_id": "user123"}}
    )

    return {"response": response.content}




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

