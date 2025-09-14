# from fastapi import FastAPI, Request, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from sqlalchemy.orm import Session, sessionmaker, declarative_base
# from sqlalchemy import create_engine, Column, Integer, String, Text, text
# from pydantic import BaseModel
# # from langchain.memory import InMemoryChatMessageHistory
# import os, time, numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from sklearn.feature_extraction.text import TfidfVectorizer



# app = FastAPI()
# import google.generativeai as genai
# from dotenv import load_dotenv
# import os

# from sqlalchemy import create_engine, Column, Integer, String
# from sqlalchemy.ext.declarative import declarative_base
# from typing import List
# # genai.configure(api_key="AIzaSyD1J0vqebuEq6W1SCXnJh4rCErVD13yn4s")


# from pydantic import BaseModel
# from fastapi import FastAPI, UploadFile, File
# from sqlalchemy import create_engine, Column, Text
# from sqlalchemy.orm import sessionmaker, declarative_base


# import time
# from fastapi import Request, Depends
# from sqlalchemy.orm import Session
# import numpy as np
# from numpy.linalg import norm
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 
# DATABASE_URL = "postgresql://postgres:password@localhost:5434/postgres"
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# filepath = "chat.txt"
# texts=''
# with open(filepath, 'r', encoding="utf-8") as f:
#         content = f.read().strip()
#         texts=content


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
        
# class FileData(Base):
#     __tablename__ = "file_data"  # double underscores
#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String, index=True)
#     description = Column(String)

# class ChunkEmbedding(Base):
#     __tablename__ = "chunk_embedding"
#     id = Column(Integer, primary_key=True, index=True)
#     chunk_text = Column(Text)
#     embedding = Column(Text)
# Base.metadata.create_all(bind=engine) 

# from pydantic import BaseModel

# # Define a Pydantic model
# class FileDataCreate(BaseModel):
#     name: str
#     description: str

# class FileDataResponse(BaseModel):
#     id: int
#     name: str
#     description: str

#     class Config:
#         from_attributes = True


# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")




# app.add_middleware(
#     CORSMiddleware,
#     allow_origins= ["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.on_event("startup")
# def get_chunk():
#     db = SessionLocal()
#     items = db.query(FileData.description).first()  

#     if not items:
#         print("No description found")
#         db.close()
#         return
#     for item in items:
#         # Check if this file's chunks already exist
#         existing_chunks = db.query(ChunkEmbedding).filter(ChunkEmbedding.chunk_text.like(f"%{item.chunks[:20]}%")).first()
#         if existing_chunks:
#             print(f"Chunks for '{item.name}' already exist. Skipping...")
#             continue

#     description = items[0] 
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=50,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunks = text_splitter.split_text(description)

#     print(f"Created {len(chunks)} chunks with recursive splitting.")
#     for chunk in chunks:
#         vector = embeddings.embed_query(chunk)
#         print(vector[:1])
#         db.execute(
#             text("INSERT INTO chunk_embedding (chunk_text, embedding) VALUES (:chunk, :vector)"),
#             {"chunk": chunk, "vector": vector}
#         )
#     db.commit()
#     db.close()
#     print("Chunking and embedding process completed.")

# store = {}


# # Query Expansion Function
# def expand_query(prompt: str):
#     # Simple Synonym-based Expansion for eyewear domain
#     expansion_dict = {
#         "glasses": ["spectacles", "eyewear"],
#         "sunglasses": ["shades", "sun glasses"],
#         "vision": ["eyesight", "eye health"],
#         "lens": ["optical lens", "eyepiece"],
#         "frame": ["eyeglass frame", "mounting"]
#     }
#     terms = prompt.split()
#     expanded_terms = []
#     for term in terms:
#         expanded_terms.append(term)
#         if term.lower() in expansion_dict:
#             expanded_terms.extend(expansion_dict[term.lower()])
#     return " ".join(expanded_terms)

# # @app.post("/api/chat/")
# # async def chat(request: Request, db: Session = Depends(get_db)):
# #     data = await request.json()
# #     prompt = data.get("prompt", "")
# #     filename = data.get("filename", "")
# #     retrieval_method = data.get("retrieval_method", "dense")  # dense/sparse/hybrid/expanded

# #     if not prompt:
# #         return {"error": "Prompt is required."}

# #     # --- Load chunks from DB ---
# #     items = db.query(ChunkEmbedding.chunk_text, ChunkEmbedding.embedding).all()
# #     if not items:
# #         return {"error": "No chunks found in database."}

# #     corpus = [chunk_text for chunk_text, _ in items]

# #     # --- Measure start time ---
# #     start_time = time.time()

# #     # --- Expand query if method == expanded ---
# #     if retrieval_method == "expanded":
# #         prompt_expanded = expand_query(prompt)
# #     else:
# #         prompt_expanded = prompt

# #     # --- Dense Retrieval ---
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# #     user_vector = embeddings.embed_query(prompt_expanded)

# #     dense_scores = []
# #     for chunk_text, emb_str in items:
# #         emb_vec = np.fromstring(emb_str.strip("[]"), sep=",")
# #         score = np.dot(user_vector, emb_vec) / (norm(user_vector) * norm(emb_vec) + 1e-9)
# #         dense_scores.append(score)
# #     dense_scores = np.array(dense_scores)

# #     # --- Sparse Retrieval (TF-IDF) ---
# #     vectorizer = TfidfVectorizer()
# #     tfidf_matrix = vectorizer.fit_transform(corpus)
# #     query_vec_sparse = vectorizer.transform([prompt_expanded])
# #     sparse_scores = sk_cosine(tfidf_matrix, query_vec_sparse).flatten()

# #     # --- Normalize scores ---
# #     def normalize(arr):
# #         return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

# #     dense_scores = normalize(dense_scores)
# #     sparse_scores = normalize(sparse_scores)

# #     # --- Select retrieval method ---
# #     if retrieval_method == "sparse":
# #         final_scores = sparse_scores
# #     elif retrieval_method == "dense":
# #         final_scores = dense_scores
# #     elif retrieval_method == "expanded":
# #         final_scores = (dense_scores + sparse_scores) / 2  
# #     else:  
# #         final_scores = (dense_scores + sparse_scores) / 2

# #     # --- Rank top chunks ---
# #     ranked_indices = np.argsort(-final_scores)
# #     top_chunks = [corpus[i] for i in ranked_indices[:3]]
# #     rag_context = "\n\n".join(top_chunks)

# #     # --- End time ---
# #     end_time = time.time()
# #     retrieval_time = end_time - start_time

# #     # --- Optional: Add file-specific instruction ---
# #     file_instruction = ""
# #     if filename:
# #         item = db.query(FileData).filter(FileData.name == filename).first()
# #         if item:
# #             file_instruction = item.description

# #     # --- Build system instruction ---
# #     system_instruction = f"{texts}\n\n{rag_context}\n\n{file_instruction}"

# #     # --- Session history ---
# #     def get_session_history(session_id: str):
# #         if session_id not in store:
# #             store[session_id] = InMemoryChatMessageHistory()
# #         return store[session_id]

# #     model = ChatGoogleGenerativeAI(
# #         model="gemini-1.5-flash",
# #         api_key=os.getenv("GOOGLE_API_KEY"),
# #         temperature=0.3,
# #     )
# #     prompt_template = ChatPromptTemplate.from_messages([
# #         ("system", system_instruction),
# #         MessagesPlaceholder(variable_name="history"),
# #         ("human", "{message}"),
# #     ])
# #     chain = prompt_template | model
# #     with_message_history = RunnableWithMessageHistory(
# #         chain,
# #         get_session_history,
# #         input_messages_key="message",
# #         history_messages_key="history",
# #     )
# #     response = with_message_history.invoke(
# #         {"message": prompt},
# #         config={"configurable": {"session_id": "user123"}}
# #     )

# #     return {
# #         "response": response.content,
# #         "retrieval_method": retrieval_method,
# #         "top_chunks": top_chunks,
# #         "retrieval_time_seconds": retrieval_time,
# #         "dense_scores": dense_scores.tolist(),
# #         "sparse_scores": sparse_scores.tolist(),
# #         "expanded_query": prompt_expanded
# #     }







# ## working code 

# @app.post("/api/chat/")
# async def chat(request: Request, db: Session = Depends(get_db)):
#     data = await request.json()
#     prompt = data.get("prompt", "")
#     filename = data.get("filename", "")
#     retrieval_method = data.get("retrieval_method", "dense")  # dense/sparse/hybrid/expanded

#     if not prompt:
#         return {"error": "Prompt is required."}

#     # --- Load chunks from DB ---
#     items = db.query(ChunkEmbedding.chunk_text, ChunkEmbedding.embedding).all()
#     if not items:
#         return {"error": "No chunks found in database."}

#     corpus = [chunk_text for chunk_text, _ in items]

#     # --- Start timer ---
#     start_time = time.time()

#     # --- Expand query if needed ---
#     prompt_expanded = expand_query(prompt) if retrieval_method == "expanded" else prompt

#     # --- Dense retrieval ---
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
#     user_vector = embeddings.embed_query(prompt_expanded)

#     dense_scores = []
#     for chunk_text, emb_str in items:
#         emb_vec = np.fromstring(emb_str.strip("[]"), sep=",")
#         score = np.dot(user_vector, emb_vec) / (norm(user_vector) * norm(emb_vec) + 1e-9)
#         dense_scores.append(score)
#     dense_scores = np.array(dense_scores)

#     # --- Sparse retrieval ---
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(corpus)
#     query_vec_sparse = vectorizer.transform([prompt_expanded])
#     sparse_scores = sk_cosine(tfidf_matrix, query_vec_sparse).flatten()

#     # --- Normalize ---
#     def normalize(arr):
#         return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

#     dense_scores = normalize(dense_scores)
#     sparse_scores = normalize(sparse_scores)

#     # --- Merge scores based on retrieval_method ---
#     if retrieval_method == "sparse":
#         final_scores = sparse_scores
#     elif retrieval_method == "dense":
#         final_scores = dense_scores
#     else:  # hybrid / expanded
#         final_scores = (dense_scores + sparse_scores) / 2

#     # --- Rank top chunks ---
#     ranked_indices = np.argsort(-final_scores)
#     top_chunks = [corpus[i] for i in ranked_indices[:3]]
#     rag_context = "\n\n".join(top_chunks)

#     # --- File-specific instruction ---
#     # file_instruction = ""
#     # if filename:
#     #     item = db.query(FileData).filter(FileData.name == filename).first()
#     #     if item:
#     #         file_instruction = item.description

#     # --- Build system instruction ---
#     system_instruction = f"{texts}\n\n{rag_context}\n"

#     # --- Session history ---
#     def get_session_history(session_id: str):
#         if session_id not in store:
#             store[session_id] = InMemoryChatMessageHistory()
#         return store[session_id]

#     # --- Initialize model & prompt ---
#     model = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         api_key=os.getenv("GOOGLE_API_KEY"),
#         temperature=0.3,
#     )
#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system", system_instruction),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{message}"),
#     ])
#     chain = prompt_template | model
#     with_message_history = RunnableWithMessageHistory(
#         chain,
#         get_session_history,
#         input_messages_key="message",
#         history_messages_key="history",
#     )
#     response = with_message_history.invoke(
#         {"message": prompt},
#         config={"configurable": {"session_id": "user123"}}
#     )

#     # --- End timer ---
#     end_time = time.time()
#     retrieval_time = end_time - start_time
#     return {
#         "response": response.content,
#         "retrieval_method": retrieval_method,
#         "top_chunks": top_chunks,
#         "retrieval_time_seconds": retrieval_time,
#         "dense_scores": dense_scores.tolist(),
#         "sparse_scores": sparse_scores.tolist(),
#         "expanded_query": prompt_expanded
#     }











# # # Query Expansion Function
# # def expand_query(prompt: str):
# #     expansion_dict = {
# #         "glasses": ["spectacles", "eyewear"],
# #         "sunglasses": ["shades", "sun glasses"],
# #         "vision": ["eyesight", "eye health"],
# #         "lens": ["optical lens", "eyepiece"],
# #         "frame": ["eyeglass frame", "mounting"]
# #     }
# #     terms = prompt.split()
# #     expanded_terms = []
# #     for term in terms:
# #         expanded_terms.append(term)
# #         if term.lower() in expansion_dict:
# #             expanded_terms.extend(expansion_dict[term.lower()])
# #     return " ".join(expanded_terms)


# # @app.post("/api/chat/")
# # async def chat(request: Request, db: Session = Depends(get_db)):
# #     data = await request.json()
# #     prompt = data.get("prompt", "")
# #     filename = data.get("filename", "")
# #     retrieval_method = data.get("retrieval_method", "dense")

# #     if not prompt:
# #         return {"error": "Prompt is required."}

# #     # --- Start timer ---
# #     start_time = time.time()

# #     # --- Expand query if needed ---
# #     if retrieval_method == "expanded":
# #         prompt_expanded = expand_query(prompt)
# #     else:
# #         prompt_expanded = prompt

# #     # --- Get embedding for query ---
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# #     user_vector = embeddings.embed_query(prompt_expanded)

# #     # --- Dense Retrieval with pgvector ---
# #     if retrieval_method in ["dense", "expanded", "hybrid"]:
# #         query_vec_str = "[" + ",".join(str(x) for x in user_vector) + "]"

# #         dense_results = db.execute(
# #             text("""
# #                 SELECT chunk_text, 1 - (embedding <=> :query_vec) AS score
# #                 FROM chunk_embedding
# #                 ORDER BY embedding <=> :query_vec
# #                 LIMIT 10
# #             """),
# #             {"query_vec": query_vec_str}
# #         ).fetchall()



# #         dense_chunks = [row[0] for row in dense_results]
# #         dense_scores = [row[1] for row in dense_results]
# #     else:
# #         dense_chunks, dense_scores = [], []

# #     # --- Sparse Retrieval (TF-IDF in Python) ---
# #     if retrieval_method in ["sparse", "hybrid", "expanded"]:
# #         items = db.query(ChunkEmbedding.chunk_text).all()
# #         corpus = [i[0] for i in items]

# #         vectorizer = TfidfVectorizer()
# #         tfidf_matrix = vectorizer.fit_transform(corpus)
# #         query_vec_sparse = vectorizer.transform([prompt_expanded])
# #         sparse_scores = sk_cosine(tfidf_matrix, query_vec_sparse).flatten()

# #         top_idx = np.argsort(-sparse_scores)[:10]
# #         sparse_chunks = [corpus[i] for i in top_idx]
# #         sparse_scores = sparse_scores[top_idx]
# #     else:
# #         sparse_chunks, sparse_scores = [], []

# #     # --- Merge results ---
# #     if retrieval_method == "dense":
# #         top_chunks = dense_chunks[:3]
# #     elif retrieval_method == "sparse":
# #         top_chunks = sparse_chunks[:3]
# #     else:  # hybrid/expanded
# #         combined = list(set(dense_chunks[:5] + sparse_chunks[:5]))
# #         top_chunks = combined[:3]

# #     rag_context = "\n\n".join(top_chunks)

# #     # --- End timer ---
# #     end_time = time.time()

# #     return {
# #         "retrieval_method": retrieval_method,
# #         "expanded_query": prompt_expanded,
# #         "top_chunks": top_chunks,
# #         "retrieval_time_seconds": end_time - start_time,
# #         "dense_scores": dense_scores,
# #         "sparse_scores": sparse_scores.tolist() if len(sparse_scores) else []
# #     }
# @app.post("/api/content/", response_model=FileDataResponse)
# async def create_item(item: FileDataCreate, db: Session = Depends(get_db)):
#     db_item = FileData(name=item.name, description=item.description)
#     db.add(db_item)
#     db.commit()
#     db.refresh(db_item)
#     return db_item

# @app.get("/api/prompts/")
# def get_files(db: Session = Depends(get_db)):
#     items = db.query(FileData).all()
#     return [{"id": item.id, "name": item.name, "description": item.description} for item in items]










from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Text, text
from pydantic import BaseModel
import os, time, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------
# Load environment & configure API
# -------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------------
# Database Setup
# -------------------------
DATABASE_URL = "postgresql://postgres:password@localhost:5434/postgres"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load system text
# -------------------------
filepath = "chat.txt"
texts = ''
with open(filepath, 'r', encoding="utf-8") as f:
    texts = f.read().strip()

# -------------------------
# Database Models
# -------------------------
class FileData(Base):
    __tablename__ = "file_data"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)

class ChunkEmbedding(Base):
    __tablename__ = "chunk_embedding"
    id = Column(Integer, primary_key=True, index=True)
    chunk_text = Column(Text)
    embedding = Column(Text)  # stored as pgvector

Base.metadata.create_all(bind=engine)

# -------------------------
# Pydantic Schemas
# -------------------------
class FileDataCreate(BaseModel):
    name: str
    description: str

class FileDataResponse(BaseModel):
    id: int
    name: str
    description: str
    class Config:
        from_attributes = True

# -------------------------
# Database Dependency
# -------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------
# Query Expansion
# -------------------------
def expand_query(prompt: str):
    expansion_dict = {
        "glasses": ["spectacles", "eyewear"],
        "spectacles": ["glasses", "eyewear"],
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

# -------------------------
# Embedding Model
# -------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# -------------------------
# Precompute TF-IDF at startup
# -------------------------
tfidf_vectorizer = None
tfidf_matrix = None
corpus = []

def precompute_tfidf():
    global tfidf_vectorizer, tfidf_matrix, corpus
    db = SessionLocal()
    items = db.query(ChunkEmbedding.chunk_text).all()
    corpus = [i[0] for i in items]
    if corpus:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    db.close()

precompute_tfidf()

# -------------------------
# Helper: Create chunks
# -------------------------
def create_and_store_chunks(description: str, db: Session):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(description)
    for chunk in chunks:
        vector = embeddings.embed_query(chunk)
        vector_str = "[" + ",".join(str(x) for x in vector) + "]"
        db.execute(
            "INSERT INTO chunk_embedding (chunk_text, embedding) VALUES (:chunk, :vector)",
            {"chunk": chunk, "vector": vector_str}
        )
    db.commit()
    return len(chunks)

# -------------------------
# API Endpoints
# -------------------------
@app.get("/api/prompts/")
def get_files(db: Session = Depends(get_db)):
    items = db.query(FileData).all()
    return [{"id": item.id, "name": item.name, "description": item.description} for item in items]

@app.post("/api/content/", response_model=FileDataResponse)
async def create_item(item: FileDataCreate, db: Session = Depends(get_db)):
    db_item = FileData(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)

    num_chunks = create_and_store_chunks(item.description, db)
    print(f"Created {num_chunks} chunks for '{item.name}'.")

    # Recompute TF-IDF after adding new chunks
    precompute_tfidf()

    return db_item

# -------------------------
# Chat Endpoint
# -------------------------
store = {}

@app.post("/api/chat/")
async def chat(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    prompt = data.get("prompt", "")
    filename = data.get("filename", "")
    retrieval_method = data.get("retrieval_method", "dense")

    if not prompt:
        return {"error": "Prompt is required."}

    start_time = time.time()
    prompt_expanded = expand_query(prompt) if retrieval_method == "expanded" else prompt

    # --- Dense Retrieval ---
    dense_chunks, dense_scores = [], []
    if retrieval_method in ["dense", "hybrid", "expanded"]:
        user_vector = embeddings.embed_query(prompt_expanded)
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

    # --- Sparse Retrieval using precomputed TF-IDF ---
    sparse_chunks, sparse_scores = [], []
    if retrieval_method in ["sparse", "hybrid", "expanded"] and corpus:
        query_vec_sparse = tfidf_vectorizer.transform([prompt_expanded])
        sparse_scores_all = sk_cosine(tfidf_matrix, query_vec_sparse).flatten()
        top_idx = np.argsort(-sparse_scores_all)[:10]
        sparse_chunks = [corpus[i] for i in top_idx]
        sparse_scores = sparse_scores_all[top_idx]

    # --- Merge Results ---
    if retrieval_method == "dense":
        top_chunks = dense_chunks[:3]
    elif retrieval_method == "sparse":
        top_chunks = sparse_chunks[:3]
    else:
        combined = list(dict.fromkeys(dense_chunks[:5] + sparse_chunks[:5]))
        top_chunks = combined[:3]

    # --- Optional file instruction ---
    # file_instruction = ""
    # if filename:
    #     item = db.query(FileData).filter(FileData.name == filename).first()
    #     if item:
    #         file_instruction = item.description

    # --- Final system instruction ---
    system_instruction = f"{top_chunks}\n\n{texts}"

    # --- Session history ---
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # --- Chat model ---
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}")
    ])
    chain = prompt_template | model
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="message",
        history_messages_key="history"
    )

    response = with_message_history.invoke(
        {"message": prompt},
        config={"configurable": {"session_id": "user123"}}
    )

    end_time = time.time()

    return {
        "retrieval_method": retrieval_method,
        "expanded_query": prompt_expanded,
        "top_chunks": top_chunks,
        "retrieval_time_seconds": end_time - start_time,
        "dense_scores": dense_scores,
        "sparse_scores": sparse_scores.tolist() if len(sparse_scores) else [],
        "response": response.content
    }
