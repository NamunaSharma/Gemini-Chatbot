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
