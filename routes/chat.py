from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
import chat_utils, models
from database import get_db
import time, numpy as np, os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import text

router = APIRouter(prefix="/api")
store = {}

@router.post("/chat/")
async def chat(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    prompt = data.get("prompt", "")
    retrieval_method = data.get("retrieval_method", "dense")

    if not prompt:
        return {"error": "Prompt is required."}

    start_time = time.time()
    prompt_expanded = chat_utils.expand_query(prompt) if retrieval_method=="expanded" else prompt

    # --- Dense retrieval ---
    dense_chunks, dense_scores = [], []
    if retrieval_method in ["dense", "hybrid", "expanded"]:
        user_vector = chat_utils.embeddings.embed_query(prompt_expanded)
        query_vec_str = "[" + ",".join(str(x) for x in user_vector) + "]"
        dense_results = db.execute(
            text("""
                SELECT chunk_text, 1 - (embedding <=> :query_vec) AS score
                FROM chunk_embedding
                ORDER BY embedding <=> :query_vec
                LIMIT 10
            """), {"query_vec": query_vec_str}
        ).fetchall()
        dense_chunks = [r[0] for r in dense_results]
        dense_scores = [r[1] for r in dense_results]

    # --- Sparse retrieval ---
    sparse_chunks, sparse_scores = [], []
    if retrieval_method in ["sparse","hybrid","expanded"] and chat_utils.corpus:
        query_vec_sparse = chat_utils.tfidf_vectorizer.transform([prompt_expanded])
        sparse_scores_all = chat_utils.sk_cosine(chat_utils.tfidf_matrix, query_vec_sparse).flatten()
        top_idx = np.argsort(-sparse_scores_all)[:10]
        sparse_chunks = [chat_utils.corpus[i] for i in top_idx]
        sparse_scores = sparse_scores_all[top_idx]

    # --- Merge ---
    if retrieval_method=="dense":
        top_chunks = dense_chunks[:3]
    elif retrieval_method=="sparse":
        top_chunks = sparse_chunks[:3]
    else:
        combined = list(dict.fromkeys(dense_chunks[:5] + sparse_chunks[:5]))
        top_chunks = combined[:3]

    system_instruction = f"{top_chunks}\n\n{chat_utils.SYSTEM_TEXT}"

    # --- Session ---
    session_id = "user123"
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    history = store[session_id]

    # --- Chat ---
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.3)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}")
    ])
    chain = prompt_template | model
    with_history = RunnableWithMessageHistory(chain, lambda x: history, input_messages_key="message", history_messages_key="history")
    response = with_history.invoke({"message": prompt}, config={"configurable":{"session_id":session_id}})

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
