import os, numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load system instruction
with open("chat.txt", "r", encoding="utf-8") as f:
    SYSTEM_TEXT = f.read().strip()

# TF-IDF globals
tfidf_vectorizer = None
tfidf_matrix = None
corpus = []

def precompute_tfidf(db, ChunkEmbedding):
    global tfidf_vectorizer, tfidf_matrix, corpus
    items = db.query(ChunkEmbedding.chunk_text).all()
    corpus = [i[0] for i in items]
    if corpus:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

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

def create_and_store_chunks(description: str, db, ChunkEmbedding):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
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
