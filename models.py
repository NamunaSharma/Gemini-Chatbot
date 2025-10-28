from sqlalchemy import Column, Integer, String, Text
from database import Base

class FileData(Base):
    __tablename__ = "file_data"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)

class ChunkEmbedding(Base):
    __tablename__ = "chunk_embedding"
    id = Column(Integer, primary_key=True, index=True)
    chunk_text = Column(Text)
    embedding = Column(Text)  # pgvector
