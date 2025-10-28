from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import schemas, models, chat_utils
from database import get_db

router = APIRouter(prefix="/api")

@router.get("/prompts/")
def get_files(db: Session = Depends(get_db)):
    items = db.query(models.FileData).all()
    return [{"id": i.id, "name": i.name, "description": i.description} for i in items]

@router.post("/content/", response_model=schemas.FileDataResponse)
def create_item(item: schemas.FileDataCreate, db: Session = Depends(get_db)):
    db_item = models.FileData(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)

    num_chunks = chat_utils.create_and_store_chunks(item.description, db, models.ChunkEmbedding)
    chat_utils.precompute_tfidf(db, models.ChunkEmbedding)
    print(f"Created {num_chunks} chunks for '{item.name}'")
    return db_item
