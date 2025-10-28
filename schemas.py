from pydantic import BaseModel

class FileDataCreate(BaseModel):
    name: str
    description: str

class FileDataResponse(BaseModel):
    id: int
    name: str
    description: str
    class Config:
        from_attributes = True
