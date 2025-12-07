from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func
from .database import Base


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(String, primary_key=True, index=True)        # UUID
    filename = Column(String, nullable=False)
    original_path = Column(String, nullable=False)
    json_path = Column(String, nullable=False)
    model_used = Column(String, nullable=False)              # ollama / openai / model name
    created_at = Column(DateTime(timezone=True), server_default=func.now())
