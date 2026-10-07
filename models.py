import uuid
from datetime import datetime
from sqlalchemy import Column, String, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Kapcsolat a meccsekhez / elemzésekhez
    matches = relationship("Match", back_populates="owner")

class Match(Base):
    __tablename__ = "matches"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False, default="Névtelen meccs")

    # MULTI-TENANCY KULCS: Melyik edzőé ez a meccs?
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    # A kliens (MatchAnalysisDto) által küldött adatok tárolására:
    video_url = Column(String, nullable=True)
    analysis_data = Column(String, nullable=True)  # JSON-ként szerializált TacticalFrameDto lista
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="matches")