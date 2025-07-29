from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, UUID4
from ..database import get_db
from ..models.models import KnowledgeCard, Channel

router = APIRouter(prefix="/knowledge_cards", tags=["knowledge_cards"])

class KnowledgeCardCreate(BaseModel):
    channel_id: UUID4
    title: str
    body: str
    source_message_ids: List[UUID4]

class KnowledgeCardResponse(BaseModel):
    id: UUID4
    channel_id: UUID4
    title: str
    body: str
    source_message_ids: List[UUID4]
    created_at: str

    class Config:
        from_attributes = True

@router.post("/", response_model=KnowledgeCardResponse)
async def create_knowledge_card(card: KnowledgeCardCreate, db: Session = Depends(get_db)):
    """Create a new knowledge card."""
    # Verify channel exists
    channel = db.query(Channel).filter(Channel.id == card.channel_id).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    db_card = KnowledgeCard(
        channel_id=card.channel_id,
        title=card.title,
        body=card.body,
        source_message_ids=[str(id) for id in card.source_message_ids]
    )
    db.add(db_card)
    db.commit()
    db.refresh(db_card)
    
    return KnowledgeCardResponse(
        id=db_card.id,
        channel_id=db_card.channel_id,
        title=db_card.title,
        body=db_card.body,
        source_message_ids=[UUID4(id) for id in db_card.source_message_ids],
        created_at=db_card.created_at.isoformat()
    )

@router.get("/", response_model=List[KnowledgeCardResponse])
async def get_knowledge_cards(
    channel_id: Optional[UUID4] = Query(None),
    db: Session = Depends(get_db)
):
    """Get knowledge cards, optionally filtered by channel."""
    query = db.query(KnowledgeCard)
    
    if channel_id:
        query = query.filter(KnowledgeCard.channel_id == channel_id)
    
    cards = query.all()
    
    return [
        KnowledgeCardResponse(
            id=card.id,
            channel_id=card.channel_id,
            title=card.title,
            body=card.body,
            source_message_ids=[UUID4(id) for id in card.source_message_ids],
            created_at=card.created_at.isoformat()
        )
        for card in cards
    ] 