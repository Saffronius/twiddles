from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel, UUID4
from ..database import get_db
from ..models.models import Channel, Message
from ..services.chat_service import chat_completion

router = APIRouter(prefix="/channels", tags=["messages"])

class MessageCreate(BaseModel):
    content: str
    rag: Optional[bool] = False

class MessageResponse(BaseModel):
    id: UUID4
    channel_id: UUID4
    role: str
    content: str
    created_at: str
    memory_layer: str

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    user_message_id: str
    assistant_message_id: str
    content: str

@router.get("/{channel_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    channel_id: UUID4, 
    after: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get messages for a channel, optionally after a timestamp."""
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    query = db.query(Message).filter(Message.channel_id == channel_id)
    
    if after:
        try:
            after_dt = datetime.fromisoformat(after.replace('Z', '+00:00'))
            query = query.filter(Message.created_at > after_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
    
    messages = query.order_by(desc(Message.created_at)).limit(100).all()
    messages.reverse()  # Oldest first
    
    return [
        MessageResponse(
            id=msg.id,
            channel_id=msg.channel_id,
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at.isoformat(),
            memory_layer=msg.memory_layer
        )
        for msg in messages
    ]

@router.post("/{channel_id}/messages", response_model=ChatResponse)
async def send_message(
    channel_id: UUID4, 
    message: MessageCreate, 
    db: Session = Depends(get_db)
):
    """Send a message to a channel and get AI response."""
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    try:
        result = await chat_completion(db, str(channel_id), message.content, message.rag)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 