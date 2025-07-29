from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, UUID4
from ..database import get_db
from ..models.models import Channel
from ..services.chat_service import summarize_old_messages

router = APIRouter(prefix="/channels", tags=["channels"])

class ChannelCreate(BaseModel):
    name: str
    parent_id: Optional[UUID4] = None
    system_prompt: Optional[str] = ""

class ChannelUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    inherit_prompt: Optional[bool] = None

class ChannelResponse(BaseModel):
    id: UUID4
    parent_id: Optional[UUID4]
    name: str
    description: str
    system_prompt: str
    inherit_prompt: bool
    created_at: str
    children: List['ChannelResponse'] = []

    class Config:
        from_attributes = True

def build_channel_tree(channels: List[Channel], parent_id: Optional[str] = None) -> List[ChannelResponse]:
    """Build nested channel tree structure."""
    tree = []
    for channel in channels:
        if str(channel.parent_id) == parent_id or (parent_id is None and channel.parent_id is None):
            channel_dict = {
                "id": channel.id,
                "parent_id": channel.parent_id,
                "name": channel.name,
                "description": channel.description,
                "system_prompt": channel.system_prompt,
                "inherit_prompt": channel.inherit_prompt,
                "created_at": channel.created_at.isoformat(),
                "children": build_channel_tree(channels, str(channel.id))
            }
            tree.append(ChannelResponse(**channel_dict))
    return tree

@router.post("/", response_model=ChannelResponse)
async def create_channel(channel: ChannelCreate, db: Session = Depends(get_db)):
    """Create a new channel."""
    # Check if parent exists
    if channel.parent_id:
        parent = db.query(Channel).filter(Channel.id == channel.parent_id).first()
        if not parent:
            raise HTTPException(status_code=404, detail="Parent channel not found")
    
    # Check name uniqueness within parent
    existing = db.query(Channel).filter(
        Channel.name == channel.name,
        Channel.parent_id == channel.parent_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Channel name must be unique within parent")
    
    db_channel = Channel(
        name=channel.name,
        parent_id=channel.parent_id,
        system_prompt=channel.system_prompt or "",
        description_json={"blocks": []}
    )
    db.add(db_channel)
    db.commit()
    db.refresh(db_channel)
    
    return ChannelResponse(
        id=db_channel.id,
        parent_id=db_channel.parent_id,
        name=db_channel.name,
        description=db_channel.description,
        system_prompt=db_channel.system_prompt,
        inherit_prompt=db_channel.inherit_prompt,
        created_at=db_channel.created_at.isoformat(),
        children=[]
    )

@router.get("/tree", response_model=List[ChannelResponse])
async def get_channel_tree(db: Session = Depends(get_db)):
    """Get complete channel tree."""
    channels = db.query(Channel).all()
    return build_channel_tree(channels)

@router.get("/{channel_id}", response_model=ChannelResponse)
async def get_channel(channel_id: UUID4, db: Session = Depends(get_db)):
    """Get single channel by ID."""
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    return ChannelResponse(
        id=channel.id,
        parent_id=channel.parent_id,
        name=channel.name,
        description=channel.description,
        system_prompt=channel.system_prompt,
        inherit_prompt=channel.inherit_prompt,
        created_at=channel.created_at.isoformat(),
        children=[]
    )

@router.patch("/{channel_id}", response_model=ChannelResponse)
async def update_channel(channel_id: UUID4, update: ChannelUpdate, db: Session = Depends(get_db)):
    """Update channel."""
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    if update.name is not None:
        # Check name uniqueness
        existing = db.query(Channel).filter(
            Channel.name == update.name,
            Channel.parent_id == channel.parent_id,
            Channel.id != channel_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Channel name must be unique within parent")
        channel.name = update.name
    
    if update.system_prompt is not None:
        channel.system_prompt = update.system_prompt
    
    if update.inherit_prompt is not None:
        channel.inherit_prompt = update.inherit_prompt
    
    db.commit()
    db.refresh(channel)
    
    return ChannelResponse(
        id=channel.id,
        parent_id=channel.parent_id,
        name=channel.name,
        description=channel.description,
        system_prompt=channel.system_prompt,
        inherit_prompt=channel.inherit_prompt,
        created_at=channel.created_at.isoformat(),
        children=[]
    )

@router.post("/{channel_id}/summarize_old")
async def summarize_channel_messages(
    channel_id: UUID4, 
    retain: int = 50, 
    db: Session = Depends(get_db)
):
    """Summarize old messages in channel."""
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    summaries_created = await summarize_old_messages(db, str(channel_id), retain)
    
    return {
        "channel_id": str(channel_id),
        "summaries_created": summaries_created,
        "retained_recent": retain
    } 