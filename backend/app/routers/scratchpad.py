from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, UUID4
from ..database import get_db
from ..models.models import ScratchpadEntry
from ..services.router_service import route_scratchpad

router = APIRouter(prefix="/scratchpad", tags=["scratchpad"])

class ScratchpadCreate(BaseModel):
    content_json: Dict[Any, Any]

class ScratchpadResponse(BaseModel):
    id: UUID4
    content_json: Dict[Any, Any]
    content_text: str
    state: str
    created_at: str

    class Config:
        from_attributes = True

class RoutingResponse(BaseModel):
    best_channel_id: Optional[str] = None
    confidence: Optional[float] = None
    suggest_new: bool

def extract_text_from_editorjs(content_json: Dict[Any, Any]) -> str:
    """Extract plain text from Editor.js JSON."""
    blocks = content_json.get("blocks", [])
    text_parts = []
    
    for block in blocks:
        block_type = block.get("type", "")
        data = block.get("data", {})
        
        if block_type == "paragraph":
            text_parts.append(data.get("text", ""))
        elif block_type == "header":
            text_parts.append(data.get("text", ""))
        elif block_type == "list":
            items = data.get("items", [])
            text_parts.extend(items)
        # Add more block types as needed
    
    return "\n".join(text_parts)

@router.post("", response_model=ScratchpadResponse)
async def create_scratchpad(scratchpad: ScratchpadCreate, db: Session = Depends(get_db)):
    """Create a new scratchpad entry."""
    content_text = extract_text_from_editorjs(scratchpad.content_json)
    
    entry = ScratchpadEntry(
        content_json=scratchpad.content_json,
        content_text=content_text,
        state='staged'
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    
    return ScratchpadResponse(
        id=entry.id,
        content_json=entry.content_json,
        content_text=entry.content_text,
        state=entry.state,
        created_at=entry.created_at.isoformat()
    )

@router.get("/staged", response_model=List[ScratchpadResponse])
async def get_staged_scratchpads(db: Session = Depends(get_db)):
    """Get all staged scratchpad entries."""
    entries = db.query(ScratchpadEntry).filter(ScratchpadEntry.state == 'staged').all()
    
    return [
        ScratchpadResponse(
            id=entry.id,
            content_json=entry.content_json,
            content_text=entry.content_text,
            state=entry.state,
            created_at=entry.created_at.isoformat()
        )
        for entry in entries
    ]

@router.post("/{entry_id}/route", response_model=RoutingResponse)
async def route_entry(entry_id: UUID4, db: Session = Depends(get_db)):
    """Get routing suggestion for scratchpad entry."""
    entry = db.query(ScratchpadEntry).filter(ScratchpadEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Scratchpad entry not found")
    
    if entry.state != 'staged':
        raise HTTPException(status_code=400, detail="Entry is not staged")
    
    try:
        result = await route_scratchpad(db, str(entry_id))
        return RoutingResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 