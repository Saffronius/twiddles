from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, UUID4
from ..database import get_db
from ..models.models import ScratchpadEntry
from ..services.router_service import accept_routing, create_and_route, undo_routing

router = APIRouter(prefix="/routing", tags=["routing"])

class AcceptRouting(BaseModel):
    channel_id: UUID4

class CreateChannel(BaseModel):
    name: str
    seed_description: Optional[str] = ""

class RoutingResult(BaseModel):
    message_id: str
    channel_id: str
    description_updated: bool

@router.post("/{entry_id}/accept", response_model=RoutingResult)
async def accept_route(
    entry_id: UUID4, 
    accept: AcceptRouting, 
    db: Session = Depends(get_db)
):
    """Accept routing suggestion and create message."""
    entry = db.query(ScratchpadEntry).filter(ScratchpadEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Scratchpad entry not found")
    
    if entry.state != 'staged':
        raise HTTPException(status_code=400, detail="Entry is not staged")
    
    try:
        result = await accept_routing(db, str(entry_id), str(accept.channel_id))
        return RoutingResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{entry_id}/create_channel", response_model=RoutingResult)
async def create_channel_and_route(
    entry_id: UUID4, 
    channel_data: CreateChannel, 
    db: Session = Depends(get_db)
):
    """Create new channel and route scratchpad to it."""
    entry = db.query(ScratchpadEntry).filter(ScratchpadEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Scratchpad entry not found")
    
    if entry.state != 'staged':
        raise HTTPException(status_code=400, detail="Entry is not staged")
    
    try:
        result = await create_and_route(
            db, 
            str(entry_id), 
            channel_data.name, 
            channel_data.seed_description
        )
        return RoutingResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{entry_id}/undo")
async def undo_route(entry_id: UUID4, db: Session = Depends(get_db)):
    """Undo routing within 5 minutes."""
    entry = db.query(ScratchpadEntry).filter(ScratchpadEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Scratchpad entry not found")
    
    success = undo_routing(db, str(entry_id))
    if not success:
        raise HTTPException(
            status_code=400, 
            detail="Cannot undo: entry not routed or timeout exceeded"
        )
    
    return {"success": True, "message": "Routing undone successfully"} 