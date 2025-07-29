from typing import Optional, Tuple, List
from sqlalchemy.orm import Session
from ..models.models import Channel, ScratchpadEntry, RoutingLog, Message
from ..config import settings
from .embedding_service import get_embedding, cosine_similarity

async def route_scratchpad(db: Session, entry_id: str) -> dict:
    """
    Route scratchpad entry to best matching channel.
    Returns: {best_channel_id?, confidence?, suggest_new: bool}
    """
    entry = db.query(ScratchpadEntry).filter(ScratchpadEntry.id == entry_id).first()
    if not entry:
        raise ValueError("Scratchpad entry not found")
    
    # Get or compute embedding for scratchpad
    if not entry.embedding:
        embedding = await get_embedding(entry.content_text)
        if embedding:
            entry.embedding = embedding
            db.commit()
        else:
            # If embedding fails, suggest new channel
            return {"suggest_new": True}
    
    scratchpad_embedding = entry.embedding
    
    # Get all channels with embeddings
    channels = db.query(Channel).filter(Channel.embedding_centroid.isnot(None)).all()
    
    if not channels:
        return {"suggest_new": True}
    
    # Calculate scores
    best_score = -float('inf')
    best_channel = None
    
    for channel in channels:
        score = cosine_similarity(scratchpad_embedding, channel.embedding_centroid)
        if score > best_score:
            best_score = score
            best_channel = channel
    
    # Check threshold
    if best_score >= settings.router_threshold:
        return {
            "best_channel_id": str(best_channel.id),
            "confidence": best_score,
            "suggest_new": False
        }
    else:
        return {"suggest_new": True}

async def accept_routing(db: Session, entry_id: str, channel_id: str) -> dict:
    """
    Accept routing suggestion - create message and update channel.
    """
    entry = db.query(ScratchpadEntry).filter(ScratchpadEntry.id == entry_id).first()
    if not entry or entry.state != 'staged':
        raise ValueError("Invalid scratchpad entry")
    
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        raise ValueError("Channel not found")
    
    # Create message
    message = Message(
        channel_id=channel_id,
        role='user',
        content=entry.content_text,
        scratchpad_origin_id=entry.id
    )
    db.add(message)
    db.flush()  # Get message ID
    
    # Update scratchpad state
    entry.state = 'routed'
    
    # Create routing log
    routing_log = RoutingLog(
        entry_id=entry.id,
        target_channel_id=channel_id,
        routed_by='user'
    )
    db.add(routing_log)
    
    # Update channel centroid if embedding exists
    if entry.embedding:
        from .embedding_service import update_channel_centroid
        update_channel_centroid(db, channel_id, entry.embedding)
    
    db.commit()
    
    # Trigger description update
    from .description_service import update_description_with_bullet
    await update_description_with_bullet(db, channel_id, entry.id, entry.content_text)
    
    return {
        "message_id": str(message.id),
        "channel_id": channel_id,
        "description_updated": True
    }

async def create_and_route(db: Session, entry_id: str, channel_name: str, seed_description: str = "") -> dict:
    """
    Create new channel and route scratchpad to it.
    """
    entry = db.query(ScratchpadEntry).filter(ScratchpadEntry.id == entry_id).first()
    if not entry or entry.state != 'staged':
        raise ValueError("Invalid scratchpad entry")
    
    # Create new channel
    channel = Channel(
        name=channel_name,
        description=seed_description,
        description_json={"blocks": [{"type": "paragraph", "data": {"text": seed_description}}] if seed_description else []}
    )
    db.add(channel)
    db.flush()  # Get channel ID
    
    # Route to new channel
    result = await accept_routing(db, entry_id, str(channel.id))
    return result

def undo_routing(db: Session, entry_id: str) -> bool:
    """
    Undo routing within 5 minutes.
    """
    from datetime import datetime, timedelta
    
    entry = db.query(ScratchpadEntry).filter(ScratchpadEntry.id == entry_id).first()
    if not entry or entry.state != 'routed':
        return False
    
    # Check if within 5 minutes
    if datetime.utcnow() - entry.created_at > timedelta(minutes=5):
        return False
    
    # Find the created message
    message = db.query(Message).filter(Message.scratchpad_origin_id == entry.id).first()
    if not message:
        return False
    
    channel_id = message.channel_id
    
    # Delete message
    db.delete(message)
    
    # Revert scratchpad state
    entry.state = 'staged'
    
    # Remove description bullet
    from .description_service import remove_description_bullet
    remove_description_bullet(db, channel_id, entry.id)
    
    # Recompute channel centroid
    from .embedding_service import recompute_channel_centroid
    recompute_channel_centroid(db, str(channel_id))
    
    db.commit()
    return True 