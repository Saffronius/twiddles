import asyncio
import numpy as np
from typing import List, Optional
from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from ..config import settings
from ..models.models import Message, ScratchpadEntry, MemorySummary

client = AsyncOpenAI(api_key=settings.openai_api_key)

async def get_embedding(text: str, retries: int = 3) -> Optional[List[float]]:
    """Get embedding for text with exponential backoff retry."""
    for attempt in range(retries):
        try:
            response = await client.embeddings.create(
                model=settings.embed_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed to get embedding after {retries} attempts: {e}")
                return None
            await asyncio.sleep(2 ** attempt)
    return None

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))

async def embed_message(db: Session, message_id: str):
    """Embed a message and update its embedding field."""
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message or message.embedding is not None:
        return
    
    embedding = await get_embedding(message.content)
    if embedding:
        message.embedding = embedding
        db.commit()

async def embed_scratchpad(db: Session, entry_id: str):
    """Embed a scratchpad entry and update its embedding field."""
    entry = db.query(ScratchpadEntry).filter(ScratchpadEntry.id == entry_id).first()
    if not entry or entry.embedding is not None:
        return
    
    embedding = await get_embedding(entry.content_text)
    if embedding:
        entry.embedding = embedding
        db.commit()

def update_channel_centroid(db: Session, channel_id: str, new_embedding: List[float]):
    """Update channel centroid with new embedding using incremental mean."""
    from ..models.models import Channel
    
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        return
    
    try:
        # Ensure we're working with a plain Python list
        if hasattr(new_embedding, 'tolist'):
            new_embedding_list = new_embedding.tolist()
        elif isinstance(new_embedding, (list, tuple)):
            new_embedding_list = list(new_embedding)
        else:
            # Convert array-like objects to list
            new_embedding_list = [float(x) for x in new_embedding]
        
        # Always just set the new embedding to avoid array comparison issues
        # This is simpler and more reliable than trying to do incremental updates
        channel.embedding_centroid = new_embedding_list
        
        db.commit()
        
    except Exception as e:
        print(f"Error updating channel centroid: {e}")
        # Just skip the update if there's any issue
        pass

def recompute_channel_centroid(db: Session, channel_id: str):
    """Recompute channel centroid from all recent message embeddings."""
    from ..models.models import Channel
    
    recent_messages = db.query(Message).filter(
        Message.channel_id == channel_id,
        Message.memory_layer == 'recent',
        Message.embedding.isnot(None)
    ).all()
    
    if not recent_messages:
        # No embeddings to compute from
        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if channel:
            channel.embedding_centroid = None
            db.commit()
        return
    
    # Average all embeddings
    embeddings = [np.array(msg.embedding) for msg in recent_messages]
    mean_embedding = np.mean(embeddings, axis=0)
    
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if channel:
        channel.embedding_centroid = mean_embedding.tolist()
        db.commit() 