from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
from openai import AsyncOpenAI
from ..models.models import Channel, Message, MemorySummary, ContextSlice
from ..config import settings
from .embedding_service import get_embedding, cosine_similarity

client = AsyncOpenAI(api_key=settings.openai_api_key)

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token)."""
    return len(text) // 4

def build_system_prompt(db: Session, channel_id: str) -> str:
    """Build system prompt with inheritance."""
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        return ""
    
    prompts = []
    current = channel
    
    # Collect prompts up the tree
    while current:
        if current.system_prompt:
            prompts.append(current.system_prompt)
        
        if not current.inherit_prompt:
            break
            
        current = current.parent
    
    # Reverse to have root prompts first
    prompts.reverse()
    return "\n\n".join(prompts)

async def build_context_plain(db: Session, channel_id: str, user_input: str) -> Tuple[List[Dict], List[str], List[str]]:
    """Build plain context with last 10 recent messages."""
    # Get last 10 recent messages
    messages = db.query(Message).filter(
        Message.channel_id == channel_id,
        Message.memory_layer == 'recent'
    ).order_by(desc(Message.created_at)).limit(10).all()
    
    messages.reverse()  # Oldest first
    
    # Build context
    context = []
    total_tokens = 0
    user_input_tokens = estimate_tokens(user_input)
    budget = settings.token_budget - user_input_tokens - 500  # Reserve space
    
    # Add system prompt
    system_prompt = build_system_prompt(db, channel_id)
    if system_prompt:
        context.append({"role": "system", "content": system_prompt})
        total_tokens += estimate_tokens(system_prompt)
    
    # Add messages
    included_message_ids = []
    for message in messages:
        msg_tokens = estimate_tokens(message.content)
        if total_tokens + msg_tokens > budget and context:
            break
        
        context.append({
            "role": message.role,
            "content": message.content
        })
        included_message_ids.append(str(message.id))
        total_tokens += msg_tokens
    
    # If still too large, replace oldest half with channel description
    if total_tokens > budget and len(context) > 2:
        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if channel and channel.description:
            # Keep system prompt and newest half
            half_point = len(context) // 2
            newer_messages = context[half_point:]
            
            # Replace with description
            desc_context = [{"role": "system", "content": f"Channel context: {channel.description}"}]
            context = desc_context + newer_messages
    
    return context, included_message_ids, []

async def build_context_rag(db: Session, channel_id: str, user_input: str) -> Tuple[List[Dict], List[str], List[str]]:
    """Build RAG context with recent messages + top summaries."""
    # Start with plain context
    context, message_ids, _ = await build_context_plain(db, channel_id, user_input)
    
    # Get user input embedding
    user_embedding = await get_embedding(user_input)
    if not user_embedding:
        return context, message_ids, []
    
    # Get all summaries for this channel
    summaries = db.query(MemorySummary).filter(
        MemorySummary.channel_id == channel_id
    ).all()
    
    if not summaries:
        return context, message_ids, []
    
    # Score summaries (need to embed them first if not done)
    scored_summaries = []
    for summary in summaries:
        # For MVP, just take top 3 by recency if no embedding
        scored_summaries.append((summary, 1.0))
    
    # Sort by score and take top 3
    scored_summaries.sort(key=lambda x: x[1], reverse=True)
    top_summaries = scored_summaries[:3]
    
    # Add summary content
    if top_summaries:
        summary_text = "\n\n".join([f"Summary: {s[0].content}" for s in top_summaries])
        context.append({
            "role": "system", 
            "content": f"[Previous Context Summaries]\n{summary_text}"
        })
    
    summary_ids = [str(s[0].id) for s in top_summaries]
    return context, message_ids, summary_ids

async def chat_completion(db: Session, channel_id: str, user_content: str, use_rag: bool = False) -> Dict:
    """Handle complete chat interaction."""
    # Create user message
    user_message = Message(
        channel_id=channel_id,
        role='user',
        content=user_content
    )
    db.add(user_message)
    db.flush()
    
    # Build context
    if use_rag:
        context, message_ids, summary_ids = await build_context_rag(db, channel_id, user_content)
        mode = 'rag'
    else:
        context, message_ids, summary_ids = await build_context_plain(db, channel_id, user_content)
        mode = 'plain'
    
    # Add user message to context
    context.append({"role": "user", "content": user_content})
    
    # Create context slice record
    context_slice = ContextSlice(
        channel_id=channel_id,
        user_message_id=user_message.id,
        included_message_ids=message_ids,
        included_summary_ids=summary_ids,
        mode=mode,
        token_estimate=sum(estimate_tokens(msg["content"]) for msg in context)
    )
    db.add(context_slice)
    
    # Call OpenAI
    try:
        response = await client.chat.completions.create(
            model=settings.chat_model,
            messages=context,
            max_tokens=1000,
            temperature=0.7
        )
        
        assistant_content = response.choices[0].message.content
        
        # Create assistant message
        assistant_message = Message(
            channel_id=channel_id,
            role='assistant',
            content=assistant_content
        )
        db.add(assistant_message)
        db.commit()
        
        # Trigger background embedding
        # In a real app, this would be a background task
        # For MVP, we'll do it synchronously
        try:
            from .embedding_service import embed_message
            await embed_message(db, str(user_message.id))
            await embed_message(db, str(assistant_message.id))
        except Exception as e:
            print(f"Background embedding failed: {e}")
        
        return {
            "user_message_id": str(user_message.id),
            "assistant_message_id": str(assistant_message.id),
            "content": assistant_content
        }
        
    except Exception as e:
        db.rollback()
        raise Exception(f"Chat completion failed: {e}")

async def summarize_old_messages(db: Session, channel_id: str, retain_count: int = 50) -> int:
    """Summarize old messages in channel, keeping newest retain_count as recent."""
    # Get messages to compress (all except newest retain_count)
    total_messages = db.query(Message).filter(
        Message.channel_id == channel_id,
        Message.memory_layer == 'recent'
    ).count()
    
    if total_messages <= retain_count:
        return 0  # Nothing to compress
    
    # Get old messages
    old_messages = db.query(Message).filter(
        Message.channel_id == channel_id,
        Message.memory_layer == 'recent'
    ).order_by(Message.created_at).limit(total_messages - retain_count).all()
    
    if not old_messages:
        return 0
    
    # Group into chunks (~2000-4000 tokens)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for message in old_messages:
        msg_tokens = estimate_tokens(message.content)
        if current_tokens + msg_tokens > 4000 and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [message]
            current_tokens = msg_tokens
        else:
            current_chunk.append(message)
            current_tokens += msg_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Summarize each chunk
    summaries_created = 0
    for chunk in chunks:
        # Build content for summarization
        content_parts = []
        for msg in chunk:
            content_parts.append(f"{msg.role}: {msg.content}")
        
        chunk_content = "\n".join(content_parts)
        
        # Generate summary
        prompt = f"""Summarize these chat messages preserving key facts, decisions, open questions. <=300 tokens.
MESSAGES:
{chunk_content}"""
        
        try:
            response = await client.chat.completions.create(
                model=settings.chat_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            summary_content = response.choices[0].message.content
            
            # Create summary record
            summary = MemorySummary(
                channel_id=channel_id,
                source_message_ids=[str(msg.id) for msg in chunk],
                content=summary_content
            )
            db.add(summary)
            
            # Mark source messages as compressed
            for msg in chunk:
                msg.memory_layer = 'compressed'
            
            summaries_created += 1
            
        except Exception as e:
            print(f"Failed to summarize chunk: {e}")
            continue
    
    db.commit()
    return summaries_created 