import json
from typing import Dict, Any
from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from ..models.models import Channel
from ..config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)

async def generate_summary(content: str) -> str:
    """Generate 25-word summary of content."""
    prompt = f"""Summarize the NEW content in <=25 words, factual, no intro words.
New Content:
\"\"\"
{content}
\"\"\"
Output only the sentence."""
    
    try:
        response = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to generate summary: {e}")
        return "New content added"

async def update_description_with_bullet(db: Session, channel_id: str, scratchpad_id: str, content: str):
    """Add bullet point to channel description with RID."""
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        return
    
    # Generate summary
    summary = await generate_summary(content)
    bullet_text = f"- [RID:{scratchpad_id}] {summary}"
    
    # Update description_json (Editor.js format)
    description_json = channel.description_json or {"blocks": []}
    
    # Add new paragraph block
    new_block = {
        "type": "paragraph",
        "data": {"text": bullet_text}
    }
    description_json["blocks"].append(new_block)
    
    # Update plain text description
    blocks = description_json.get("blocks", [])
    text_parts = []
    for block in blocks:
        if block.get("type") == "paragraph":
            text_parts.append(block.get("data", {}).get("text", ""))
    
    channel.description_json = description_json
    channel.description = "\n".join(text_parts)
    db.commit()

def remove_description_bullet(db: Session, channel_id: str, scratchpad_id: str):
    """Remove bullet point with specific RID from channel description."""
    channel = db.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        return
    
    rid_pattern = f"[RID:{scratchpad_id}]"
    description_json = channel.description_json or {"blocks": []}
    
    # Remove blocks containing the RID
    updated_blocks = []
    for block in description_json.get("blocks", []):
        if block.get("type") == "paragraph":
            text = block.get("data", {}).get("text", "")
            if rid_pattern not in text:
                updated_blocks.append(block)
        else:
            updated_blocks.append(block)
    
    description_json["blocks"] = updated_blocks
    
    # Update plain text description
    text_parts = []
    for block in updated_blocks:
        if block.get("type") == "paragraph":
            text_parts.append(block.get("data", {}).get("text", ""))
    
    channel.description_json = description_json
    channel.description = "\n".join(text_parts)
    db.commit() 