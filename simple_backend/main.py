from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import sqlite3
import json
import uuid
from datetime import datetime
import numpy as np
import os
from dotenv import load_dotenv
import httpx

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI setup - using direct httpx calls to avoid compatibility issues
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Simple SQLite database
def init_db():
    conn = sqlite3.connect('notes.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

class Note(BaseModel):
    id: Optional[str] = None
    title: str
    content: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

async def get_embedding(text: str) -> List[float]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "text-embedding-3-small",
                    "input": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    a_arr = np.array(a)
    b_arr = np.array(b)
    return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))

@app.post("/notes")
async def create_note(note: Note):
    note_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    # Get embedding for the full content
    full_text = f"{note.title} {note.content}"
    embedding = await get_embedding(full_text)
    
    conn = sqlite3.connect('notes.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO notes (id, title, content, embedding, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (note_id, note.title, note.content, json.dumps(embedding), now, now))
    conn.commit()
    conn.close()
    
    return {"id": note_id, "title": note.title, "content": note.content, "created_at": now}

@app.get("/notes")
async def get_notes():
    conn = sqlite3.connect('notes.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, content, created_at, updated_at FROM notes ORDER BY updated_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    notes = []
    for row in rows:
        notes.append({
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "created_at": row[3],
            "updated_at": row[4]
        })
    return notes

@app.get("/notes/{note_id}")
async def get_note(note_id: str):
    conn = sqlite3.connect('notes.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, content, created_at, updated_at FROM notes WHERE id = ?', (note_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")
    
    return {
        "id": row[0],
        "title": row[1],
        "content": row[2],
        "created_at": row[3],
        "updated_at": row[4]
    }

@app.put("/notes/{note_id}")
async def update_note(note_id: str, note: Note):
    now = datetime.now().isoformat()
    
    # Get new embedding
    full_text = f"{note.title} {note.content}"
    embedding = await get_embedding(full_text)
    
    conn = sqlite3.connect('notes.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE notes SET title = ?, content = ?, embedding = ?, updated_at = ?
        WHERE id = ?
    ''', (note.title, note.content, json.dumps(embedding), now, note_id))
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Note not found")
    
    conn.commit()
    conn.close()
    
    return {"id": note_id, "title": note.title, "content": note.content, "updated_at": now}

@app.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    conn = sqlite3.connect('notes.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Note not found")
    
    conn.commit()
    conn.close()
    
    return {"message": "Note deleted"}

@app.post("/search")
async def search_notes(search_query: SearchQuery):
    # Get embedding for search query
    query_embedding = await get_embedding(search_query.query)
    if not query_embedding:
        return []
    
    conn = sqlite3.connect('notes.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, content, embedding FROM notes')
    rows = cursor.fetchall()
    conn.close()
    
    # Calculate similarities
    results = []
    for row in rows:
        note_embedding = json.loads(row[3]) if row[3] else []
        if note_embedding:
            similarity = cosine_similarity(query_embedding, note_embedding)
            results.append({
                "id": row[0],
                "title": row[1],
                "content": row[2],
                "similarity": similarity
            })
    
    # Sort by similarity and return top results
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:search_query.limit]

@app.post("/chat")
async def chat_with_notes(search_query: SearchQuery):
    """RAG-powered chat with your notes"""
    # First, search for relevant notes
    relevant_notes = await search_notes(SearchQuery(query=search_query.query, limit=3))
    
    # Build context from relevant notes
    context = "Here are some relevant notes from your knowledge base:\n\n"
    for note in relevant_notes:
        if note['similarity'] > 0.1:  # Only include reasonably similar notes
            context += f"**{note['title']}**\n{note['content']}\n\n"
    
    # Create chat completion
    messages = [
        {"role": "system", "content": f"You are a helpful assistant that can answer questions based on the user's notes. Use the following context to inform your responses:\n\n{context}"},
        {"role": "user", "content": search_query.query}
    ]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            data = response.json()
        
            return {
                "response": data["choices"][0]["message"]["content"],
                "relevant_notes": relevant_notes[:3]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)