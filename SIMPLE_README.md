# Simple Notes with RAG

A clean, minimal note-taking app with AI-powered search and chat functionality.

## What This Is

- **Simple Notes**: Like Apple Notes but with AI superpowers
- **Clean Interface**: Beautiful, minimal design with proper typography
- **Smart Search**: RAG (Retrieval Augmented Generation) for intelligent note search
- **AI Chat**: Ask questions about your notes and get contextual answers

## Features

✅ **Clean UI**: Minimal, Apple Notes-inspired interface  
✅ **RAG Search**: Semantic search through all your notes  
✅ **AI Chat**: Chat with your notes using GPT-4  
✅ **Auto-save**: Changes saved automatically  
✅ **Fast**: SQLite backend, no complex setup needed  

## Quick Start

### 1. Backend Setup
```bash
cd simple_backend

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=your_actual_key_here" > .env

# Run the server
python main.py
```

### 2. Frontend Setup  
```bash
cd simple_frontend

# Install dependencies
npm install

# Run the frontend
npm run dev
```

### 3. Use the App
- Open http://localhost:5173
- Create your first note
- Use "Chat with Notes" for RAG functionality

## What's Different

**Instead of the complex original system:**
- ❌ Complex channel hierarchies
- ❌ Complicated routing algorithms  
- ❌ Over-engineered memory management
- ❌ Multiple database tables
- ❌ Confusing UI with bad fonts

**You get:**
- ✅ Simple notes (title + content)
- ✅ One search bar
- ✅ Clean, readable Inter font
- ✅ RAG chat in a modal
- ✅ SQLite database (no Docker needed)
- ✅ Minimal, fast, beautiful

## API Endpoints

- `GET /notes` - Get all notes
- `POST /notes` - Create note
- `PUT /notes/{id}` - Update note  
- `DELETE /notes/{id}` - Delete note
- `POST /search` - Semantic search
- `POST /chat` - RAG chat

## Tech Stack

**Backend**: FastAPI + SQLite + OpenAI  
**Frontend**: React + TypeScript + Inter font  
**No Docker, no Postgres, no complexity**

This is what the original should have been - simple, clean, and focused.