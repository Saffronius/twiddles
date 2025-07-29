# Channel Second Brain MVP

![Channel Second Brain](https://img.shields.io/badge/Status-MVP-blue) ![Python](https://img.shields.io/badge/Python-3.11+-green) ![React](https://img.shields.io/badge/React-18+-blue) ![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue)

A sophisticated AI-powered knowledge management system that organizes your thoughts into a multi-level channel tree with intelligent routing, living descriptions, and advanced memory management.

## 🌟 What is Channel Second Brain?

Channel Second Brain is an innovative AI-powered application that helps you capture, organize, and interact with your ideas seamlessly. Think of it as a combination of:

- **Notion** (for organization)
- **Obsidian** (for knowledge graphs) 
- **ChatGPT** (for AI interaction)
- **Roam Research** (for connected thinking)

### Key Innovation: "Spitball" Smart Routing

The core feature is the **Spitball** system - you quickly capture thoughts in a rich text editor, and AI automatically suggests which channel (topic/project) they belong to, or creates new channels automatically. No more manual filing!

## ✨ Features

### 🗂️ Smart Organization
- **Multi-level Channel Tree**: Organize topics in unlimited hierarchical depth
- **AI-Powered Routing**: Spitball ideas get automatically routed to the right channel
- **Living Descriptions**: Channel descriptions update automatically as you add content
- **System Prompt Inheritance**: Child channels inherit and build upon parent prompts

### 🧠 Intelligent Memory
- **RAG (Retrieval Augmented Generation)**: AI remembers relevant context from your entire knowledge base
- **Memory Compression**: Old conversations get summarized to save tokens while preserving key information
- **Context Slicing**: Full audit trail of what information was used for each AI response

### 💬 Advanced Chat Interface
- **Plain Mode**: Simple chat with recent message history
- **RAG Mode**: Enhanced chat that pulls in relevant summaries from across all channels
- **Undo System**: 5-minute window to undo routing decisions
- **Knowledge Cards**: Manually distill key insights from conversations

### 🔧 Developer-Friendly
- **RESTful API**: Full programmatic access to all features
- **Modern Stack**: FastAPI + React + TypeScript + PostgreSQL + pgvector
- **Vector Search**: Semantic similarity for intelligent content matching

## 🎯 Perfect For

- **Researchers** organizing literature and insights
- **Students** managing coursework and notes
- **Writers** tracking story ideas and character development
- **Product Managers** organizing feature discussions
- **Consultants** managing client projects and knowledge
- **Anyone** who thinks in interconnected ideas

## 🚀 Quick Start

### Prerequisites

Make sure you have these installed:
- **Python 3.11+** ([Download](https://python.org/downloads/))
- **Node.js 18+** ([Download](https://nodejs.org/))
- **Docker & Docker Compose** ([Download](https://docker.com/get-started/))
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))

### 1. Clone & Setup

```bash
# Clone the repository
git clone <your-repository-url>
cd channel-second-brain

# Create environment file
echo 'DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/second_brain
OPENAI_API_KEY=your_openai_api_key_here
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
ROUTER_THRESHOLD=0.45
TOKEN_BUDGET=8000' > backend/.env
```

**Important**: Replace `your_openai_api_key_here` with your actual OpenAI API key!

### 2. Start the Database

```bash
# Start PostgreSQL with pgvector extension
docker compose up -d postgres

# Wait for it to be ready (should see "database system is ready to accept connections")
docker compose logs -f postgres
```

### 3. Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn app.main:app --reload
```

✅ **Backend Ready!** API available at `http://localhost:8000` (docs at `http://localhost:8000/docs`)

### 4. Setup Frontend

Open a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

✅ **Frontend Ready!** App available at `http://localhost:5173`

### 5. Create Your First Channel

1. Open `http://localhost:5173` in your browser
2. Click "**+ New**" in the sidebar
3. Name your first channel (e.g., "Personal Projects")
4. Start chatting or use the **✎ Spitball** button!

## 📖 User Guide

### Getting Started: Your First Spitball

1. **Click the floating ✎ button** (bottom right of screen)
2. **Write your thoughts** using the rich text editor:
   - Add headers, lists, formatted text
   - Capture ideas as they come to you
3. **Click "Route to Channel"**
4. **AI suggests** where it belongs or proposes creating a new channel
5. **Accept or customize** the suggestion
6. **Your idea is now saved** and the channel description updates automatically!

### Channel Management

#### Creating Channels
- **Method 1**: Click "+ New" in sidebar for manual creation
- **Method 2**: Let Spitball create them automatically when routing ideas
- **Hierarchy**: Create nested channels for complex organization
- **System Prompts**: Set custom AI behavior for each channel

#### System Prompts & Inheritance
```
Marketing (Parent)
├── System Prompt: "You are a marketing strategist focused on B2B SaaS..."
├── Content Strategy (Child) 
│   ├── Inherits parent prompt + adds: "Focus on content marketing..."
├── Social Media (Child)
    ├── Inherits parent prompt + adds: "Focus on social platforms..."
```

### Chat Modes Explained

#### Plain Mode (Default)
- Uses last 10 messages in the channel
- Fast and simple
- Best for focused, recent conversations

#### RAG Mode (Toggle ✓ "Use RAG")
- Includes relevant summaries from across ALL your channels
- AI has access to your entire knowledge base
- Best for complex questions requiring broad context
- Slightly slower but much more informed responses

### Memory Management

As your channels grow, the system automatically manages memory:

1. **Recent Messages** (default): Full message history
2. **Compressed Summaries**: When you hit the "Summarize Old" button:
   - Groups old messages into 2-4k token chunks
   - AI creates concise summaries preserving key information
   - Original messages marked as "compressed" 
   - Summaries become available for RAG mode

### Advanced Features

#### Undo System
- **5-minute window** to undo any routing decision
- Removes the message, reverts scratchpad state
- Updates channel centroid (the AI's understanding of what belongs there)

#### Knowledge Cards
- Manually create distilled insights from conversations
- Reference specific message IDs for provenance
- Build a curated knowledge base of key learnings

#### Context Slices
- Full audit trail of every AI interaction
- See exactly which messages/summaries were used
- Debug and understand AI responses

## 🔧 Technical Architecture

### Backend (Python)
```
backend/
├── app/
│   ├── models/models.py      # Database schemas
│   ├── routers/              # API endpoints
│   │   ├── channels.py       # Channel CRUD
│   │   ├── messages.py       # Chat functionality  
│   │   ├── scratchpad.py     # Spitball system
│   │   ├── routing.py        # AI routing logic
│   │   └── knowledge_cards.py# Manual insights
│   ├── services/             # Business logic
│   │   ├── embedding_service.py    # OpenAI embeddings
│   │   ├── router_service.py       # Smart routing algorithm
│   │   ├── chat_service.py         # Memory & context management
│   │   └── description_service.py  # Living descriptions
│   └── main.py               # FastAPI application
├── alembic/                  # Database migrations
└── requirements.txt
```

### Frontend (React + TypeScript)
```
frontend/
├── src/
│   ├── components/
│   │   ├── ChannelTree.tsx   # Sidebar navigation
│   │   ├── ChannelView.tsx   # Chat interface
│   │   └── SpitballModal.tsx # Idea capture modal
│   ├── api.ts                # Backend integration
│   ├── types.ts              # TypeScript definitions
│   └── main.tsx              # Application entry
└── package.json
```

### Database Schema
```sql
-- Core entities
channels (id, parent_id, name, description, system_prompt, embedding_centroid)
messages (id, channel_id, role, content, embedding, memory_layer)
scratchpad_entries (id, content_json, content_text, state, embedding)

-- Intelligence & audit
routing_logs (id, entry_id, target_channel_id, confidence)
memory_summaries (id, channel_id, source_message_ids[], content)
context_slices (id, channel_id, user_message_id, included_*_ids[], mode)
knowledge_cards (id, channel_id, title, body, source_message_ids[])
```

## 🔍 How the AI Routing Works

The magic of Spitball lies in vector similarity:

1. **Embedding Generation**: Your text gets converted to a 1536-dimension vector using OpenAI's `text-embedding-3-small`

2. **Channel Centroids**: Each channel maintains a "centroid" - the average of all message vectors in that channel

3. **Similarity Calculation**: When you Spitball:
   ```
   similarity = cosine_similarity(spitball_vector, channel_centroid)
   ```

4. **Threshold Decision**: 
   - If `similarity ≥ 0.45` → suggest the best matching channel
   - If `similarity < 0.45` → propose creating a new channel

5. **Learning**: When you route content, the channel centroid updates incrementally:
   ```
   new_centroid = old_centroid + (new_vector - old_centroid) / message_count
   ```

This means channels get "smarter" about what belongs there over time!

## 🛠️ Development

### Adding New Features

1. **Database**: Add models to `backend/app/models/models.py`, create migration with `alembic revision --autogenerate -m "description"`
2. **API**: Add router to `backend/app/routers/`, include in `main.py`
3. **Frontend**: Add component to `frontend/src/components/`, wire up API calls

### Testing

```bash
cd backend
pytest  # Runs router, embedding, and integration tests
```

### API Documentation

Start the backend and visit `http://localhost:8000/docs` for interactive API documentation.

Key endpoints:
- `POST /scratchpad` - Create Spitball entry
- `POST /scratchpad/{id}/route` - Get AI routing suggestion  
- `POST /routing/{entry_id}/accept` - Accept routing
- `POST /channels/{id}/messages` - Send chat message
- `POST /channels/{id}/summarize_old` - Compress old messages

## 🚨 Troubleshooting

### Common Issues

#### "ModuleNotFoundError" when starting backend
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# And dependencies are installed
pip install -r requirements.txt
```

#### Database connection errors
```bash
# Check if PostgreSQL is running
docker compose ps

# Check logs
docker compose logs postgres

# Restart if needed
docker compose down
docker compose up -d postgres
```

#### Frontend won't start
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### OpenAI API errors
- Check your API key in `backend/.env`
- Verify you have credits in your OpenAI account
- Ensure the API key has the necessary permissions

#### Empty channels in sidebar
- Create your first channel manually using "+ New"
- The system needs at least one channel to start routing

### Environment Variables

Create `backend/.env` with these settings:

```bash
# Database (use Docker default or your own PostgreSQL)
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/second_brain

# OpenAI API (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-your-key-here

# AI Models (recommended defaults)
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini

# Routing behavior (0.45 = balanced, higher = more selective)
ROUTER_THRESHOLD=0.45

# Token management (8000 = good balance of context vs cost)
TOKEN_BUDGET=8000
```

## 🚀 Production Deployment

For production use:

1. **Database**: Use managed PostgreSQL with pgvector (AWS RDS, Google Cloud SQL, etc.)
2. **Backend**: Deploy with Gunicorn + Uvicorn workers
3. **Frontend**: Build with `npm run build` and serve via CDN/static hosting
4. **Environment**: Use secure environment variable management
5. **Authentication**: Add user authentication (not included in MVP)

## 📋 Current Limitations (MVP)

- **Single User**: No authentication or multi-user support
- **No Image Support**: Text-only content in Spitball editor
- **Manual Memory Management**: No automatic summarization scheduling
- **Basic UI**: Functional but not fully polished design
- **No Mobile Optimization**: Designed for desktop use

## 🔮 Future Enhancements (Roadmap)

- [ ] Multi-user support with authentication
- [ ] Image and file upload support
- [ ] Automatic background summarization
- [ ] Graph visualization of channel relationships
- [ ] Mobile-responsive design
- [ ] Automated knowledge card extraction
- [ ] Integration with external tools (Notion, Google Drive, etc.)
- [ ] Semantic search across all content
- [ ] Export/import functionality

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for powerful language models and embeddings
- **pgvector** for efficient vector similarity search
- **Editor.js** for the rich text editing experience
- **FastAPI** and **React** communities for excellent frameworks

---

**Ready to build your second brain?** Start with `docker compose up -d postgres` and follow the setup guide above! 🧠✨ 