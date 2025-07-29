from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import channels, messages, scratchpad, routing, knowledge_cards

app = FastAPI(
    title="Channel Second Brain API",
    description="Multi-level channel tree with scratchpad routing and memory layering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(channels.router)
app.include_router(messages.router)
app.include_router(scratchpad.router)
app.include_router(routing.router)
app.include_router(knowledge_cards.router)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Channel Second Brain API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Channel Second Brain API", 
        "version": "1.0.0",
        "docs": "/docs"
    } 