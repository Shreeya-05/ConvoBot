from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analytics, websocket, chat

app = FastAPI(
    title="Real-Time Conversation Analytics",
    description="Streaming ASR conversation analytics powered by Groq LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analytics.router, prefix="/api", tags=["Analytics"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(websocket.router, tags=["WebSocket"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Conversation Analytics API is running"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
