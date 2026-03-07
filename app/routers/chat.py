import logging
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a helpful, professional customer support agent.
Keep responses concise (2-3 sentences max). Be empathetic and solution-focused.
You handle calls about billing, account issues, technical problems, and subscriptions."""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None


@router.post("/chat", response_model=ChatResponse, summary="Get a bot reply")
async def chat(
    request: ChatRequest,
    x_groq_api_key: Optional[str] = Header(None),
):
    key = x_groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is not set.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "messages": messages,
                },
            )

        data = response.json()
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=data.get("error", {}).get("message", "Groq error"))

        reply = data["choices"][0]["message"]["content"]
        return ChatResponse(reply=reply, session_id=request.session_id)

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Groq API timed out.")
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))
