import json
import os
import re
import logging
from typing import List

import httpx
from dotenv import load_dotenv

from app.models.schemas import (
    AnalyticsResponse,
    TranscriptSegment,
    IntentType,
    TopicType,
    SentimentType,
    EscalationRisk,
)

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are a real-time conversation analytics engine for a call center.
Analyze the provided conversation transcript and return ONLY a valid JSON object with this exact structure:
{
  "intent": "<one of: billing dispute, account inquiry, payment issue, technical support, service complaint, subscription cancellation>",
  "intent_confidence": <number 0-100>,
  "topic": "<one of: credit card charges, account access issues, subscription cancellation, service outage, duplicate transaction, password reset, other>",
  "topic_confidence": <number 0-100>,
  "sentiment": "<one of: positive, neutral, negative>",
  "sentiment_confidence": <number 0-100>,
  "escalation_risk": "<one of: high, medium, low>",
  "escalation_signals": ["<exact phrase from transcript>", ...],
  "summary": "<one sentence summary of the conversation>",
  "key_phrases": ["<phrase1>", "<phrase2>", "<phrase3>"]
}
Return ONLY valid JSON. No markdown, no explanation, no code fences."""


def format_transcript(segments: List[TranscriptSegment]) -> str:
    return "\n".join(f"{seg.speaker}: {seg.text}" for seg in segments)


def _safe_enum(value: str, enum_class, default):
    """Coerce a string value to an enum, falling back to default."""
    try:
        return enum_class(value.lower().strip())
    except ValueError:
        return default


async def analyze_conversation(
    segments: List[TranscriptSegment],
    session_id: str | None = None,
    api_key: str | None = None,
) -> AnalyticsResponse:
    """
    Send transcript segments to Groq and parse structured analytics.
    Raises ValueError on API errors or unparseable responses.
    """
    key = api_key or GROQ_API_KEY
    if not key:
        raise ValueError("GROQ_API_KEY is not set. Provide it via env or request header.")

    transcript_text = format_transcript(segments)
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": 1024,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this conversation:\n\n{transcript_text}"},
        ],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code != 200:
        error_detail = response.text
        logger.error("Groq API error %s: %s", response.status_code, error_detail)
        raise ValueError(f"Groq API returned {response.status_code}: {error_detail}")

    data = response.json()
    raw_content = data["choices"][0]["message"]["content"]

    # Strip markdown fences if model ignores the instruction
    cleaned = re.sub(r"```(?:json)?|```", "", raw_content).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s\nRaw: %s", e, cleaned)
        raise ValueError(f"LLM returned invalid JSON: {e}")

    return AnalyticsResponse(
        intent=_safe_enum(parsed.get("intent", ""), IntentType, IntentType.SERVICE_COMPLAINT),
        intent_confidence=int(parsed.get("intent_confidence", 0)),
        topic=_safe_enum(parsed.get("topic", ""), TopicType, TopicType.OTHER),
        topic_confidence=int(parsed.get("topic_confidence", 0)),
        sentiment=_safe_enum(parsed.get("sentiment", ""), SentimentType, SentimentType.NEUTRAL),
        sentiment_confidence=int(parsed.get("sentiment_confidence", 0)),
        escalation_risk=_safe_enum(parsed.get("escalation_risk", ""), EscalationRisk, EscalationRisk.LOW),
        escalation_signals=parsed.get("escalation_signals", []),
        summary=parsed.get("summary", ""),
        key_phrases=parsed.get("key_phrases", []),
        segment_count=len(segments),
        session_id=session_id,
    )
