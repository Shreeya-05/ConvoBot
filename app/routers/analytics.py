from fastapi import APIRouter, HTTPException, Header
from typing import Optional

from app.models.schemas import AnalyticsRequest, AnalyticsResponse
from app.services.analytics_service import analyze_conversation

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyticsResponse,
    summary="Analyze a conversation transcript",
    description=(
        "Submit one or more transcript segments and receive structured analytics: "
        "intent, topic, sentiment, escalation risk, and key phrases."
    ),
)
async def analyze(
    request: AnalyticsRequest,
    x_groq_api_key: Optional[str] = Header(None, description="Groq API key (overrides env var)"),
):
    """
    **REST endpoint** — submit the full (or partial) transcript and get analytics back.

    Call this endpoint after each new ASR segment to get updated real-time analytics.
    """
    try:
        result = await analyze_conversation(
            segments=request.segments,
            session_id=request.session_id,
            api_key=x_groq_api_key,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get(
    "/demo",
    response_model=AnalyticsResponse,
    summary="Run analytics on a built-in demo transcript",
    description="Useful for testing without sending a request body.",
)
async def demo_analyze(
    x_groq_api_key: Optional[str] = Header(None),
):
    from app.models.schemas import TranscriptSegment, Speaker

    demo_segments = [
        TranscriptSegment(speaker=Speaker.CUSTOMER, text="I was charged twice for the same transaction."),
        TranscriptSegment(speaker=Speaker.CUSTOMER, text="I already called yesterday but it was not resolved."),
        TranscriptSegment(speaker=Speaker.CUSTOMER, text="I need this issue fixed immediately."),
    ]
    try:
        return await analyze_conversation(demo_segments, session_id="demo", api_key=x_groq_api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
