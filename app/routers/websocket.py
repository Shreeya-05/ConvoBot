import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional

from app.models.schemas import TranscriptSegment, Speaker
from app.services.analytics_service import analyze_conversation
from app.services.session_store import session_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    session_id: Optional[str] = Query(default="default"),
    groq_api_key: Optional[str] = Query(default=None),
):
    """
    **WebSocket endpoint** for real-time streaming ASR analytics.

    Connect with:  `ws://localhost:8000/ws/stream?session_id=call-001&groq_api_key=gsk_...`

    **Send** JSON segments one at a time:
    ```json
    {"speaker": "Customer", "text": "I was charged twice."}
    ```

    **Receive** updated analytics after each segment:
    ```json
    {"intent": "billing dispute", "sentiment": "negative", ...}
    ```

    Send `{"action": "reset"}` to clear the session history.
    Send `{"action": "end"}` to close the connection.
    """
    await websocket.accept()
    logger.info("WebSocket connected: session=%s", session_id)

    await websocket.send_json({
        "event": "connected",
        "session_id": session_id,
        "message": "Send transcript segments as JSON: {speaker, text}",
    })

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"event": "error", "detail": "Invalid JSON"})
                continue

            # Handle control actions
            action = data.get("action")
            if action == "reset":
                session_store.clear(session_id)
                await websocket.send_json({"event": "reset", "session_id": session_id})
                continue
            if action == "end":
                await websocket.send_json({"event": "goodbye", "session_id": session_id})
                break

            # Parse transcript segment
            try:
                speaker_raw = data.get("speaker", "Customer")
                speaker = Speaker(speaker_raw)
                text = data.get("text", "").strip()
                if not text:
                    await websocket.send_json({"event": "error", "detail": "text field is required"})
                    continue
                segment = TranscriptSegment(speaker=speaker, text=text)
            except (ValueError, KeyError) as e:
                await websocket.send_json({"event": "error", "detail": f"Invalid segment: {e}"})
                continue

            # Accumulate segment
            all_segments = session_store.append(session_id, segment)

            # Notify client we received the segment
            await websocket.send_json({
                "event": "segment_received",
                "segment_count": len(all_segments),
                "segment": {"speaker": segment.speaker, "text": segment.text},
            })

            # Run analytics
            try:
                analytics = await analyze_conversation(
                    segments=all_segments,
                    session_id=session_id,
                    api_key=groq_api_key,
                )
                await websocket.send_json({
                    "event": "analytics",
                    **analytics.model_dump(),
                })
            except ValueError as e:
                await websocket.send_json({"event": "error", "detail": str(e)})
            except Exception as e:
                logger.exception("Analytics error for session %s", session_id)
                await websocket.send_json({"event": "error", "detail": f"Analysis failed: {e}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
    finally:
        session_store.clear(session_id)
