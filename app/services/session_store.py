from collections import defaultdict
from typing import Dict, List
from app.models.schemas import TranscriptSegment


class SessionStore:
    """
    Simple in-memory store mapping session_id → list of transcript segments.
    In production, replace with Redis or a DB-backed store.
    """

    def __init__(self):
        self._sessions: Dict[str, List[TranscriptSegment]] = defaultdict(list)

    def append(self, session_id: str, segment: TranscriptSegment) -> List[TranscriptSegment]:
        self._sessions[session_id].append(segment)
        return self._sessions[session_id]

    def get(self, session_id: str) -> List[TranscriptSegment]:
        return self._sessions.get(session_id, [])

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def all_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# Singleton used across routers
session_store = SessionStore()
