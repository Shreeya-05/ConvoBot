from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from enum import Enum


class Speaker(str, Enum):
    CUSTOMER = "Customer"
    AGENT = "Agent"


class TranscriptSegment(BaseModel):
    speaker: Speaker = Field(..., description="Who is speaking")
    text: str = Field(..., min_length=1, description="Spoken text segment")

    model_config = {"json_schema_extra": {"example": {"speaker": "Customer", "text": "I was charged twice for the same transaction."}}}


class AnalyticsRequest(BaseModel):
    segments: List[TranscriptSegment] = Field(..., min_length=1, description="Conversation transcript segments so far")
    session_id: Optional[str] = Field(None, description="Optional session identifier for tracking")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "call-001",
                "segments": [
                    {"speaker": "Customer", "text": "I was charged twice for the same transaction."},
                    {"speaker": "Customer", "text": "I already called yesterday but it was not resolved."},
                    {"speaker": "Customer", "text": "I need this issue fixed immediately."},
                ],
            }
        }
    }


class IntentType(str, Enum):
    BILLING_DISPUTE = "billing dispute"
    ACCOUNT_INQUIRY = "account inquiry"
    PAYMENT_ISSUE = "payment issue"
    TECHNICAL_SUPPORT = "technical support"
    SERVICE_COMPLAINT = "service complaint"
    SUBSCRIPTION_CANCELLATION = "subscription cancellation"


class TopicType(str, Enum):
    CREDIT_CARD_CHARGES = "credit card charges"
    ACCOUNT_ACCESS_ISSUES = "account access issues"
    SUBSCRIPTION_CANCELLATION = "subscription cancellation"
    SERVICE_OUTAGE = "service outage"
    DUPLICATE_TRANSACTION = "duplicate transaction"
    PASSWORD_RESET = "password reset"
    OTHER = "other"


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class EscalationRisk(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnalyticsResponse(BaseModel):
    intent: IntentType = Field(..., description="Primary detected user intent")
    intent_confidence: int = Field(..., ge=0, le=100, description="Confidence score 0-100")
    topic: TopicType = Field(..., description="Conversation topic classification")
    topic_confidence: int = Field(..., ge=0, le=100, description="Confidence score 0-100")
    sentiment: SentimentType = Field(..., description="Overall user sentiment")
    sentiment_confidence: int = Field(..., ge=0, le=100, description="Confidence score 0-100")
    escalation_risk: EscalationRisk = Field(..., description="Likelihood of conversation escalating")
    escalation_signals: List[str] = Field(default_factory=list, description="Detected escalation trigger phrases")
    summary: str = Field(..., description="One-sentence conversation summary")
    key_phrases: List[str] = Field(default_factory=list, description="Key phrases extracted from conversation")
    segment_count: int = Field(..., description="Number of segments analyzed")
    session_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "intent": "billing dispute",
                "intent_confidence": 95,
                "topic": "duplicate transaction",
                "topic_confidence": 92,
                "sentiment": "negative",
                "sentiment_confidence": 88,
                "escalation_risk": "high",
                "escalation_signals": ["I need this issue fixed immediately", "I already called yesterday"],
                "summary": "Customer is disputing a duplicate credit card charge that was unresolved from a previous call.",
                "key_phrases": ["charged twice", "not resolved", "fixed immediately"],
                "segment_count": 3,
                "session_id": "call-001",
            }
        }
    }


class StreamSegmentRequest(BaseModel):
    """Used for WebSocket — send one segment at a time."""
    speaker: Speaker
    text: str
    session_id: Optional[str] = None
