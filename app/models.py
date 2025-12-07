from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# --- DB Model (SQLAlchemy) ---
class MessageDB(Base):
    """Database model for an ingested message."""
    __tablename__ = "messages"
    
    # message_id TEXT PRIMARY KEY [cite: 191]
    message_id = Column(String, primary_key=True) 
    # from_msisdn TEXT NOT NULL [cite: 192]
    from_msisdn = Column(String, nullable=False)
    # to_msisdn TEXT NOT NULL [cite: 193]
    to_msisdn = Column(String, nullable=False)
    # ts TEXT NOT NULL (ISO-8601 UTC string) [cite: 194, 197]
    ts = Column(String, nullable=False)
    # text TEXT [cite: 198]
    text = Column(String, nullable=True)
    # created_at TEXT NOT NULL (server time ISO-8601) [cite: 199, 200]
    created_at = Column(String, nullable=False) 

# --- Pydantic Models ---
def validate_e164(v):
    """Simple validation for E.164-like form (+ then digits)[cite: 46]."""
    if not v or not (v.startswith('+') and v[1:].isdigit()):
        raise ValueError("Must be in E.164-like form (e.g., +1234567890).")
    return v

def validate_iso8601(v):
    """Validation for ISO-8601 UTC string with Z suffix[cite: 47]."""
    if not v.endswith('Z'):
        raise ValueError("Must end with 'Z' for UTC.")
    try:
        datetime.strptime(v, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        try:
            datetime.strptime(v, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            raise ValueError("Must be a valid ISO-8601 UTC string with 'Z'.")
    return v

class WebhookMessageIn(BaseModel):
    """Inbound message payload for /webhook [cite: 18-25]."""
    message_id: str = Field(..., min_length=1)
    # from/to: E.164-like form [cite: 46]
    from_: str = Field(..., alias="from") 
    to: str
    # ts: ISO-8601 UTC string with Z suffix [cite: 47]
    ts: str
    # text: optional string, max length 4096 [cite: 48]
    text: Optional[str] = Field(None, max_length=4096)

    # Apply validators
    _validate_from = validator('from_', allow_reuse=True)(validate_e164)
    _validate_to = validator('to', allow_reuse=True)(validate_e164)
    _validate_ts = validator('ts', allow_reuse=True)(validate_iso8601)

class MessageOut(BaseModel):
    """Response shape for GET /messages [cite: 88-96]."""
    message_id: str
    from_: str = Field(..., alias="from")
    to: str
    ts: str
    text: Optional[str]

    class Config:
        allow_population_by_field_name = True

class MessagesResponse(BaseModel):
    """Response for GET /messages, including pagination data [cite: 85-102]."""
    data: List[MessageOut]
    total: int # total rows matching filters [cite: 103]
    limit: int
    offset: int

class StatsPerSender(BaseModel):
    """Sender statistics structure [cite: 113-114]."""
    from_: str = Field(..., alias="from")
    count: int

class StatsResponse(BaseModel):
    """Response for GET /stats [cite: 108-118]."""
    total_messages: int
    senders_count: int
    messages_per_sender: List[StatsPerSender]
    first_message_ts: Optional[str]
    last_message_ts: Optional[str]