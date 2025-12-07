from sqlalchemy import create_engine, func, text, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from datetime import datetime
from typing import Optional, List, Dict, Any

from .config import settings
from .models import Base, MessageDB, WebhookMessageIn

# Setup DB connection
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all tables defined in Base."""
    Base.metadata.create_all(bind=engine)

def db_ready() -> bool:
    """Checks if DB connection is active and schema is applied."""
    try:
        # Check connection and table existence by querying
        init_db() 
        with SessionLocal() as db:
            db.execute(text("SELECT 1 FROM messages LIMIT 1"))
        return True
    except Exception:
        return False

def insert_message(message_data: WebhookMessageIn) -> str:
    """
    Inserts a new message into the DB. 
    Returns 'created' or 'duplicate'.
    """
    db = SessionLocal()
    try:
        now_ts = datetime.utcnow().isoformat() + "Z"
        new_msg = MessageDB(
            message_id=message_data.message_id,
            from_msisdn=message_data.from_,
            to_msisdn=message_data.to,
            ts=message_data.ts,
            text=message_data.text,
            created_at=now_ts
        )
        db.add(new_msg)
        db.commit()
        return "created"
    except IntegrityError:
        # Gracefully handle UNIQUE constraint violation on message_id [cite: 176]
        db.rollback()
        return "duplicate"
    finally:
        db.close()

def get_messages(limit: int, offset: int, from_msisdn: Optional[str] = None, 
                 since: Optional[str] = None, q: Optional[str] = None) -> Dict[str, Any]:
    """Fetches messages with pagination and filters [cite: 65-84]."""
    db = SessionLocal()
    try:
        # Start with a base query
        query = db.query(MessageDB)
        
        # Apply filters
        if from_msisdn:
            # Filter by from_msisdn exact match [cite: 76]
            query = query.filter(MessageDB.from_msisdn == from_msisdn)
        if since:
            # Filter by ts >= since [cite: 79]
            query = query.filter(MessageDB.ts >= since)
        if q:
            # Free-text search in text (case-insensitive substring match) [cite: 81]
            query = query.filter(MessageDB.text.ilike(f"%{q}%"))
        
        # Get total count (before limit/offset) [cite: 103]
        total = query.count()
        
        # Apply ordering: ts ASC, message_id ASC [cite: 83]
        query = query.order_by(MessageDB.ts.asc(), MessageDB.message_id.asc())
        
        # Apply pagination [cite: 68-73]
        query = query.limit(limit).offset(offset)
        
        data = query.all()
        
        return {
            "data": data,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    finally:
        db.close()

def get_stats() -> Dict[str, Any]:
    """Computes message-level analytics [cite: 105-117]."""
    db = SessionLocal()
    try:
        # Total messages [cite: 109]
        total_messages = db.query(MessageDB).count()
        
        # Senders count [cite: 110]
        senders_count = db.query(func.count(func.distinct(MessageDB.from_msisdn))).scalar()
        
        # Top 10 senders (messages_per_sender) [cite: 111-115]
        top_senders = db.query(
            MessageDB.from_msisdn,
            func.count(MessageDB.from_msisdn).label('count')
        ).group_by(MessageDB.from_msisdn).order_by(desc('count')).limit(10).all()
        
        messages_per_sender = [{"from": s.from_msisdn, "count": s.count} for s in top_senders]
        
        # First and last message timestamps [cite: 116-117]
        first_message_ts = db.query(func.min(MessageDB.ts)).scalar()
        last_message_ts = db.query(func.max(MessageDB.ts)).scalar()
        
        return {
            "total_messages": total_messages,
            "senders_count": senders_count or 0,
            "messages_per_sender": messages_per_sender,
            "first_message_ts": first_message_ts,
            "last_message_ts": last_message_ts
        }
    finally:
        db.close()