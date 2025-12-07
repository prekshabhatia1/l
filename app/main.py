import os
import hmac
import hashlib
import json
import sqlite3
import time
import secrets
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, Depends, Query, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# =====================================================
# ENVIRONMENT CONFIG
# =====================================================
DATABASE_URL = os.environ.get("DATABASE_URL")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

if not WEBHOOK_SECRET:
    raise RuntimeError("WEBHOOK_SECRET must be set before startup.")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be set (e.g., sqlite:////data/app.db)")

DB_PATH = DATABASE_URL.replace("sqlite:///", "")

# =====================================================
# LOGGING (Structured JSON)
# =====================================================
import logging

class JSONFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created)),
            "level": record.levelname,
            "request_id": getattr(record, "request_id", None),
            "method": getattr(record, "method", None),
            "path": getattr(record, "path", None),
            "status": getattr(record, "status", None),
            "latency_ms": getattr(record, "latency_ms", None),
            "message_id": getattr(record, "message_id", None),
            "dup": getattr(record, "dup", None),
            "result": getattr(record, "result", None),
        }
        # remove None entries
        payload = {k: v for k, v in payload.items() if v is not None}
        return json.dumps(payload)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.handlers.clear()
logger.addHandler(handler)

def log_event(request: Request, **kwargs):
    logger.info("", extra={
        "request_id": request.state.request_id,
        "method": request.method,
        "path": request.url.path,
        **kwargs
    })

# =====================================================
# PROMETHEUS METRICS
# =====================================================
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "status"]
)

WEBHOOK_RESULTS = Counter(
    "webhook_requests_total",
    "Webhook outcomes",
    ["result"]
)

REQUEST_LATENCY = Histogram(
    "request_latency_ms",
    "Request latency in ms",
    buckets=[10, 50, 100, 250, 500, 1000, float("inf")]
)

# =====================================================
# DATABASE
# =====================================================
class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._conn()
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                from_msisdn TEXT NOT NULL,
                to_msisdn TEXT NOT NULL,
                ts TEXT NOT NULL,
                text TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def insert_message(self, msg: dict) -> bool:
        """Return True if inserted, False if duplicate."""
        conn = self._conn()
        c = conn.cursor()
        try:
            c.execute("""
                INSERT INTO messages
                (message_id, from_msisdn, to_msisdn, ts, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                msg["message_id"],
                msg["from"],
                msg["to"],
                msg["ts"],
                msg.get("text"),
                time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def list_messages(self, limit, offset, from_filter, since, q):
        conn = self._conn()
        c = conn.cursor()
        clauses = ["1=1"]
        params = []

        if from_filter:
            clauses.append("from_msisdn = ?")
            params.append(from_filter)

        if since:
            clauses.append("ts >= ?")
            params.append(since)

        if q:
            clauses.append("LOWER(text) LIKE LOWER(?)")
            params.append(f"%{q}%")

        where = " AND ".join(clauses)

        # total count
        c.execute(f"SELECT COUNT(*) FROM messages WHERE {where}", params)
        total = c.fetchone()[0]

        # paginated
        c.execute(f"""
            SELECT message_id, from_msisdn, to_msisdn, ts, text
            FROM messages
            WHERE {where}
            ORDER BY ts ASC, message_id ASC
            LIMIT ? OFFSET ?
        """, params + [limit, offset])

        rows = c.fetchall()
        conn.close()

        data = []
        for r in rows:
            data.append({
                "message_id": r[0],
                "from": r[1],
                "to": r[2],
                "ts": r[3],
                "text": r[4]
            })

        return {
            "data": data,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    def stats(self):
        conn = self._conn()
        c = conn.cursor()

        c.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM messages")
        total, first_ts, last_ts = c.fetchone()

        c.execute("""
            SELECT from_msisdn, COUNT(*)
            FROM messages
            GROUP BY from_msisdn
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        senders = [{"from": r[0], "count": r[1]} for r in c.fetchall()]

        c.execute("SELECT COUNT(DISTINCT from_msisdn) FROM messages")
        senders_count = c.fetchone()[0]

        conn.close()

        return {
            "total_messages": total,
            "senders_count": senders_count,
            "messages_per_sender": senders,
            "first_message_ts": first_ts,
            "last_message_ts": last_ts
        }


storage = Storage(DB_PATH)

def get_storage():
    return storage


# =====================================================
# SCHEMA
# =====================================================
class WebhookMessage(BaseModel):
    message_id: str
    from_: str = Field(alias="from")
    to: str
    ts: str
    text: Optional[str] = Field(default=None, max_length=4096)

# =====================================================
# FASTAPI APP + MIDDLEWARE
# =====================================================
app = FastAPI()

@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.time()
    request.state.request_id = secrets.token_hex(8)

    try:
        response = await call_next(request)
    finally:
        latency = (time.time() - start) * 1000
        REQUEST_LATENCY.observe(latency)

        HTTP_REQUESTS.labels(
            path=request.url.path,
            status=getattr(request.state, "status", response.status_code)
        ).inc()

        log_event(
            request,
            status=response.status_code,
            latency_ms=round(latency, 2),
            result=getattr(request.state, "result", None),
            message_id=getattr(request.state, "message_id", None),
            dup=getattr(request.state, "dup", None)
        )

    return response

# =====================================================
# SIGNATURE VERIFICATION DEPENDENCY
# =====================================================
# =====================================================
# SIGNATURE VERIFICATION DEPENDENCY
# =====================================================
async def verify_signature(request: Request):
    sig = request.headers.get("x-signature")
    if not sig:
        request.state.status = 401
        WEBHOOK_RESULTS.labels("invalid_signature").inc()
        raise HTTPException(status_code=401, detail="invalid signature")

    # 1. READ THE RAW BODY BYTES
    body = await request.body()
    
    # ...
    # 1. READ THE RAW BODY BYTES
    body = await request.body()
    
    # 2. FIX: Strip the secret of any invisible leading/trailing whitespace/newlines
    CLEAN_SECRET = WEBHOOK_SECRET.strip() 

    # 3. COMPUTE THE HMAC DIGEST (using the clean secret)
    calc = hmac.new(
        CLEAN_SECRET.encode(),    # <-- USE THE CLEAN SECRET HERE
        body, 
        hashlib.sha256
    ).hexdigest()
# ...

    # 3. COMPARE THE DIGESTS (Crucial: hmac.compare_digest)
    if not hmac.compare_digest(calc, sig):
        request.state.status = 401
        WEBHOOK_RESULTS.labels("invalid_signature").inc()
        raise HTTPException(status_code=401, detail="invalid signature")
    

    # SUCCESS: Execution continues to the webhook handler.

# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/health/live")
def health_live():
    return {"status": "live"}


@app.get("/health/ready")
def health_ready():
    try:
        sqlite3.connect(DB_PATH).cursor()
        return {"status": "ready"}
    except:
        raise HTTPException(status_code=503)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/webhook", dependencies=[Depends(verify_signature)])
async def webhook(request: Request, storage: Storage = Depends(get_storage)):
    raw = await request.body()
    try:
        data = json.loads(raw)
    except:
        request.state.status = 422
        WEBHOOK_RESULTS.labels("validation_error").inc()
        raise HTTPException(status_code=422, detail="Invalid JSON")

    # Validate schema
    try:
        msg = WebhookMessage(**data)
    except Exception as e:
        request.state.status = 422
        WEBHOOK_RESULTS.labels("validation_error").inc()
        raise HTTPException(status_code=422, detail=str(e))

    request.state.message_id = msg.message_id

    inserted = storage.insert_message(data)
    if inserted:
        request.state.dup = False
        request.state.status = 200
        request.state.result = "created"
        WEBHOOK_RESULTS.labels("created").inc()
    else:
        request.state.dup = True
        request.state.status = 200
        request.state.result = "duplicate"
        WEBHOOK_RESULTS.labels("duplicate").inc()

    return {"status": "ok"}


@app.get("/messages")
def messages(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    from_: Optional[str] = Query(None, alias="from"),
    since: Optional[str] = None,
    q: Optional[str] = None,
    storage: Storage = Depends(get_storage)
):
    result = storage.list_messages(limit, offset, from_, since, q)
    return result


@app.get("/stats")
def stats(storage: Storage = Depends(get_storage)):
    return storage.stats()
