Lyftr AI Containerized Webhook API
A FastAPI service that ingests webhooks, enforces HMAC security and idempotency, and exposes analytics and metrics.

How to Run
bash
# Set environment variables
export WEBHOOK_SECRET="testsecret"
export DATABASE_URL="sqlite:////data/app.db"

# Start stack
make up   # (docker compose up -d --build)

# Logs / Shutdown
make logs
make down
Service runs at: http://localhost:8000

 Endpoints
Health

/health/live → always 200

/health/ready → 200 if DB + secret OK, else 503

Webhook

POST /webhook with JSON body + X-Signature

Invalid sig → 401; valid sig → 200 + insert; duplicate → 200 no new row

Messages

GET /messages with limit, offset, from, since, q

Returns data, total, limit, offset

Stats

GET /stats → totals, distinct senders, per-sender counts, first/last timestamps

Metrics

GET /metrics → Prometheus counters (http_requests_total, webhook_requests_total, latency, message_total)

 Design Decisions
HMAC verification: X-Signature checked via HMAC-SHA256(secret, body) with hmac.compare_digest.

Pagination: /messages enforces limit (1–100), offset, deterministic ordering (ts ASC, message_id ASC).

Stats & Metrics: /stats aggregates DB counts; /metrics exposes Prometheus counters for requests, webhook outcomes, latency, and message totals.