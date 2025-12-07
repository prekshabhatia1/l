from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time

# --- Metrics Definitions ---
# Counter for total HTTP requests [cite: 133-135]
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total', 'Total HTTP Requests', 
    ['path', 'status']
)

# Counter for webhook processing outcomes [cite: 136-140]
WEBHOOK_REQUESTS_TOTAL = Counter(
    'webhook_requests_total', 'Webhook Processing Outcomes', 
    ['result']
)

# Histogram for request latency [cite: 141-145]
REQUEST_LATENCY_MS = Histogram(
    'request_latency_ms', 'Request latency in milliseconds',
    buckets=[10, 50, 100, 500, 1000, 5000, float('inf')]
)

# --- Middleware ---
class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        latency_ms = process_time * 1000

        # Increment total HTTP requests counter
        HTTP_REQUESTS_TOTAL.labels(
            path=request.url.path, 
            status=response.status_code
        ).inc()
        
        # Observe latency
        REQUEST_LATENCY_MS.observe(latency_ms)

        # Increment webhook outcome counter if applicable
        if request.url.path == "/webhook" and hasattr(request.state, 'webhook_log'):
            result = request.state.webhook_log.get('result')
            if result:
                WEBHOOK_REQUESTS_TOTAL.labels(result=result).inc()

        return response