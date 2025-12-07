import logging
import sys
import json
from datetime import datetime
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from uuid import uuid4
import time

# Custom JSON Formatter
class JsonFormatter(logging.Formatter):
    """Formats logs as a single JSON line."""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "ts": datetime.fromtimestamp(record.created).isoformat() + "Z", # ISO-8601 [cite: 155]
            "level": record.levelname, # [cite: 157]
            "message": record.getMessage(),
            "logger_name": record.name,
            # Add extra fields if they exist
            **getattr(record, 'extra', {}) 
        }
        return json.dumps(log_record)

def setup_logging(log_level: str):
    """Sets up the global JSON logger."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Configure handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)
    
    # Optional: Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# Middleware for request logging
class JsonLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid4()) # unique per request [cite: 158]
        
        try:
            response = await call_next(request)
        except Exception as ex:
            # Handle unhandled exceptions for logging the 500 status
            response = Response("Internal Server Error", status_code=500)
            raise ex
        finally:
            process_time = time.time() - start_time
            latency_ms = int(process_time * 1000) # [cite: 163]
            
            # Base log record
            log_record = {
                "request_id": request_id,
                "method": request.method, # [cite: 160]
                "path": request.url.path, # [cite: 161]
                "status": response.status_code, # [cite: 162]
                "latency_ms": latency_ms,
            }
            
            # Get webhook-specific fields from request state if they exist
            if request.url.path == "/webhook":
                webhook_data = getattr(request.state, 'webhook_log', {})
                # message_id, dup, result fields [cite: 167-169]
                log_record.update(webhook_data) 
            
            # Log the request
            logger = logging.getLogger("api.request")
            # Using logger.info with an 'extra' dict to pass structured data
            logger.info("Request processed", extra=log_record)

        return response