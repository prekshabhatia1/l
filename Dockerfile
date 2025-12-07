# Stage 1: Build Stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential 

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
# Expose the application port
EXPOSE 8000

WORKDIR /app

# Copy only the necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app/requirements.txt .
COPY app /app/app

# Command to run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/app/app/logging_config.py"] 
# Note: log_config is often needed to suppress uvicorn's default logging, 
# but we use JsonLogMiddleware instead. The above CMD is the standard entry.
# For this example, we'll rely on the Python logging setup in main.py
# Use python -m to execute uvicorn as a module, ensuring it runs correctly 
# even when the executable is not in the default PATH.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]