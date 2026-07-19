FROM python:3.11-slim

WORKDIR /app

# Install only the runtime libraries needed by OpenCV and GUI-free image handling.
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first so Docker can cache the Python install layer.
COPY requirements.txt ./

# Install Python dependencies before copying the application source.
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code into the image.
COPY . .

# Create mount points used by the runtime and docker-compose volumes.
RUN mkdir -p models reports

# Default API port.
EXPOSE 8000

# Default command: run the FastAPI application.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
