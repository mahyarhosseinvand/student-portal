FROM python:3.10-slim

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    DATA_DIR=/tmp/data

# Install system dependencies required for dlib/face_recognition and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as non-root user with UID 1000
RUN useradd -m -u 1000 user

# Set up working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create writable data directory with chmod 777 so HF user 1000 can save photos and JSON without PermissionError
RUN mkdir -p /tmp/data/cloud_faces && \
    chmod -R 777 /tmp/data && \
    chown -R user:user /app

# Switch to non-root user
USER user

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Run Flask application via gunicorn on 0.0.0.0:7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
