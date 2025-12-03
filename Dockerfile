# syntax=docker/dockerfile:1

# ------------------------------
# Stage 1: Build environment
# ------------------------------
  FROM python:3.10-slim-bookworm AS base

  # Prevent Python from writing .pyc files and buffering stdout
  ENV PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1

  # Install only minimal runtime dependencies for OpenCV etc.
  RUN apt-get update --allow-releaseinfo-change && \
      apt-get install -y --no-install-recommends \
          libgl1-mesa-glx \
          libglib2.0-0 \
          curl \
      && rm -rf /var/lib/apt/lists/*

  WORKDIR /app

  # ------------------------------
  # Stage 2: Install Python deps
  # ------------------------------
  # Copy only requirements to leverage Docker layer caching
  COPY requirements.txt .

  # Install production dependencies
  RUN pip install --no-cache-dir --upgrade pip && \
      pip install --no-cache-dir -r requirements.txt

  # ------------------------------
  # Stage 3: Copy app code
  # ------------------------------
  # Copy the minimal necessary source files (after .dockerignore filtering)
  COPY . .

  # Prepare directories your app expects
  RUN mkdir -p /root/camera

  # ------------------------------
  # Environment variables
  # ------------------------------
  ARG SECRET_FTP_USER="user"
  ARG SECRET_FTP_PASS="password"
  ARG SECRET_MONGO_HOST="localhost"
  ARG SECRET_FTP_PORT="2121"
  ARG IS_TEST="prod"
  ARG FACES_HISTORY="24"

  ENV IS_TEST=$IS_TEST \
      FACES_HISTORY=$FACES_HISTORY \
      FTP_HOST=0.0.0.0 \
      FTP_PORT=$SECRET_FTP_PORT \
      PORT=$SECRET_FTP_PORT \
      FTP_USER=$SECRET_FTP_USER \
      FTP_PASS=$SECRET_FTP_PASS \
      MONGO_HOST=$SECRET_MONGO_HOST \
      MONGO_PORT=27017 \
      MONGO_DB="nill-home" \
      MONGO_COLLECTION="nill-home-photos" \
      PYTHONPATH=/app:$PYTHONPATH

  # ------------------------------
  # Health check (optional)
  # ------------------------------
  # HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  #   CMD curl -fs http://localhost:8080/health || exit 1

  # ------------------------------
  # Final CMD
  # ------------------------------
  CMD ["python", "main.py"]
