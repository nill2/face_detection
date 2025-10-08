# syntax=docker/dockerfile:1
FROM python:3.10-slim-bullseye

# Install dependencies for OpenCV
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /root/camera

# Environment variables
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

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

# Add a simple health check (adjust the URL to your app)
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
  CMD bash -c 'test $(($(date +%s) - $(cat /tmp/healthcheck.txt 2>/dev/null || echo 0))) -lt 120'


CMD ["python", "main.py"]
