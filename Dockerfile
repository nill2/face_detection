# syntax=docker/dockerfile:1
FROM python:3.10

# Install dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Create a folder for FTP
RUN mkdir -p /root/camera

# Define environment variables (secrets, FTP, MongoDB, etc.)
ARG SECRET_FTP_USER="user"
ARG SECRET_FTP_PASS="password"
ARG SECRET_MONGO_HOST="localhost"
ARG SECRET_FTP_PORT="2121"
ARG IS_TEST="prod"
ARG FACES_HISTORY="24"

# Set environment variables
ENV IS_TEST=$IS_TEST
ENV FACES_HISTORY=$FACES_HISTORY
ENV FTP_HOST=0.0.0.0
ENV FTP_PORT=$SECRET_FTP_PORT
ENV PORT=$SECRET_FTP_PORT
ENV FTP_USER=$SECRET_FTP_USER
ENV FTP_PASS=$SECRET_FTP_PASS
ENV MONGO_HOST=$SECRET_MONGO_HOST
ENV MONGO_PORT=27017
ENV MONGO_DB="nill-home"
ENV MONGO_COLLECTION="nill-home-photos"

ENV PYTHONPATH=/app:$PYTHONPATH


# Copy the entire project into the container
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the Python script
CMD ["python", "main.py"]
