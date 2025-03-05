FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository and install requirements
RUN git clone https://github.com/AlphaCoAI-Projects/remote_qdrant_client.git && \
    pip install --no-cache-dir -r remote_qdrant_client/requirements.txt

# Set WORKDIR to inside the repo (this is important if your app runs from inside the repo)
WORKDIR /app/remote_qdrant_client

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
