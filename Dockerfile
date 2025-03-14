# Use the official PyTorch image as the base
FROM pytorch/pytorch

# Set the working directory inside the container
WORKDIR /app

# Install additional dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository and install Python requirements
RUN rm -rf remote_qdrant_client && \
    git clone https://github.com/AlphaCoAI-Projects/remote_qdrant_client.git && \
    pip install --no-cache-dir -r remote_qdrant_client/requirements.txt

# Set the working directory to the cloned repository
WORKDIR /app/remote_qdrant_client

# Expose ports
EXPOSE 80 8000 

# Set the ENTRYPOINT to use Uvicorn to run the FastAPI app
ENTRYPOINT ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 80"]
