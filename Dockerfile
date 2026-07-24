FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for FAISS and compiling some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY axiom_model/requirements.txt ./axiom_model/requirements.txt
COPY axiom_web/backend/requirements.txt ./axiom_web/backend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r axiom_web/backend/requirements.txt
RUN pip install --no-cache-dir -r axiom_model/requirements.txt

# Copy the entire monorepo into the container
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PORT=8000
ENV HOST=0.0.0.0

# Run the FastAPI server
CMD ["uvicorn", "axiom_web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
