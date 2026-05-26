FROM python:3.11-slim

# Set environment variables to optimize Python behavior inside Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt ./

# Update pip
# Install PyTorch CPU first to avoid large CUDA packages
# Install remaining dependencies from requirements.txt
# Install uvicorn (required to run the server)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn

# Copy entire src directory (includes api.py, artifacts/, static/)
COPY src/ ./src/

# Copy ml-100k-dataset if it's needed at runtime
COPY ml-100k-dataset/ ./ml-100k-dataset/

# Change to src directory so api.py paths are relative to there
WORKDIR /app/src

# Expose port 8080
EXPOSE 8080

# Command to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
