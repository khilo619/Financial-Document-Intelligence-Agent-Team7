FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv

# Use the virtual environment by default
ENV PATH="/opt/venv/bin:$PATH"

COPY services/retrieval_api/requirements.txt .

# Install CUDA-enabled PyTorch
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY services/retrieval_api/src/ /app/src/

EXPOSE 8003

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8003"]