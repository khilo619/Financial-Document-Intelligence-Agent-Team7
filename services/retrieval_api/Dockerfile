FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HOME=/root/.cache/huggingface

# --------------------------------------------------
# PyTorch with CUDA 12.8
# --------------------------------------------------
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# --------------------------------------------------
# Python dependencies
# --------------------------------------------------
COPY services/retrieval_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------
# Application code
# --------------------------------------------------
COPY shared/ /app/shared/
COPY services/retrieval_api/src/ /app/src/

EXPOSE 8003

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8003"]