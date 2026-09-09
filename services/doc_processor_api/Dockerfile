# FROM python:3.11-slim

# WORKDIR /app

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONPATH=/app

# COPY services/doc_processor_api/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY shared/ /app/shared/
# COPY services/doc_processor_api/src/ /app/src/

# EXPOSE 8002

# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/doc_processor_api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY services/doc_processor_api/src/ /app/src/

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]