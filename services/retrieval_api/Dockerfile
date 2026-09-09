FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY services/retrieval_api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY services/retrieval_api/src/ /app/src/

EXPOSE 8003

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8003"]