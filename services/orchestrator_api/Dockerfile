FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install service dependencies
COPY services/orchestrator_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared contracts and service source code
COPY shared/ /app/shared/
COPY services/orchestrator_api/src/ /app/src/

EXPOSE 8001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
