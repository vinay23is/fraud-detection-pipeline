FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API source and model artifact
COPY api/ ./api/
COPY model/artifacts/model.pkl ./model/artifacts/model.pkl

# Run from api/ so local imports (feature_store, schemas) resolve correctly
WORKDIR /app/api

# Shell form so ${PORT} is expanded at runtime (Render injects PORT)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
