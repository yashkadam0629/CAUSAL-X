FROM python:3.12-slim
WORKDIR /app/backend
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
ENV CAUSALX_DATA_DIR=/app/data CAUSALX_ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
