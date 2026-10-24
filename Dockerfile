FROM python:3.11.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DO_NOT_TRACK=1 \
    PREFECT_SERVER_ANALYTICS_ENABLED=false

COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt
ENV PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS=300

COPY src ./src
COPY tests ./tests

CMD ["python", "-m", "src.pipeline"]