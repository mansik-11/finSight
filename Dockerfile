# FinSight — Streamlit application image
FROM python:3.12-slim

WORKDIR /app

# System dependencies needed by xgboost / scientific stack at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project.
COPY . .

# Build the analytical pipeline artifacts (data -> models -> evaluation) at
# image build time, so the container starts instantly with everything the
# dashboard needs already on disk — no training happens at runtime.
RUN python -m src.pipeline \
    && python -m src.train \
    && python -m src.evaluate

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
    "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
