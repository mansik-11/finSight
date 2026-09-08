.PHONY: setup data train evaluate pipeline test run docker-build docker-run clean

# Install Python dependencies
setup:
	pip install -r requirements.txt

# Run the reproducible data pipeline (ingest -> validate -> feature engineer -> experiment sim)
data:
	python -m src.pipeline

# Train Logistic Regression + XGBoost and log to local MLflow
train:
	python -m src.train

# Evaluate trained models (ROC-AUC, PR-AUC, calibration, plots)
evaluate:
	python -m src.evaluate

# Run the full pipeline end-to-end: data -> train -> evaluate
pipeline: data train evaluate

# Run the automated test suite
test:
	pytest tests/ -v

# Launch the Streamlit dashboard
run:
	streamlit run app/streamlit_app.py

# Build the Docker image
docker-build:
	docker build -t finsight:latest .

# Run the Docker container (maps port 8501)
docker-run:
	docker run --rm -p 8501:8501 finsight:latest

# Remove generated artifacts (does not remove raw data)
clean:
	rm -rf data/processed/* models/* reports/figures/* reports/metrics/* mlruns/* .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
