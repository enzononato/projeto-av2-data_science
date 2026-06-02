.PHONY: install ingest train serve docker-build docker-run mlflow help

install:
	pip install -e ".[dev]"

# --- Dados ---
ingest:
	python -m src.data_ingestion

# --- Treinamento ---
train:
	python -m src.train

# --- Serving ---
serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# --- MLflow UI ---
mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# --- Docker ---
docker-build:
	docker build -t brasileirao-predicao:latest .

docker-run:
	docker run --env-file .env -p 8000:8000 brasileirao-predicao:latest

docker-up:
	docker compose up --build

docker-down:
	docker compose down

# --- Qualidade ---
lint:
	ruff check src/

test:
	pytest tests/ -v --cov=src

help:
	@echo "Comandos disponíveis:"
	@echo "  make install       — instala dependências"
	@echo "  make ingest        — coleta dados da API"
	@echo "  make train         — treina e registra modelos no MLflow"
	@echo "  make serve         — sobe a API de predição"
	@echo "  make mlflow        — abre o MLflow UI"
	@echo "  make docker-up     — sobe todos os serviços via Docker Compose"
