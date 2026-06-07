.PHONY: train up down logs

# Train the model — run this once before docker-compose up
train:
	cd model && pip install -q -r requirements.txt && python train.py

# Bring up the full pipeline (producer, processor, API, dashboard + infra)
up:
	docker compose up --build -d
	@echo ""
	@echo "  API:       http://localhost:8000/docs"
	@echo "  Dashboard: http://localhost:8501"

down:
	docker compose down -v

logs:
	docker compose logs -f processor
