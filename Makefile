.PHONY: setup data train serve test docker-up docker-down score

setup:
	pip install -r requirements.txt

data:
	python -m src.data.generate_synthetic

train: data
	python -m src.training.train

serve:
	uvicorn src.serving.api:app --reload --port 8000

test:
	pytest tests/ -v

docker-up:
	docker compose -f docker/docker-compose.yml up --build -d
	@echo ""
	@echo "Services started:"
	@echo "  API    → http://localhost:8000"
	@echo "  Docs   → http://localhost:8000/docs"
	@echo "  MLflow → http://localhost:5000"

docker-down:
	docker compose -f docker/docker-compose.yml down

score:
	@curl -s -X POST http://localhost:8000/score \
		-H "Content-Type: application/json" \
		-d '{"amount":850,"hour":2,"day_of_week":5,"distance_from_home_km":120,"distance_from_last_transaction_km":80,"used_chip":0,"used_pin":0,"online_order":1,"merchant_category":"electronics","merchant_age_days":45,"repeat_merchant":0}' \
		| python3 -m json.tool
