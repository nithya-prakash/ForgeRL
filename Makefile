.PHONY: docker install fix-pth test lint typecheck bench train compare dashboard mlflow demo
install:
	uv sync
	-chflags nohidden .venv/lib/python3.12/site-packages/*.pth   # macOS: hidden .pth files are skipped
test:
	uv run pytest -q
lint:
	uv run ruff check src tests && uv run ruff format --check src tests
typecheck:
	uv run mypy src
bench:
	uv run factory-scheduler benchmark --n-seeds 20
train:
	uv run factory-scheduler train
compare:
	uv run factory-scheduler compare --n-seeds 20
dashboard:
	uv run streamlit run src/factory_scheduler/dashboard/app.py
mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
demo: bench
	uv run factory-scheduler cpsat --n-seeds 3
docker:   ## dashboard on :8501 and MLflow UI on :5000 (touch avoids Docker creating a directory)
	touch mlflow.db && docker compose up --build
