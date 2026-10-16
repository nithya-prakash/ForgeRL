FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# dependencies first so source edits do not invalidate the heavy layer (CPU-only torch on Linux)
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" MLFLOW_DISABLE_AGENT_HINT=1
CMD ["streamlit", "run", "src/factory_scheduler/dashboard/app.py", "--server.address=0.0.0.0"]
