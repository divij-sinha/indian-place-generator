FROM ghcr.io/astral-sh/uv:python3.11-bookworm

ENV UV_SYSTEM_PYTHON=1
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8080"]
