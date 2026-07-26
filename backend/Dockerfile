FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

# Security: run as non-root
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv==0.11.16 && uv sync --frozen --no-dev --no-install-project

COPY . .
RUN mkdir -p saved_models && chown appuser:appgroup saved_models

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["/app/.venv/bin/uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
