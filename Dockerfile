FROM python:3.10.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app \
    XDG_CACHE_HOME=/home/app/.cache

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/app app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[graph]" \
    && mkdir -p /home/app/.cache \
    && chown -R app:app /home/app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["python", "-m", "uvicorn", "enterprise_knowledge_agent.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--timeout-keep-alive", "5", "--timeout-graceful-shutdown", "30", \
     "--no-access-log"]
