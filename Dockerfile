FROM python:3.12-slim

WORKDIR /app

COPY requirements.render.txt ./
RUN pip install --no-cache-dir -r requirements.render.txt

COPY src/__init__.py ./src/__init__.py
COPY src/api ./src/api
COPY src/executor ./src/executor
COPY src/mcp_servers ./src/mcp_servers

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]