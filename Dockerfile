FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY README.md .

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.training.evaluate"]
