FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/luel-du/adult-income-analysis" \
      org.opencontainers.image.description="Adult income analysis with pandas and polars"

# PYTHONUNBUFFERED: show output as it is printed instead of when the container exits
# PYTHONDONTWRITEBYTECODE: no __pycache__ folders inside the image
# MPLCONFIGDIR: a writable cache folder for matplotlib, whichever user the container runs as
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# dependencies first: this layer is rebuilt only when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY tests ./tests

# do not run as root; data/ and figures/ are the only folders the analysis writes to
RUN useradd --create-home appuser \
    && mkdir -p data figures \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "src/main.py"]
