# Build:  docker build -t mobile-addiction-app .
# Run:    docker run -p 8501:8501 mobile-addiction-app
# The Python version must support the package versions in requirements.txt
# (change it with --build-arg PYTHON_VERSION=3.13).
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install only what the app needs at run time, using the versions pinned in requirements.txt
COPY requirements.txt .
RUN tr -d '\r' < requirements.txt \
      | grep -iE '^(streamlit|pandas|numpy|scikit-learn|joblib)([=<>~! ]|$)' > requirements-app.txt \
    && pip install -r requirements-app.txt

COPY app.py .
COPY .streamlit/ .streamlit/
COPY models/best_model.pkl models/feature_schema.json models/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
