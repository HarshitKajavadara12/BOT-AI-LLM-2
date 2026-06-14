# QUANTUM-FORGE Production Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Python dependencies and R environment
FROM python:3.11-slim as python-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gnupg2 \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install R
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create application user
RUN useradd --create-home --shell /bin/bash quantum
WORKDIR /app
RUN chown quantum:quantum /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: R package installation
FROM python-base as r-packages

# Install R packages
COPY infrastructure/scripts/install_r_packages.R /tmp/
RUN Rscript /tmp/install_r_packages.R

# Stage 3: Application build
FROM r-packages as app-build

# Copy application code
COPY --chown=quantum:quantum . .

# Install Python package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs data/{raw,processed,features,models} config backtest_results

# Stage 4: Production runtime
FROM app-build as production

# Switch to application user
USER quantum

# Set environment variables
ENV ENVIRONMENT=production \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    TRADING_MODE=live

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["python", "-m", "uvicorn", "interface.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 5: Development runtime
FROM app-build as development

# Install development dependencies
RUN pip install \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    jupyter \
    ipykernel

# Switch to application user
USER quantum

# Set development environment
ENV ENVIRONMENT=development \
    PYTHONPATH=/app \
    LOG_LEVEL=DEBUG \
    TRADING_MODE=paper

# Development command with hot reload
CMD ["python", "-m", "uvicorn", "interface.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]