# syntax=docker/dockerfile:1.7

############################################
# Builder: compile TA-Lib from source
############################################
FROM python:3.12-slim AS ta_builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget -q https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make -j"$(nproc)" \
    && make install

############################################
# Runtime image
############################################
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl tini \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib runtime artifacts from builder
COPY --from=ta_builder /usr/lib/libta_lib* /usr/lib/
COPY --from=ta_builder /usr/include/ta-lib /usr/include/ta-lib

WORKDIR /app

# Install Python dependencies used by both services
RUN pip install --no-cache-dir \
    pandas numpy requests yfinance \
    scikit-learn statsmodels schedule joblib \
    tensorflow backtrader \
    transformers torch \
    ta-lib

# App code + data seed + production model seed
# (build context is repository root; docker-compose points dockerfile to research/Dockerfile)
COPY research/train_pipeline.py /app/train_pipeline.py
COPY research/trading_engine.py /app/trading_engine.py
COPY research/notifications.py /app/notifications.py
COPY research/preflight_warmup.py /app/preflight_warmup.py
COPY research/phase6_lpa_enriched.csv /app/phase6_lpa_enriched.csv
COPY production/models/ /app/models/

# Runtime dirs and non-root user
RUN mkdir -p /app/models /app/logs /app/data \
    && useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
ENTRYPOINT ["/usr/bin/tini", "--"]

