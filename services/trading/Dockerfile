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
# Runtime image — Trading service
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

# Install Python dependencies from service-specific requirements file
COPY services/trading/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy service source files
# train_pipeline.py and preflight_warmup.py are included because
# trading_engine.py uses --run-preflight which invokes the training pipeline.
COPY services/trading/trading_engine.py   /app/trading_engine.py
COPY services/trading/preflight_warmup.py /app/preflight_warmup.py
COPY services/trading/train_pipeline.py   /app/train_pipeline.py
COPY services/trading/notifications.py   /app/notifications.py

# Runtime dirs for the mounted shared volume
RUN mkdir -p /app/shared/models /app/shared/data /app/shared/logs \
    && useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
ENTRYPOINT ["/usr/bin/tini", "--"]
