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
# Runtime image — MLOps service
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
COPY services/mlops/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy service source files
COPY services/mlops/train_pipeline.py   /app/train_pipeline.py
COPY services/mlops/preflight_warmup.py /app/preflight_warmup.py
COPY services/mlops/notifications.py   /app/notifications.py
COPY services/mlops/gather_openbb_extended_data.py /app/gather_openbb_extended_data.py
COPY services/mlops/deeper_research_with_skills.py /app/deeper_research_with_skills.py
COPY services/mlops/phase1_panel_fe.py              /app/phase1_panel_fe.py
COPY services/mlops/phase2_var_granger.py           /app/phase2_var_granger.py
COPY services/mlops/phase3_ml_prediction.py         /app/phase3_ml_prediction.py
COPY services/mlops/phase4_preprocessing.py         /app/phase4_preprocessing.py
COPY services/mlops/phase4_lstm_model.py            /app/phase4_lstm_model.py
COPY services/mlops/phase4_event_study.py           /app/phase4_event_study.py
COPY services/mlops/phase5_data_repair_rectangularize.py /app/phase5_data_repair_rectangularize.py
COPY services/mlops/phase6_finbert_lpa.py           /app/phase6_finbert_lpa.py
COPY services/mlops/phase7_lstm_volatility.py       /app/phase7_lstm_volatility.py

# Runtime dirs for the mounted shared volume
RUN mkdir -p /app/shared/models /app/shared/data /app/shared/logs \
    && useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
ENTRYPOINT ["/usr/bin/tini", "--"]

