FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

# CUDA-enabled PyTorch (the wheel ships its own CUDA runtime; the host only needs the NVIDIA driver).
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126

# fasttext compiles from source, so a compiler is needed only during this layer.
COPY api/requirements.txt /tmp/requirements.txt
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && pip install -r /tmp/requirements.txt \
 && apt-get purge -y build-essential && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* /tmp/requirements.txt

RUN useradd --create-home --uid 1000 app && mkdir -p /cache/huggingface && chown app /cache/huggingface
WORKDIR /app
COPY api ./api
USER app

# The database and the models are mounted at run time (see docker-compose.yml), not baked in.
ENV METADATA_DB=/data/api/metadata.db \
    EXTRACTION_ENABLED=1 \
    MODEL_DIR=/models/Ministral-3-3B-Instruct-2512-BF16 \
    LID_MODEL=/models/lid.176.bin

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
  CMD python -c "import json,urllib.request;h=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'));raise SystemExit(h['extraction']['status'] not in ('ready','disabled'))"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
