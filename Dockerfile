# Pin the CUDA-enabled PyTorch runtime to a release compatible with NumPy 2.
FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep the CPU/backend dependency set separate from the GPU inference stack.
COPY requirements/backend.txt /app/requirements/backend.txt
COPY vision_service/inference/requirements.txt /app/vision_service/inference/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /app/requirements/backend.txt && \
    python -m pip install --no-cache-dir -r /app/vision_service/inference/requirements.txt && \
    python -m pip check

COPY . /app

RUN chmod +x /app/start.sh

EXPOSE 8000 8001

CMD ["/app/start.sh"]
