# GPU inference and backend runtime image.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The repository root requirements file is the tracked dependency manifest.
# Do not reference a backend-specific requirements file that does not exist.
COPY requirements.txt /app/requirements.txt
COPY vision_service/inference/requirements.txt /app/vision_service/inference/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /app/requirements.txt && \
    python -m pip install --no-cache-dir -r /app/vision_service/inference/requirements.txt

COPY . /app

RUN chmod +x /app/start.sh

EXPOSE 8000 8001

CMD ["/app/start.sh"]
