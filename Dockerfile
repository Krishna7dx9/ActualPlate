# Use official PyTorch image with CUDA 12.1 for GPU acceleration
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app

# Install system dependencies (git for MoGe, libgl for OpenCV)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements from the root and vision_service
COPY requirements.txt /app/requirements.txt
COPY vision_service/inference/requirements.txt /app/vision_service/inference/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r vision_service/inference/requirements.txt

# Copy the rest of the application
COPY . /app

# Make the start script executable
RUN chmod +x /app/start.sh

# Expose both ports
EXPOSE 8000 8001

# Run the startup script
CMD ["/app/start.sh"]