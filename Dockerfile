FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Copy training scripts
COPY scripts/ /workspace/scripts/

# Set environment variables for cache directories
# These will be overridden by PVC mounts in Kubernetes
ENV HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf/transformers \
    HF_DATASETS_CACHE=/cache/hf/datasets \
    TORCH_HOME=/cache/torch \
    PIP_CACHE_DIR=/cache/pip \
    OUTPUT_DIR=/outputs

# Default command (will be overridden in Kubernetes Job)
CMD ["/bin/bash"]
