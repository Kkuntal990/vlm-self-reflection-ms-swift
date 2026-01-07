# Two-tier image system: Use pre-built base image with CUDA, Python, and PyTorch
# Using GitHub Container Registry (ghcr.io)
# GitHub username: kkuntal990 | Docker Hub username: kkokate990
FROM ghcr.io/kkuntal990/ms-swift-base:latest

# If base image not available, build it first:
#   ./build_base.sh
# (defaults to GHCR)

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy training scripts (frequent changes - keep this layer small and last)
COPY scripts/ /workspace/scripts/

# Note: WORKDIR, ENV, and CMD inherited from base image
# Default command (will be overridden in Kubernetes Job)
CMD ["/bin/bash"]
