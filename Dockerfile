# Using official ms-swift image from ModelScope
# This image includes: CUDA 12.8, PyTorch 2.8, Python 3.11, flash-attention, ms-swift 3.10.3, vllm, and more
# No need to build base image anymore!
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.3

# Previous two-tier system (now deprecated):
# - build_base.sh is no longer needed
# - Dockerfile.base is no longer used

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy training scripts (frequent changes - keep this layer small and last)
COPY scripts/ /workspace/scripts/

# Note: WORKDIR, ENV, and CMD inherited from base image
# Default command (will be overridden in Kubernetes Job)
CMD ["/bin/bash"]
