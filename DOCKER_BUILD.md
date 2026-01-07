# Docker Build Guide - Two-Tier System

## Problem Statement

**Before**: Every code change required pushing a 2GB+ image (PyTorch included) to Docker Hub, taking 5-10 minutes per push.

**After**: Code changes only push ~50-100MB (requirements + scripts), taking 1-2 minutes per push.

**Savings**: 95% reduction in push time for iterative development!

## Architecture

```text
┌─────────────────────────────────────┐
│   Base Image (~2GB)                 │
│   - CUDA 12.1                       │
│   - Python 3.10                     │
│   - PyTorch 2.2.0 + dependencies    │
│   - Build: RARELY (only for CUDA/  │
│     PyTorch version updates)        │
└──────────────┬──────────────────────┘
               │ FROM base
               ▼
┌─────────────────────────────────────┐
│   Application Image (~50-100MB)     │
│   - requirements.txt                │
│   - scripts/                        │
│   - Build: FREQUENTLY (every code   │
│     change during development)      │
└─────────────────────────────────────┘
```

## Username Guide

This project uses **two different registries** with different usernames:

- **GitHub Container Registry (GHCR)**: `ghcr.io` → Use GitHub username `kkuntal990` (default)
- **Docker Hub**: `docker.io` → Use Docker Hub username `kkokate990`

Current setup uses **GHCR** (GitHub Container Registry).

## Quick Start

### First Time Setup (One-time) - GHCR (Default)

```bash
# No environment variables needed - defaults are already set!
# Just run the build script:
./build_base.sh
# When prompted, type 'y' to push to GHCR

# Dockerfile is already configured:
# FROM ghcr.io/kkuntal990/ms-swift-base:latest
```

That's it! The defaults are configured for GHCR with username `kkuntal990`.

### First Time Setup (One-time) - Docker Hub (Alternative)

```bash
# To use Docker Hub instead:
export DOCKER_USERNAME=kkokate990  # Docker Hub username
export DOCKER_REGISTRY=docker.io

# Build and push base image
./build_base.sh
# When prompted, type 'y' to push to Docker Hub

# Update Dockerfile line 4:
# Change: FROM ghcr.io/kkuntal990/ms-swift-base:latest
# To: FROM kkokate990/ms-swift-base:latest
```

### Daily Development Workflow

```bash
# Make changes to your code (scripts/, requirements.txt, etc.)
# ...

# Build and push application image (takes 1-2 minutes)
./build_main.sh
# When prompted, type 'y' to push to Docker Hub

# Deploy to Kubernetes
kubectl apply -f k8s/job-sft-qwen3-8b-2gpu.yaml
```

## When to Rebuild Each Image

### Base Image (Dockerfile.base)

Rebuild ONLY when:

- ✅ Updating CUDA version
- ✅ Updating PyTorch version
- ✅ Changing Python version
- ✅ Adding system-level dependencies (apt packages)

Do NOT rebuild for:

- ❌ Code changes in scripts/
- ❌ Changes to requirements.txt
- ❌ Documentation updates

### Application Image (Dockerfile)

Rebuild OFTEN for:

- ✅ Changes to scripts/
- ✅ Updates to requirements.txt
- ✅ Any code modifications

## Build Scripts Reference

### build_base.sh

```bash
# Build base image with CUDA, Python, PyTorch
./build_base.sh

# Options:
export DOCKER_USERNAME=myusername
export BASE_TAG=v1.0  # Default: latest
./build_base.sh
```

**Output**: `<username>/ms-swift-base:latest` (~2GB)

### build_main.sh

```bash
# Build application image with your code
./build_main.sh

# Options:
export DOCKER_USERNAME=myusername
export APP_TAG=v2.3.1  # Default: latest
./build_main.sh
```

**Output**: `<username>/ms-swift-qwen:latest` (~50-100MB layer)

## Troubleshooting

### Error: "Dockerfile still contains placeholder <username>"

**Solution**: Update line 4 in Dockerfile with your Docker Hub username:

```bash
sed -i '' "s/<username>/your-actual-username/" Dockerfile
```

### Error: "Cannot find base image"

**Solution**: Build and push the base image first:

```bash
./build_base.sh
# Type 'y' when prompted to push
```

### Base image takes too long to build

**Expected**: 5-10 minutes (PyTorch is large!)
**This is normal** - you only build it once, so the time investment is worth it.

### Want to test without pushing?

```bash
# Build without pushing
./build_base.sh
# Type 'n' when prompted

# Build app image locally (will use local base)
./build_main.sh
# Type 'n' when prompted

# Test locally
docker run --rm -it --gpus all $DOCKER_USERNAME/ms-swift-qwen:latest bash
```

## Size Comparison

| Layer | Before (Single Image) | After (Two-Tier) |
|-------|----------------------|------------------|
| Base (CUDA/PyTorch) | 2.1 GB | 2.1 GB (push once) |
| Requirements | 300 MB | 300 MB |
| Scripts | 50 MB | 50 MB |
| **Total push per code change** | **2.45 GB** | **~350 MB** |
| **Time per push (typical network)** | **5-10 min** | **1-2 min** |

## Advanced Usage

### Using different PyTorch versions

Edit [Dockerfile.base](Dockerfile.base) lines 32-36:

```dockerfile
RUN pip install --no-cache-dir \
    torch==2.3.0 \          # Change version here
    torchvision==0.18.0 \   # Match with torch version
    torchaudio==2.3.0 \     # Match with torch version
    --index-url https://download.pytorch.org/whl/cu121
```

Then rebuild and push the base image:

```bash
./build_base.sh
```

### Creating versioned base images

```bash
# Tag base image with version
export BASE_TAG=cuda12.1-torch2.2.0
./build_base.sh

# Update Dockerfile to use versioned base
# Change line 4 to: FROM your-username/ms-swift-base:cuda12.1-torch2.2.0
```

### Switching between base image versions

```bash
# Development: use latest
FROM kuntalkokate/ms-swift-base:latest

# Production: use specific version
FROM kuntalkokate/ms-swift-base:cuda12.1-torch2.2.0
```

## Tips

1. **Always push base image after building** - Otherwise team members or CI/CD can't pull it
2. **Use semantic versioning for production** - Tag base images like `v1.0.0`, `v1.1.0`
3. **Document base image changes** - Keep a changelog when updating CUDA/PyTorch
4. **Cache is your friend** - Docker layers are cached locally for faster rebuilds

## Summary

This two-tier system trades a one-time 10-minute setup for 95% faster iteration during development. It's especially valuable when:

- Working on a team (everyone shares the same base image)
- Iterating quickly on code changes
- Running CI/CD pipelines (faster build times = faster feedback)
- Dealing with slow network connections (less data to push)
