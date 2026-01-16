# GitHub Container Registry (GHCR) Setup

Your Kubernetes manifests use **ghcr.io** (GitHub Container Registry) instead of Docker Hub. Here's how to use the two-tier system with GHCR.

## Quick Start

```bash
# 1. Set environment variables for GHCR
export DOCKER_USERNAME=kkuntal990  # GitHub username (not Docker Hub username!)
export DOCKER_REGISTRY=ghcr.io

# 2. Login to GHCR (one-time)
echo $GITHUB_TOKEN | docker login ghcr.io -u $DOCKER_USERNAME --password-stdin
# Or create a Personal Access Token: https://github.com/settings/tokens
# Required scopes: write:packages, read:packages, delete:packages

# 3. Build and push base image (once)
./build_base.sh
# When prompted, type 'y' to push

# 4. Build and push application image (frequently)
./build_main.sh
# When prompted, type 'y' to push
```

## Important: Username Guide

You have **two different usernames** for different registries:

- **GitHub username**: `kkuntal990` (use for ghcr.io)
- **Docker Hub username**: `kkokate990` (use for docker.io)

Always set `DOCKER_USERNAME` based on which registry you're using:

```bash
# For GHCR (GitHub Container Registry)
export DOCKER_USERNAME=kkuntal990
export DOCKER_REGISTRY=ghcr.io

# For Docker Hub
export DOCKER_USERNAME=kkokate990
export DOCKER_REGISTRY=docker.io
```

## Two Images, One Pull

When Kubernetes pulls `ghcr.io/kkuntal990/ms-swift-qwen:latest`:

1. **K8s pulls application image** from ghcr.io
2. **Docker sees**: `FROM ghcr.io/kkokate990/ms-swift-base:latest` in the Dockerfile
3. **Docker auto-pulls base image** if not cached
4. **Result**: Both images available, only referenced one in K8s YAML

## Your K8s Manifests Are Fine!

Your manifests only need ONE image reference:

```yaml
spec:
  containers:
  - name: fire-preprocessor
    image: ghcr.io/kkuntal990/ms-swift-qwen:latest  # ✅ Only this!
    imagePullPolicy: Always
```

The base image (`ghcr.io/kkuntal990/ms-swift-base:latest`) is pulled automatically as a dependency.

## Image Names

After building, you'll have:

```text
Base Image (push once):
  ghcr.io/kkuntal990/ms-swift-base:latest (~2GB)

Application Image (push often):
  ghcr.io/kkuntal990/ms-swift-qwen:latest (~350MB total, only ~50-100MB new layer)
```

## Switching Registries

### Use Docker Hub instead of GHCR

```bash
# Build for Docker Hub (use Docker Hub username)
export DOCKER_USERNAME=kkokate990
export DOCKER_REGISTRY=docker.io
./build_base.sh
./build_main.sh

# Update Dockerfile line 4:
FROM kkokate990/ms-swift-base:latest

# Update K8s manifests:
image: kkokate990/ms-swift-qwen:latest
```

### Use GHCR (current setup)

```bash
# Build for GHCR (use GitHub username)
export DOCKER_USERNAME=kkuntal990
export DOCKER_REGISTRY=ghcr.io
./build_base.sh
./build_main.sh

# Dockerfile already set:
FROM ghcr.io/kkuntal990/ms-swift-base:latest

# K8s manifests already set:
image: ghcr.io/kkuntal990/ms-swift-qwen:latest
```

## Troubleshooting

### Error: "denied: permission_denied"

**Solution**: Create a GitHub Personal Access Token with `write:packages` scope:

1. Go to: https://github.com/settings/tokens
2. Create token with scopes: `write:packages`, `read:packages`, `delete:packages`
3. Login: `echo $GITHUB_TOKEN | docker login ghcr.io -u $DOCKER_USERNAME --password-stdin`

### Error: "manifest unknown"

**Solution**: Push the base image first:

```bash
export DOCKER_REGISTRY=ghcr.io
./build_base.sh
# Type 'y' to push
```

### Username mismatch errors

**Remember**: Use the correct username for each registry:

- **GHCR** (ghcr.io): Use GitHub username `kkuntal990`
- **Docker Hub** (docker.io): Use Docker Hub username `kkokate990`

Always set `DOCKER_USERNAME` and `DOCKER_REGISTRY` together before building!

## Benefits Recap

With the two-tier system:

| Action | Before | After |
|--------|--------|-------|
| Change script file | Push 2GB | Push 50MB |
| Update requirements.txt | Push 2GB | Push 350MB |
| Update CUDA/PyTorch | Push 2GB | Push 2GB (rare!) |

**Result**: 95% faster iterations during development!
