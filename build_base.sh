#!/bin/bash
# Build and push base image with stable dependencies (CUDA, Python, PyTorch)
# This should be run RARELY - only when updating CUDA/PyTorch versions

set -e

# Configuration
# Defaults to GHCR (GitHub Container Registry)
DOCKER_USERNAME="${DOCKER_USERNAME:-kkuntal990}"  # GitHub username
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"  # ghcr.io or docker.io
BASE_IMAGE_NAME="ms-swift-base"
BASE_TAG="${BASE_TAG:-latest}"

# Build full image name with registry
if [ "$DOCKER_REGISTRY" = "docker.io" ]; then
    FULL_IMAGE_NAME="${DOCKER_USERNAME}/${BASE_IMAGE_NAME}:${BASE_TAG}"
else
    FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${BASE_IMAGE_NAME}:${BASE_TAG}"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Building Base Image ===${NC}"
echo "Registry: ${DOCKER_REGISTRY}"
echo "Image: ${FULL_IMAGE_NAME}"
echo ""

# Check if DOCKER_USERNAME is set
if [ "$DOCKER_USERNAME" = "<username>" ]; then
    echo -e "${RED}ERROR: Please set DOCKER_USERNAME environment variable${NC}"
    echo "Example: export DOCKER_USERNAME=your-username"
    exit 1
fi

# Build base image
echo -e "${GREEN}Building base image for linux/amd64 (this will take 5-10 minutes)...${NC}"
docker build \
    --platform linux/amd64 \
    -f Dockerfile.base \
    -t "${FULL_IMAGE_NAME}" \
    .

echo ""
echo -e "${GREEN}Base image built successfully!${NC}"
echo ""

# Ask if user wants to push
REGISTRY_NAME="Docker Hub"
if [ "$DOCKER_REGISTRY" != "docker.io" ]; then
    REGISTRY_NAME="${DOCKER_REGISTRY}"
fi

read -p "Push to ${REGISTRY_NAME}? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Pushing base image to ${REGISTRY_NAME}...${NC}"
    docker push "${FULL_IMAGE_NAME}"
    echo ""
    echo -e "${GREEN}✓ Base image pushed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Update Dockerfile line 4 to use: FROM ${FULL_IMAGE_NAME}"
    echo "2. Run ./build_main.sh to build your application image"
else
    echo -e "${YELLOW}Skipping push. You can push later with:${NC}"
    echo "docker push ${FULL_IMAGE_NAME}"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
