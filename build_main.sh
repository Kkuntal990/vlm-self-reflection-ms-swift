#!/bin/bash
# Build and push main application image (requirements + scripts)
# This should be run FREQUENTLY - every time you change scripts or requirements

set -e

# Configuration
# Defaults to GHCR (GitHub Container Registry)
DOCKER_USERNAME="${DOCKER_USERNAME:-kkuntal990}"  # GitHub username
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"  # ghcr.io or docker.io
APP_IMAGE_NAME="ms-swift-qwen"
APP_TAG="${APP_TAG:-latest}"

# Build full image name with registry
if [ "$DOCKER_REGISTRY" = "docker.io" ]; then
    FULL_IMAGE_NAME="${DOCKER_USERNAME}/${APP_IMAGE_NAME}:${APP_TAG}"
else
    FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${APP_IMAGE_NAME}:${APP_TAG}"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Building Application Image ===${NC}"
echo "Registry: ${DOCKER_REGISTRY}"
echo "Image: ${FULL_IMAGE_NAME}"
echo ""

# Check if DOCKER_USERNAME is set
if [ "$DOCKER_USERNAME" = "<username>" ]; then
    echo -e "${RED}ERROR: Please set DOCKER_USERNAME environment variable${NC}"
    echo "Example: export DOCKER_USERNAME=your-username"
    exit 1
fi

# Check if Dockerfile has been updated with correct base image
BASE_IMAGE_LINE=$(grep "^FROM" Dockerfile | head -1)
if echo "$BASE_IMAGE_LINE" | grep -q "<username>"; then
    echo -e "${RED}ERROR: Dockerfile still contains placeholder <username>${NC}"
    echo "Please update line 4 in Dockerfile to use your actual base image"
    exit 1
fi

# Build application image
echo -e "${GREEN}Building application image for linux/amd64 (this should be fast - 1-2 minutes)...${NC}"
docker build \
    --platform linux/amd64 \
    -t "${FULL_IMAGE_NAME}" \
    .

echo ""
echo -e "${GREEN}Application image built successfully!${NC}"
echo ""

# Show image sizes
echo -e "${YELLOW}Image sizes:${NC}"
docker images | grep -E "REPOSITORY|ms-swift" | head -5

echo ""

# Ask if user wants to push
REGISTRY_NAME="Docker Hub"
if [ "$DOCKER_REGISTRY" != "docker.io" ]; then
    REGISTRY_NAME="${DOCKER_REGISTRY}"
fi

read -p "Push to ${REGISTRY_NAME}? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Pushing application image to ${REGISTRY_NAME}...${NC}"
    docker push "${FULL_IMAGE_NAME}"
    echo ""
    echo -e "${GREEN}✓ Application image pushed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Update your Kubernetes manifests to use: ${FULL_IMAGE_NAME}"
    echo "2. Deploy: kubectl apply -f k8s/job-*.yaml"
else
    echo -e "${YELLOW}Skipping push. You can push later with:${NC}"
    echo "docker push ${FULL_IMAGE_NAME}"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
