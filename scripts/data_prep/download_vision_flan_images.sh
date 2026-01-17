#!/bin/bash
set -euo pipefail

# Download Vision-Flan images for ALLaVA-VFLAN dataset
# Size: 34GB, Contains: ~191,000 images (191 tasks × 1,000 examples)

DOWNLOAD_DIR="${1:-/outputs/vision_flan}"
OUTPUT_DIR="${2:-/outputs/vision_flan}"

echo "========================================="
echo "Vision-Flan Image Download"
echo "========================================="
echo ""
echo "Download directory: $DOWNLOAD_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Expected size: ~34 GB"
echo ""

# Create directories
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$DOWNLOAD_DIR"

# Download the image ZIP
echo "Downloading image_191-task_1k.zip (34GB)..."
echo "This may take 30-60 minutes depending on network speed..."
echo ""
 
wget -c "https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k/resolve/main/image_191-task_1k.zip" \
    -O image_191-task_1k.zip

echo ""
echo "Download complete! Checking file..."
ls -lh image_191-task_1k.zip

# Verify download
if [ ! -f "image_191-task_1k.zip" ]; then
    echo "ERROR: Download failed!"
    exit 1
fi

FILE_SIZE=$(stat -f%z image_191-task_1k.zip 2>/dev/null || stat -c%s image_191-task_1k.zip)
echo "Downloaded size: $(numfmt --to=iec-i --suffix=B $FILE_SIZE)"

# Extract images
echo ""
echo "Extracting images..."
echo "This may take 10-20 minutes..."
echo ""

unzip -q image_191-task_1k.zip -d "$OUTPUT_DIR"

echo ""
echo "Extraction complete!"
echo ""

# Verify extraction
if [ -d "$OUTPUT_DIR/images_191task_1k" ]; then
    IMAGE_COUNT=$(find "$OUTPUT_DIR/images_191task_1k" -type f | wc -l)
    echo "✓ Extracted $IMAGE_COUNT images"
    echo "✓ Location: $OUTPUT_DIR/images_191task_1k"

    # Show sample files
    echo ""
    echo "Sample files:"
    find "$OUTPUT_DIR/images_191task_1k" -type f | head -10
else
    echo "ERROR: Extraction failed - images_191task_1k directory not found"
    exit 1
fi

echo ""
echo "========================================="
echo "Download Complete!"
echo "========================================="
echo ""
echo "Images location: $OUTPUT_DIR/images_191task_1k"
echo ""
echo "To use with ALLaVA mapping, the images should be at:"
echo "  /outputs/vision_flan/images_191task_1k/"
echo ""
echo "You can optionally remove the ZIP to save space:"
echo "  rm $DOWNLOAD_DIR/image_191-task_1k.zip"
echo ""
