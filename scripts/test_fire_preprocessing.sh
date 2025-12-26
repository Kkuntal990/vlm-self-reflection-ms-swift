#!/usr/bin/env bash
#
# Local test script for FIRE preprocessing
# Tests with 10 samples to validate output format
#
set -euo pipefail

# Create local test directories
TEST_DIR="$(pwd)/test_fire_local"
OUTPUT_DIR="${TEST_DIR}/outputs"
IMAGE_DIR="${TEST_DIR}/images"

echo "========================================="
echo "FIRE Preprocessing Local Test"
echo "========================================="
echo ""
echo "Creating test directories..."
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${IMAGE_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Image dir: ${IMAGE_DIR}"
echo ""

# Run preprocessing with 10 samples
echo "Running preprocessing (10 samples)..."
echo ""
python scripts/prepare_fire_full_state_jsonl.py \
  --output_dir "${OUTPUT_DIR}" \
  --image_dir "${IMAGE_DIR}" \
  --max_samples 10 \
  --max_history_rounds 6 \
  --splits train

echo ""
echo "========================================="
echo "Validation"
echo "========================================="
echo ""

# Check outputs
echo "Output files:"
ls -lh "${OUTPUT_DIR}/"
echo ""

# Count examples
echo "Sample counts:"
wc -l "${OUTPUT_DIR}"/fire_bc_train.jsonl
echo ""

# Show statistics
echo "Statistics:"
cat "${OUTPUT_DIR}/stats.json"
echo ""

# Validate JSON format
echo "Validating JSONL format..."
python -c "
import json
import sys

jsonl_file = '${OUTPUT_DIR}/fire_bc_train.jsonl'
print(f'Reading {jsonl_file}...\n')

with open(jsonl_file, 'r') as f:
    lines = f.readlines()
    print(f'Total examples: {len(lines)}\n')

    for i, line in enumerate(lines[:3], 1):
        example = json.loads(line)
        print(f'--- Example {i} ---')
        print(f'Fields: {list(example.keys())}')
        print(f'Messages: {len(example[\"messages\"])} turns')
        print(f'Images: {len(example[\"images\"])} image(s)')

        # Validate structure
        assert 'messages' in example, 'Missing messages field'
        assert 'images' in example, 'Missing images field'
        assert len(example['messages']) == 2, 'Should have user + assistant'
        assert example['messages'][0]['role'] == 'user', 'First message should be user'
        assert example['messages'][1]['role'] == 'assistant', 'Second message should be assistant'
        assert '<image>' in example['messages'][0]['content'], 'User message should contain <image> token'
        assert len(example['images']) == 1, 'Should have exactly 1 image'

        print(f'✓ Structure valid')
        print()

print('✅ All examples have valid ms-swift format!')
"

echo ""
echo "Displaying first example:"
echo ""
head -1 "${OUTPUT_DIR}/fire_bc_train.jsonl" | python -m json.tool

echo ""
echo "========================================="
echo "Test Complete!"
echo "========================================="
echo ""
echo "Review the output above. If everything looks good,"
echo "you can run the full preprocessing on the cluster."
echo ""
echo "Test files are in: ${TEST_DIR}"
echo ""
