#!/bin/bash

# Long Horizontal Example - Run Script
# This example demonstrates how GAM can be used in long-horizon agent tasks
# where search results are compressed into GAM to maintain context.

# Set your API keys and base URL (override via environment variables)
export OPENAI_API_KEY="${OPENAI_API_KEY:?Please set OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"

# Directory for GAM memory storage
GAM_DIR="./gam_example_storage"

# Run the example
python "$(dirname "$0")/run.py" \
    --model "${MODEL_NAME}" \
    --api-key "${OPENAI_API_KEY}" \
    --api-base "${OPENAI_BASE_URL}" \
    --gam-dir "${GAM_DIR}"
