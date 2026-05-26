#!/usr/bin/env bash
# Embedding service (single GPU). Usage:
#   conda activate <env-with-vllm-installed>
#   bash deploy-qwen3-embed.sh
# Or: chmod +x deploy-qwen3-embed.sh && ./deploy-qwen3-embed.sh
#
# The model directory can be overridden via environment variable
# (point this at a local clone or HF cache of Qwen3-Embedding-0.6B).
: "${EMBEDDING_MODEL_PATH:=/path/to/Qwen3-Embedding-0.6B}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# When sharing the GPU with the main model, the default 0.9 will fail because
# of insufficient free memory; lower this or use a different card
# (e.g. CUDA_VISIBLE_DEVICES=1).
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.07}"

# vLLM >= 0.19 requires --runner pooling + --convert embed (no more --task embed).
exec vllm serve "${EMBEDDING_MODEL_PATH}" \
  --runner pooling \
  --convert embed \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name Qwen3-Embedding-0.6B \
  --trust-remote-code
