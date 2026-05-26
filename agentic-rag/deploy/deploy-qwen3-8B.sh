#!/usr/bin/env bash
# Qwen3-8B main model served via vLLM (OpenAI-compatible Chat/Completions).
# Usage:
#   conda activate <env-with-vllm-installed>
#   bash deploy-qwen3-8B.sh
#
# The model directory can be overridden via environment variable.
: "${LLM_MODEL_PATH:=/path/to/Qwen3-8B}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Lower this when sharing the GPU with other processes (~ free_mem / total_mem).
# Use 0.9 if the card is dedicated to this service.
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.3}"

# Aligned with VLLM_MODEL_NAME / VLLM_BASE_URL defaults in .env (port 8000 / v1).
: "${VLLM_PORT:=8000}"
: "${VLLM_SERVED_MODEL_NAME:=Qwen3-8B}"

# LangGraph / LangChain may issue tool_choice=auto; vLLM must enable the tool-call
# options below or chat completions will return HTTP 400.
# Recommended parser: qwen3_xml (Qwen3); hermes also works for some Qwen variants.
: "${VLLM_TOOL_CALL_PARSER:=qwen3_xml}"

exec vllm serve "${LLM_MODEL_PATH}" \
  --served-model-name "${VLLM_SERVED_MODEL_NAME}" \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${VLLM_TOOL_CALL_PARSER}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --trust-remote-code
