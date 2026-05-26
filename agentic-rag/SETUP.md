# Agentic RAG: Environment & Deployment

This sub-project deploys an **Agentic RAG** stack used by the high-recall encoding examiner. The components are:

| Component | Choice | Notes |
|-----------|--------|-------|
| Main LLM | **vLLM** (OpenAI-compatible API) | Local Qwen3-8B / Qwen3-32B weights |
| Embedding | **Qwen3-Embedding-0.6B** | Local load or vLLM pooling service |
| Vector store | **Qdrant** | Default HTTP port `6333` |
| Agent runtime | **LangChain + LangGraph** | See `agentic_rag/` |

> The cache directory naming `Qwen3-Embedding-0___6B` is the standard Hugging Face cache layout (`.` replaced by `___`); pass the directory as a local path.

---

## 1. Prerequisites

### Hardware / system

- **GPU**: Qwen3-8B served with vLLM benefits from **>=16 GB VRAM** (FP16/BF16). Use quantization or a smaller `max-model-len` if VRAM is tight.
- **CUDA / driver**: must be compatible with the **PyTorch / vLLM** wheel versions you install. Verify with `nvidia-smi`.
- **Disk**: keep large weights and caches on a data disk; **do not** rely on the home / system drive.

### Software (suggested versions)

| Package | Notes |
|---------|-------|
| Python | **3.10+** (3.11 works well) |
| Conda | Optional; point `pkgs_dirs` and `--prefix` at a data disk |
| Docker | Optional; convenient for running Qdrant |
| transformers | **>=4.51.0** (required for Qwen3, otherwise `KeyError: 'qwen3'`) |
| sentence-transformers | **>=3.0.0** |

### Services & ports (defaults)

| Service | Port | Purpose |
|---------|------|---------|
| vLLM OpenAI API | **8000** | Chat / Completion for `ChatOpenAI` |
| Qdrant REST | **6333** | Vector retrieval |
| Qdrant gRPC | 6334 | Optional |

### Environment variables

Set the following in `~/.bashrc` or in a project-level `env` file (adjust to your machine):

```bash
# Keep caches on a data disk
export HF_HOME=${HOME}/.cache/huggingface
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PIP_CACHE_DIR=${HOME}/.cache/pip
```

The application reads `env.example`-style variables (see `env.example`):

- `VLLM_BASE_URL` (default `http://127.0.0.1:8000/v1`)
- `EMBEDDING_MODEL_PATH` (local Qwen3-Embedding directory)
- `QDRANT_URL` (default `http://127.0.0.1:6333`)
- `EMBEDDING_DIM` (`1024` for Qwen3-Embedding-0.6B full dim; pin to the value used at index time)

---

## 2. Deployment

### Step 1: Conda environment

```bash
conda create -n agentic-rag python=3.11 -y
conda activate agentic-rag
```

If you keep packages on a data disk:

```bash
conda config --add pkgs_dirs /path/to/conda-pkgs
```

### Step 2: vLLM

Pick a wheel matching your CUDA version. **Do not** mix this install with arbitrary torch versions. See https://docs.vllm.ai for installation.

### Step 3: Verify

```bash
python -c "import vllm; print(vllm.__version__)"
nvidia-smi
```

### Step 4: Launch vLLM (Qwen3-8B)

Set `MODEL_PATH` to the directory containing the weights:

```bash
export MODEL_PATH=/path/to/Qwen3-8B

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3-8B \
  --dtype auto \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000
```

Notes:

- `--served-model-name` is the `model` field used by OpenAI clients; it must match `VLLM_MODEL_NAME`.
- Lower `--max-model-len` (e.g. `4096`) when VRAM is tight.
- Multi-GPU: add `--tensor-parallel-size N`.

Verify:

```bash
curl http://127.0.0.1:8000/v1/models
```

### Step 5: Launch Qdrant

**Option A: Docker (recommended)**

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v ${HOME}/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Option B: Binary release**

Download the pre-built binary from the [Qdrant releases page](https://github.com/qdrant/qdrant/releases) (prefer the `musl` build on x86_64 Linux for static linking against glibc):

```bash
export QDRANT_ROOT="${QDRANT_ROOT:-${HOME}/agentic-rag/opt/qdrant}"
export QDRANT_STORAGE="${QDRANT_STORAGE:-${HOME}/agentic-rag/qdrant_storage}"
mkdir -p "$QDRANT_ROOT" "$QDRANT_STORAGE"
cd "$QDRANT_ROOT"

export QDRANT_VER="v1.17.1"
curl -fL -O "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-musl.tar.gz"
tar -xzf "qdrant-x86_64-unknown-linux-musl.tar.gz"

export QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE"
export QDRANT__SERVICE__HOST="${QDRANT__SERVICE__HOST:-0.0.0.0}"

./qdrant
```

For long-running deployments, run inside `tmux` / `screen` or via `nohup ... &`. ARM hosts should fetch `qdrant-aarch64-unknown-linux-musl.tar.gz`.

If `./qdrant` fails with `GLIBC_2.xx not found`, you downloaded the `gnu` build but the system `glibc` is too old; switch to the `musl` build.

Verify: `curl -s http://127.0.0.1:6333/collections`.

### Step 6: Install application dependencies

In the agent environment (can be the same as the vLLM env or separate):

```bash
cd <repo-root>/agentic-rag
pip install -r requirements.txt
```

This sub-project ships **LangGraph ReAct + retrieval tools** (see `agentic_rag/` and `main.py`). Configure `.env` and run:

```bash
cp env.example .env   # edit paths and ports
python main.py ingest sample_docs --recreate
python main.py chat
```

See `README.md` for the user-facing interface.

### Step 7: Embedding (Qwen3-Embedding-0.6B)

A separate embedding process is **not required**. The recommended path is to load the local model directly inside the indexing / retrieval process using `SentenceTransformer`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("/path/to/Qwen3-Embedding-0.6B")
# For retrieval queries, follow the model card and use the appropriate prompt
emb = model.encode(["example text"], prompt_name="query")
```

- If GPU memory is contended with vLLM, use `device="cpu"` or `cuda:1`.
- Create the Qdrant collection with `vector_size=1024` (the Qwen3-Embedding-0.6B default full dim). The dimension at index time and query time must match.

### Step 8: LangChain / LangGraph integration

1. Chat model: `ChatOpenAI` (`langchain-openai`) with `base_url=http://127.0.0.1:8000/v1`, `api_key="EMPTY"`, `model="Qwen3-8B"` (matching `--served-model-name`).
2. Vector store: `langchain_qdrant.Qdrant` + `QdrantClient(url=...)`, using a `SentenceTransformer`-backed embedding (or `HuggingFaceEmbeddings`).
3. Agentic RAG: orchestrate retrieval -> (optional reranker) -> tool-calling LLM with LangGraph; tools may include "re-retrieve", "calculator", etc.

---

## 3. Troubleshooting

1. `KeyError: 'qwen3'` — upgrade `transformers` to `>=4.51.0`.
2. vLLM OOM — lower `max-model-len`, use a quantized weight (e.g. AWQ), or enable tensor parallelism.
3. Qdrant dimension mismatch — drop the collection and recreate it with `vector_size` equal to the embedding output dimension.
4. System drive full — make sure `HF_HOME`, `PIP_CACHE_DIR`, conda `pkgs_dirs`, and the conda env `--prefix` all live on a data disk.

---

## 4. Next steps

Copy the environment template:

```bash
cp env.example .env
# edit .env, then load via python-dotenv in code
```
