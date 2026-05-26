# Examples

This directory contains usage guides, runnable examples, and server launchers for GAM.

## Directory Structure

```
examples/
├── docs/                     # Usage guides
│   ├── sdk_usage.md          # Python SDK (Workflow API) usage
│   ├── cli_usage.md          # CLI tools (gam-add / gam-request) usage
│   ├── rest_api_usage.md     # REST API endpoints & integration
│   └── web_usage.md          # Web platform setup & features
│
├── long_text/                # Long-text GAM: build & QA
│   ├── add.py                # Build text GAM from documents
│   └── request.py            # Query an existing text GAM
│
├── long_video/               # Long-video GAM: build & QA
│   ├── add.py                # Build video GAM from a video directory
│   └── request.py            # Query an existing video GAM
│
├── long_horizon/             # Long-horizon agent trajectory compression
│   ├── run.py                # Multi-round agent demo with search/memorize/recall
│   └── run.sh                # Shell wrapper with env-var configuration
│
├── rest_api_client.py        # Example REST API client (requests-based)
├── run_api.py                # Launch REST API server (FastAPI + Uvicorn)
└── run_web.py                # Launch Web platform (Flask)
```

## Quick Start

### 1. Environment Setup

Configure LLM credentials via environment variables so all examples can share them:

```bash
export GAM_API_KEY="sk-your-api-key"
export GAM_MODEL="gpt-4o-mini"
export GAM_API_BASE="https://api.openai.com/v1"
```

### 2. Task-specific Examples

#### Long Text

```bash
# Build GAM from a PDF / text file
python examples/long_text/add.py paper.pdf ./output/my_gam

# Query the GAM
python examples/long_text/request.py ./output/my_gam "What is the main conclusion?"
```

#### Long Video

```bash
# Build GAM from a video directory (containing video.mp4 + optional subtitles.srt)
python examples/long_video/add.py ./video_dir ./output/my_video_gam

# Query the video GAM
python examples/long_video/request.py ./output/my_video_gam "What happens in the video?"
```

#### Long-Horizon (Agent Trajectory)

```bash
# Run the multi-round agent demo that uses search → memorize → recall
python examples/long_horizon/run.py --model gpt-4o-mini

# Or use the shell wrapper
bash examples/long_horizon/run.sh
```

### 3. Servers

```bash
# REST API server
python examples/run_api.py --port 5001

# Web platform
python examples/run_web.py --port 5000
```

### 4. REST API Client

```bash
# Start the server first, then run the example client
python examples/rest_api_client.py
```

## Documentation

| Guide | Description |
|---|---|
| [SDK Usage](docs/sdk_usage.md) | `Workflow` API, low-level components, and API reference |
| [CLI Usage](docs/cli_usage.md) | `gam-add` and `gam-request` command-line tools |
| [REST API Usage](docs/rest_api_usage.md) | FastAPI endpoints, request/response formats |
| [Web Usage](docs/web_usage.md) | Web platform installation and features |
