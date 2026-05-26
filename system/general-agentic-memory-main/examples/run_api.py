#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run GAM REST API Server (FastAPI + Uvicorn)

Usage:
    python examples/run_api.py --port 5001
    python examples/run_api.py --model gpt-4o-mini --api-key sk-xxx
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Run GAM REST API Server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=5001,
        help="Port to bind to (default: 5001)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="",
        help="Default output root directory",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Default LLM model name",
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="Default LLM API base URL",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Default LLM API key",
    )

    args = parser.parse_args()

    generator = None
    api_key = args.api_key or os.environ.get("GAM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        from gam.generators.openai_generator import OpenAIGenerator
        from gam.generators.config import OpenAIGeneratorConfig

        config = OpenAIGeneratorConfig(
            model_name=args.model or os.environ.get("GAM_MODEL", "gpt-4o-mini"),
            base_url=args.api_base or os.environ.get("GAM_API_BASE", "https://api.openai.com/v1"),
            api_key=api_key,
        )
        generator = OpenAIGenerator(config)

    from gam.rest_api import run_server

    run_server(
        generator=generator,
        output_base=args.output_dir,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
