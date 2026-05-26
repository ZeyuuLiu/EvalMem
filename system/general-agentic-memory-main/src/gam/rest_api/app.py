# -*- coding: utf-8 -*-
"""
GAM REST API Application

A standalone FastAPI server for programmatic access to GAM.
Provides auto-generated OpenAPI documentation at /docs (Swagger UI)
and /redoc (ReDoc).
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import routes

DEFAULT_OUTPUT_BASE = os.environ.get(
    "GAM_OUTPUT_BASE",
    "/share/project/chaofan/code/memory/gam/examples/output/rest_api",
)


def create_app(
    generator=None,
    video_generator=None,
    output_base: str = "",
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        generator: Default LLM generator instance (optional; can be overridden per-request).
        video_generator: Multimodal LLM for video inspection (optional).
        output_base: Default output root directory.
    """
    app = FastAPI(
        title="GAM REST API",
        description=(
            "General Agentic Memory — RESTful API for building and querying "
            "hierarchical knowledge bases from text and video."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    routes.configure(
        generator=generator,
        video_generator=video_generator,
        output_base=output_base or DEFAULT_OUTPUT_BASE,
    )

    app.include_router(routes.router)

    @app.get("/", tags=["Health"])
    async def health_check():
        return {
            "service": "GAM REST API",
            "version": "1.0.0",
            "docs": "/docs",
        }

    return app


def run_server(
    generator=None,
    video_generator=None,
    output_base: str = "",
    host: str = "0.0.0.0",
    port: int = 5001,
):
    """
    Launch the GAM REST API server.

    Args:
        generator: Default LLM generator (optional).
        video_generator: Multimodal LLM for video inspection (optional).
        output_base: Default output root directory.
        host: Bind address.
        port: Bind port.
    """
    import uvicorn

    app = create_app(
        generator=generator,
        video_generator=video_generator,
        output_base=output_base,
    )

    actual_output_base = output_base or DEFAULT_OUTPUT_BASE

    print(f"\n{'=' * 60}")
    print(f"GAM REST API Server (FastAPI + Uvicorn)")
    print(f"{'=' * 60}")
    print()
    print(f"  Base URL : http://{host}:{port}")
    print(f"  Docs     : http://{host}:{port}/docs")
    print(f"  ReDoc    : http://{host}:{port}/redoc")
    print()
    print(f"  Endpoints:")
    print(f"    POST /api/v1/add    — Add content to a GAM")
    print(f"    POST /api/v1/query  — Query an existing GAM")
    print()
    print(f"  Output   : {actual_output_base}/")
    print()
    if generator:
        print(f"  LLM default configured (can be overridden per-request)")
    else:
        print(f"  No default LLM — provide model/api_key in each request")
    print()
    print(f"  Press Ctrl+C to stop.")
    print(f"{'=' * 60}\n")

    uvicorn.run(app, host=host, port=port)
