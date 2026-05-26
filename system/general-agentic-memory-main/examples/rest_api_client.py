#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of the GAM REST API (FastAPI).

Before running, start the API server:
    python examples/run_api.py --port 5001
"""

import json
import os

import requests

BASE_URL = "http://localhost:5001/api/v1"


def test_add_text():
    """Add text content to a GAM via the REST API."""
    print("\n--- Add Text Content ---")
    payload = {
        "type": "text",
        "content": [
            "This is a test content for GAM.",
            "It should be organized automatically.",
        ],
        "context": "Integration test",
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "use_chunking": True,
    }

    resp = requests.post(f"{BASE_URL}/add", json=payload)
    print(f"Status : {resp.status_code}")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    return resp.json().get("gam_dir")


def test_query_text(gam_dir: str):
    """Query a GAM via the REST API."""
    print("\n--- Query GAM ---")
    payload = {
        "type": "text",
        "gam_dir": gam_dir,
        "question": "What is the content about?",
        "max_iter": 5,
        "model": "gpt-4o-mini",
    }

    resp = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status : {resp.status_code}")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


def test_add_video():
    """Add video content to a GAM via the REST API."""
    print("\n--- Add Video Content ---")
    video_dir = "/path/to/video_dir"
    if not os.path.exists(video_dir):
        print(f"Skipping video test: {video_dir} not found")
        return

    payload = {
        "type": "video",
        "input": video_dir,
        "model": "gpt-4o",
        "segmentor_model": "gpt-4o-mini",
        "caption_with_subtitles": True,
    }

    resp = requests.post(f"{BASE_URL}/add", json=payload)
    print(f"Status : {resp.status_code}")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


def test_health():
    """Check the API health endpoint."""
    print("\n--- Health Check ---")
    resp = requests.get("http://localhost:5001/")
    print(f"Status : {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    try:
        test_health()
        gam_dir = test_add_text()
        if gam_dir:
            test_query_text(gam_dir)
        # test_add_video()
    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not connect to the server. "
            "Is it running on http://localhost:5001?"
        )
