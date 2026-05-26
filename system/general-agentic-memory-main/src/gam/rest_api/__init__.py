# -*- coding: utf-8 -*-
"""
GAM REST API Module

A FastAPI-based REST API for programmatic access to GAM.
"""

from .app import create_app, run_server

__all__ = ["create_app", "run_server"]
