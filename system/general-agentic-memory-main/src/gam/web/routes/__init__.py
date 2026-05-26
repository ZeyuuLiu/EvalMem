# -*- coding: utf-8 -*-
"""
GAM Web Routes

Flask Blueprints organising the web application endpoints.
"""

from .pages import pages_bp
from .pipeline import pipeline_bp
from .browse import browse_bp
from .research import research_bp
from .video_pipeline import video_pipeline_bp
from .video_research import video_research_bp
from .long_horizontal import long_horizontal_bp
from .custom_api import custom_api_bp

__all__ = [
    "pages_bp",
    "pipeline_bp",
    "browse_bp",
    "research_bp",
    "video_pipeline_bp",
    "video_research_bp",
    "long_horizontal_bp",
    "custom_api_bp",
]
