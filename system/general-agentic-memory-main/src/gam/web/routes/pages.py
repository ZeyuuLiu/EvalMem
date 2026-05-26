# -*- coding: utf-8 -*-
"""
Page routes – serve the main HTML pages.
"""

from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    """首页 - 选择 Text 或 Video 平台"""
    return render_template("index.html")


@pages_bp.route("/text")
def text():
    """Text GAM Platform 主页面"""
    return render_template("text_platform.html")


@pages_bp.route("/video")
def video():
    """Video GAM Platform 主页面"""
    return render_template("video_platform.html")


@pages_bp.route("/long-horizontal")
def long_horizontal():
    """Long Horizontal GAM Platform 主页面"""
    return render_template("long_horizontal_platform.html")