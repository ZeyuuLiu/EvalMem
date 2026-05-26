# -*- coding: utf-8 -*-
"""
Browse API routes – directory listing, file reading, GAM tree inspection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from flask import Blueprint, current_app, jsonify, request

from ...core.tree import GAMTree
from ...workspaces.local_workspace import LocalWorkspace

browse_bp = Blueprint("browse", __name__)


# ---------------------------------------------------------------------------
# /api/list_dir
# ---------------------------------------------------------------------------

@browse_bp.route("/api/list_dir")
def api_list_dir():
    """列出目录内容（用于文件夹浏览器）"""
    try:
        dir_path = request.args.get("path", "").strip()
        if not dir_path:
            dir_path = current_app.config["OUTPUT_BASE"]

        dir_path = Path(dir_path)
        if not dir_path.exists():
            return jsonify({"error": "Directory not found"})
        if not dir_path.is_dir():
            return jsonify({"error": "Not a directory"})

        items = []
        for item in sorted(dir_path.iterdir()):
            if item.name.startswith("."):
                continue
            items.append({"name": item.name, "is_dir": item.is_dir()})

        return jsonify({"path": str(dir_path), "items": items})

    except PermissionError:
        return jsonify({"error": "Permission denied"})
    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------------------------------------------------------------------
# /api/recent_sessions
# ---------------------------------------------------------------------------

@browse_bp.route("/api/recent_sessions")
def api_recent_sessions():
    """获取最近的会话列表（从默认输出目录扫描）"""
    try:
        base_path = Path(current_app.config["OUTPUT_BASE"])
        sessions = []

        if base_path.exists():
            for item in sorted(base_path.iterdir(), reverse=True):
                if item.is_dir() and not item.name.startswith("."):
                    chunk_dir = item / "chunks"
                    gam_dir = item / "gam"

                    session_info = {
                        "name": item.name,
                        "path": str(item),
                        "chunk_dir": str(chunk_dir) if chunk_dir.exists() else None,
                        "gam_dir": str(gam_dir) if gam_dir.exists() else None,
                    }

                    if session_info["chunk_dir"] or session_info["gam_dir"]:
                        sessions.append(session_info)

                    if len(sessions) >= 10:
                        break

        return jsonify({"success": True, "sessions": sessions})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "sessions": []})


# ---------------------------------------------------------------------------
# /api/browse
# ---------------------------------------------------------------------------

@browse_bp.route("/api/browse")
def api_browse():
    """浏览任意目录（chunk_dir 或 gam_dir）"""
    try:
        dir_path = request.args.get("path", "").strip()
        if not dir_path:
            return jsonify({"error": "Path is required"})

        dir_path = Path(dir_path)
        if not dir_path.exists():
            return jsonify({"error": "Directory not found"})

        def scan_directory(path: Path) -> dict:
            result = {"name": path.name, "is_dir": path.is_dir()}
            if path.is_dir():
                children = []
                for child in sorted(path.iterdir()):
                    if child.name.startswith("."):
                        continue
                    children.append(scan_directory(child))
                result["children"] = children
            return result

        tree = scan_directory(dir_path)

        def count_items(node):
            dirs = 1 if node["is_dir"] else 0
            files = 0 if node["is_dir"] else 1
            for child in node.get("children", []):
                d, f = count_items(child)
                dirs += d
                files += f
            return dirs, files

        total_dirs, total_files = count_items(tree)

        return jsonify({
            "tree": tree,
            "stats": {"dirs": total_dirs, "files": total_files},
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ---------------------------------------------------------------------------
# /api/browse_chunks
# ---------------------------------------------------------------------------

@browse_bp.route("/api/browse_chunks")
def api_browse_chunks():
    """浏览会话内所有 chunks 目录（chunks, chunks_1, chunks_2…）"""
    try:
        dir_path = request.args.get("path", "").strip()
        if not dir_path:
            return jsonify({"error": "Path is required"})

        base_path = Path(dir_path)
        if not base_path.exists():
            return jsonify({"error": "Directory not found"})
        if not base_path.is_dir():
            return jsonify({"error": "Not a directory"})

        def is_chunk_dir(name: str) -> bool:
            return name.startswith("chunks_") and name.split("chunks_", 1)[-1].isdigit()

        def scan_directory(path: Path) -> dict:
            result = {"name": path.name, "is_dir": path.is_dir()}
            if path.is_dir():
                children = []
                for child in sorted(path.iterdir()):
                    if child.name.startswith("."):
                        continue
                    children.append(scan_directory(child))
                result["children"] = children
            return result

        root = {"name": base_path.name, "is_dir": True, "children": []}

        for child in sorted(base_path.iterdir()):
            if not child.is_dir():
                continue
            if not is_chunk_dir(child.name):
                continue
            root["children"].append(scan_directory(child))

        def count_items(node):
            dirs = 1 if node["is_dir"] else 0
            files = 0 if node["is_dir"] else 1
            for child in node.get("children", []):
                d, f = count_items(child)
                dirs += d
                files += f
            return dirs, files

        total_dirs, total_files = count_items(root)

        return jsonify({
            "tree": root,
            "stats": {"dirs": total_dirs, "files": total_files},
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ---------------------------------------------------------------------------
# /api/file_content
# ---------------------------------------------------------------------------

@browse_bp.route("/api/file_content")
def api_file_content():
    """获取文件/目录内容（直接从文件系统读取）"""
    try:
        file_path = request.args.get("path", "").strip()
        if not file_path:
            return jsonify({"error": "Path is required"})

        file_path = Path(file_path)
        if not file_path.exists():
            return jsonify({"error": "Path not found"})

        if file_path.is_dir():
            children = []
            for child in sorted(file_path.iterdir()):
                if child.name.startswith("."):
                    continue
                children.append({
                    "name": child.name,
                    "is_dir": child.is_dir(),
                    "size": child.stat().st_size if child.is_file() else None,
                })

            return jsonify({
                "name": file_path.name,
                "is_dir": True,
                "children": children,
                "full_path": str(file_path),
            })
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                content = f"[Error reading file: {e}]"

            return jsonify({"name": file_path.name, "is_dir": False, "content": content})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


# ---------------------------------------------------------------------------
# /api/find_source
# ---------------------------------------------------------------------------

@browse_bp.route("/api/find_source")
def api_find_source():
    """根据文件名在指定目录中查找文件的完整路径"""
    try:
        gam_dir = request.args.get("gam_dir", "").strip()
        source_name = request.args.get("source_name", "").strip()

        if not gam_dir or not source_name:
            return jsonify({"found": False, "error": "gam_dir and source_name are required"})

        # If source_name is an absolute path, try it first
        if os.path.isabs(source_name):
            source_path = Path(source_name)
            if source_path.exists() and source_path.is_file():
                return jsonify({
                    "found": True,
                    "full_path": str(source_path),
                    "relative_path": source_name,
                })
            rel_candidate = Path(gam_dir) / source_name.lstrip("/")
            if rel_candidate.exists() and rel_candidate.is_file():
                return jsonify({
                    "found": True,
                    "full_path": str(rel_candidate),
                    "relative_path": str(Path(source_name).as_posix()),
                })
            source_name = os.path.basename(source_name)
        elif "/" in source_name or "\\" in source_name:
            rel_candidate = Path(gam_dir) / source_name.lstrip("/").lstrip("\\")
            if rel_candidate.exists() and rel_candidate.is_file():
                return jsonify({
                    "found": True,
                    "full_path": str(rel_candidate),
                    "relative_path": str(Path(source_name).as_posix()),
                })
            source_name = os.path.basename(source_name)

        gam_path = Path(gam_dir)
        if not gam_path.exists():
            return jsonify({"found": False, "error": "Directory not found"})

        # Search for the file recursively
        for root, dirs, files in os.walk(gam_path):
            for file in files:
                if file == source_name:
                    full_path = os.path.join(root, file)
                    return jsonify({
                        "found": True,
                        "full_path": full_path,
                        "relative_path": os.path.relpath(full_path, gam_dir),
                    })

        # If exact match not found, try partial match
        source_base = os.path.splitext(source_name)[0]
        for root, dirs, files in os.walk(gam_path):
            for file in files:
                file_base = os.path.splitext(file)[0]
                if file_base == source_base:
                    full_path = os.path.join(root, file)
                    return jsonify({
                        "found": True,
                        "full_path": full_path,
                        "relative_path": os.path.relpath(full_path, gam_dir),
                    })

        return jsonify({"found": False, "error": f'Source "{source_name}" not found in {gam_dir}'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"found": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/load
# ---------------------------------------------------------------------------

@browse_bp.route("/api/load")
def api_load():
    """加载现有 GAM"""
    try:
        gam_path = request.args.get("path", "").strip()
        if not gam_path:
            return jsonify({"success": False, "error": "Path is required"})

        gam_path = Path(gam_path)
        if not gam_path.exists():
            return jsonify({"success": False, "error": "Path does not exist"})

        workspace = LocalWorkspace(
            root_path=str(gam_path),
            name=gam_path.name,
            description=f"Loaded GAM: {gam_path.name}",
        )
        tree = GAMTree.from_disk(gam_path, workspace)
        gam_cache: Dict = current_app.config["GAM_CACHE"]
        gam_cache[str(gam_path)] = tree

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/tree
# ---------------------------------------------------------------------------

@browse_bp.route("/api/tree")
def api_tree():
    """获取目录树"""
    try:
        gam_path = request.args.get("path", "").strip()
        if not gam_path:
            return jsonify({"error": "Path is required"})

        gam_path = Path(gam_path)

        gam_cache: Dict = current_app.config["GAM_CACHE"]
        if str(gam_path) in gam_cache:
            tree = gam_cache[str(gam_path)]
        elif gam_path.exists():
            workspace = LocalWorkspace(
                root_path=str(gam_path),
                name=gam_path.name,
                description=f"GAM: {gam_path.name}",
            )
            tree = GAMTree.from_disk(gam_path, workspace)
            gam_cache[str(gam_path)] = tree
        else:
            return jsonify({"error": "GAM not found"})

        def node_to_dict(node):
            result = {"name": node.name, "is_dir": node.is_dir}
            if node.is_dir:
                result["children"] = [node_to_dict(c) for c in node.children]
            return result

        return jsonify({
            "tree": node_to_dict(tree.root),
            "stats": {
                "dirs": len(tree.root.get_all_dirs()),
                "files": len(tree.root.get_all_files()),
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------------------------------------------------------------------
# /api/content
# ---------------------------------------------------------------------------

@browse_bp.route("/api/content")
def api_content():
    """获取文件/目录内容"""
    try:
        gam_path = request.args.get("gam_path", "").strip()
        node_path = request.args.get("path", "/").strip()

        if not gam_path:
            return jsonify({"error": "GAM path is required"})

        gam_path = Path(gam_path)

        gam_cache: Dict = current_app.config["GAM_CACHE"]
        if str(gam_path) in gam_cache:
            tree = gam_cache[str(gam_path)]
        elif gam_path.exists():
            workspace = LocalWorkspace(
                root_path=str(gam_path),
                name=gam_path.name,
                description=f"GAM: {gam_path.name}",
            )
            tree = GAMTree.from_disk(gam_path, workspace)
            gam_cache[str(gam_path)] = tree
        else:
            return jsonify({"error": "GAM not found"})

        node = tree.get_node(node_path)
        if node is None:
            return jsonify({"error": "Node not found"})

        if node.is_dir:
            return jsonify({"name": node.name, "is_dir": True, "readme": node.summary or ""})
        else:
            return jsonify({"name": node.name, "is_dir": False, "content": node.content or ""})

    except Exception as e:
        return jsonify({"error": str(e)})
