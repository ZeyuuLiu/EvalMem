# -*- coding: utf-8 -*-
"""
Pipeline API routes – chunking & GAM building endpoints.
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Dict

from flask import Blueprint, current_app, jsonify, request

from ...core.tree import GAMTree
from ...workspaces.local_workspace import LocalWorkspace
from ..helpers import (
    get_timestamp_dir,
    next_chunks_dir,
    parse_bool_param,
    parse_int_param,
    read_uploaded_files,
    task_results,
)

pipeline_bp = Blueprint("pipeline", __name__)


# ---------------------------------------------------------------------------
# /api/pipeline  (synchronous, kept for backward compat)
# ---------------------------------------------------------------------------

@pipeline_bp.route("/api/pipeline", methods=["POST"])
def api_pipeline():
    """
    完整的处理流程（同步版本）：
    1. 读取上传的文件/文本
    2. 使用 TextGAMAgent 切分成 chunks 并生成 memory
    3. 组织成 GAM 结构
    """
    try:
        generator = current_app.config.get("GENERATOR")
        if not generator:
            return jsonify({"success": False, "error": "LLM generator not configured. Pipeline requires an LLM."})

        # 生成时间戳目录
        timestamp = get_timestamp_dir()
        base_dir = Path(current_app.config["OUTPUT_BASE"]) / timestamp
        chunk_dir = next_chunks_dir(base_dir)
        gam_dir = base_dir / "gam"

        chunk_dir.mkdir(parents=True, exist_ok=True)
        gam_dir.mkdir(parents=True, exist_ok=True)

        content = ""
        source_name = "text_input"
        split_within_file = parse_bool_param(request.form.get("split_within_file"), True)

        uploaded_files = request.files.getlist("files")
        if not uploaded_files and "file" in request.files:
            uploaded_files = [request.files["file"]]

        file_entries = read_uploaded_files(uploaded_files) if uploaded_files else []

        if file_entries:
            source_name = file_entries[0]["stem"] if len(file_entries) == 1 else "multi_files"
            if split_within_file:
                content = "\n\n".join(
                    [f"[File: {entry['filename']}]\n{entry['content']}" for entry in file_entries]
                )
        else:
            content = request.form.get("content", "").strip()

        if file_entries and not split_within_file:
            if not any(entry["content"].strip() for entry in file_entries):
                return jsonify({"success": False, "error": "No content provided"})
        elif not content:
            return jsonify({"success": False, "error": "No content provided"})

        # ===== 使用 TextGAMAgent 处理 =====
        from ...agents.text_gam_agent import TextGAMAgent

        window_size = parse_int_param(request.form.get("window_size"), 8000, 1000, 100000)
        overlap_size = parse_int_param(request.form.get("overlap_size"), 1000, 100, 20000)

        workspace = LocalWorkspace(
            root_path=str(gam_dir),
            name=source_name,
            description=f"GAM built from {source_name}",
        )
        tree = GAMTree.create_empty(gam_dir, name=source_name)

        agent = TextGAMAgent(
            generator=generator,
            tree=tree,
            workspace=workspace,
            use_chunking=split_within_file,
            window_size=window_size,
            overlap_size=overlap_size,
            auto_save=True,
            verbose=False,
            memory_workers=4,
        )

        temp_input_file = chunk_dir / f"{source_name}.txt"
        if file_entries and not split_within_file:
            content = "\n\n".join(
                [entry["content"].strip() for entry in file_entries if entry["content"].strip()]
            )

        with open(temp_input_file, "w", encoding="utf-8") as f:
            f.write(content)

        gam_result = agent.add(
            input_file=temp_input_file,
            context=f"Source: {source_name}",
            output_dir=chunk_dir,
        )

        tree = tree.reload(workspace)

        gam_actions = []
        if hasattr(gam_result, "memorized_chunks") and gam_result.memorized_chunks:
            gam_actions.append(f"Memorized {len(gam_result.memorized_chunks)} chunks")
        if hasattr(gam_result, "organization_plan") and gam_result.organization_plan:
            gam_actions.append(f"Created {len(gam_result.organization_plan.directories)} directories")
        if hasattr(gam_result, "created_files") and gam_result.created_files:
            gam_actions.append(f"Created {len(gam_result.created_files)} files")

        chunk_count = 0
        if hasattr(gam_result, "memorized_chunks"):
            chunk_count = len(gam_result.memorized_chunks) if gam_result.memorized_chunks else 0

        gam_cache: Dict = current_app.config["GAM_CACHE"]
        gam_cache[str(gam_dir)] = tree

        return jsonify({
            "success": True,
            "chunk_dir": str(chunk_dir),
            "gam_dir": str(gam_dir),
            "chunk_count": chunk_count,
            "gam_actions": gam_actions,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/pipeline_start  (asynchronous)
# ---------------------------------------------------------------------------

@pipeline_bp.route("/api/pipeline_start", methods=["POST"])
def api_pipeline_start():
    """异步启动 pipeline，立即返回 task_id 和输出目录路径。"""
    try:
        generator = current_app.config.get("GENERATOR")
        if not generator:
            return jsonify({"success": False, "error": "LLM generator not configured. Pipeline requires an LLM."})

        timestamp = get_timestamp_dir()
        base_dir = Path(current_app.config["OUTPUT_BASE"]) / timestamp
        chunk_dir = next_chunks_dir(base_dir)
        gam_dir = base_dir / "gam"

        chunk_dir.mkdir(parents=True, exist_ok=True)
        gam_dir.mkdir(parents=True, exist_ok=True)

        content = ""
        source_name = "text_input"
        split_within_file = parse_bool_param(request.form.get("split_within_file"), True)

        uploaded_files = request.files.getlist("files")
        if not uploaded_files and "file" in request.files:
            uploaded_files = [request.files["file"]]

        file_entries = read_uploaded_files(uploaded_files) if uploaded_files else []

        if file_entries:
            source_name = file_entries[0]["stem"] if len(file_entries) == 1 else "multi_files"
            if split_within_file:
                content = "\n\n".join(
                    [f"[File: {entry['filename']}]\n{entry['content']}" for entry in file_entries]
                )
        else:
            content = request.form.get("content", "").strip()

        window_size = parse_int_param(request.form.get("window_size"), 8000, 1000, 100000)
        overlap_size = parse_int_param(request.form.get("overlap_size"), 1000, 100, 20000)

        task_id = str(uuid.uuid4())

        task_results[task_id] = {
            "status": "running",
            "stage": "initializing",
            "chunk_dir": str(chunk_dir),
            "gam_dir": str(gam_dir),
            "chunk_count": 0,
            "gam_actions": [],
            "error": None,
            "message": "",
        }

        # Capture app-level references for the background thread
        gam_cache: Dict = current_app.config["GAM_CACHE"]

        def run_pipeline():
            try:
                from ...agents.text_gam_agent import TextGAMAgent

                task_results[task_id]["stage"] = "processing"
                task_results[task_id]["message"] = "Starting TextGAMAgent processing..."

                workspace = LocalWorkspace(
                    root_path=str(gam_dir),
                    name=source_name,
                    description=f"GAM built from {source_name}",
                )
                tree = GAMTree.create_empty(gam_dir, name=source_name)

                agent = TextGAMAgent(
                    generator=generator,
                    tree=tree,
                    workspace=workspace,
                    use_chunking=split_within_file,
                    window_size=window_size,
                    overlap_size=overlap_size,
                    auto_save=True,
                    verbose=False,
                    memory_workers=4,
                )

                task_results[task_id]["stage"] = "chunking"
                task_results[task_id]["message"] = "Chunking and generating memories..."

                def progress_callback(event_type, data):
                    if event_type == "memory_generated":
                        task_results[task_id]["stage"] = "chunking"
                        task_results[task_id]["message"] = f'Generated memory: {data.get("title", "")}'
                        task_results[task_id]["chunk_count"] += 1
                    elif event_type == "memorized_chunks_loaded":
                        task_results[task_id]["stage"] = "gam"
                        task_results[task_id]["message"] = "Building GAM structure..."
                    elif event_type == "stage_1_complete":
                        task_results[task_id]["message"] = f'Organized {data.get("directories", 0)} directories, generating READMEs...'
                    elif event_type == "stage_2_complete":
                        task_results[task_id]["message"] = f'Created {data.get("created_files", 0)} files, finalizing...'

                if file_entries and not split_within_file:
                    content_list = [
                        entry["content"].strip()
                        for entry in file_entries
                        if entry["content"].strip()
                    ]
                    if not content_list:
                        raise ValueError("No content provided")

                    gam_result = agent.add(
                        content=content_list,
                        context=f"Source: {source_name}",
                        output_dir=chunk_dir,
                        callback=progress_callback,
                    )
                else:
                    final_content = content
                    if not final_content:
                        raise ValueError("No content provided")

                    temp_input_file = chunk_dir / f"{source_name}.txt"
                    with open(temp_input_file, "w", encoding="utf-8") as f:
                        f.write(final_content)

                    gam_result = agent.add(
                        input_file=temp_input_file,
                        context=f"Source: {source_name}",
                        output_dir=chunk_dir,
                        callback=progress_callback,
                    )

                tree = tree.reload(workspace)

                gam_actions = []
                chunk_count = 0

                if hasattr(gam_result, "memorized_chunks") and gam_result.memorized_chunks:
                    chunk_count = len(gam_result.memorized_chunks)
                    gam_actions.append(f"Memorized {chunk_count} chunks")
                if hasattr(gam_result, "organization_plan") and gam_result.organization_plan:
                    gam_actions.append(f"Created {len(gam_result.organization_plan.directories)} directories")
                if hasattr(gam_result, "created_files") and gam_result.created_files:
                    gam_actions.append(f"Created {len(gam_result.created_files)} files")

                gam_cache[str(gam_dir)] = tree

                task_results[task_id]["status"] = "completed"
                task_results[task_id]["chunk_count"] = chunk_count
                task_results[task_id]["gam_actions"] = gam_actions
                task_results[task_id]["message"] = "Pipeline completed successfully"

            except Exception as e:
                import traceback
                traceback.print_exc()
                task_results[task_id]["status"] = "error"
                task_results[task_id]["error"] = str(e)

        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "task_id": task_id,
            "chunk_dir": str(chunk_dir),
            "chunk_base_dir": str(chunk_dir.parent),
            "gam_dir": str(gam_dir),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/pipeline_add  (incremental add, asynchronous)
# ---------------------------------------------------------------------------

@pipeline_bp.route("/api/pipeline_add", methods=["POST"])
def api_pipeline_add():
    """异步增量添加 pipeline。"""
    try:
        generator = current_app.config.get("GENERATOR")
        if not generator:
            return jsonify({"success": False, "error": "LLM generator not configured. Pipeline requires an LLM."})

        gam_dir_value = request.form.get("gam_dir", "").strip()
        if not gam_dir_value:
            return jsonify({"success": False, "error": "gam_dir is required for incremental add."})

        gam_dir = Path(gam_dir_value)
        if not gam_dir.exists():
            return jsonify({"success": False, "error": f"GAM directory not found: {gam_dir_value}"})

        base_dir = gam_dir.parent
        chunk_dir = next_chunks_dir(base_dir)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        content = ""
        source_name = "text_input"
        split_within_file = parse_bool_param(request.form.get("split_within_file"), True)

        uploaded_files = request.files.getlist("files")
        if not uploaded_files and "file" in request.files:
            uploaded_files = [request.files["file"]]

        file_entries = read_uploaded_files(uploaded_files) if uploaded_files else []

        if file_entries:
            source_name = file_entries[0]["stem"] if len(file_entries) == 1 else "multi_files"
            if split_within_file:
                content = "\n\n".join(
                    [f"[File: {entry['filename']}]\n{entry['content']}" for entry in file_entries]
                )
        else:
            content = request.form.get("content", "").strip()

        window_size = parse_int_param(request.form.get("window_size"), 8000, 1000, 100000)
        overlap_size = parse_int_param(request.form.get("overlap_size"), 1000, 100, 20000)

        task_id = str(uuid.uuid4())

        task_results[task_id] = {
            "status": "running",
            "stage": "initializing",
            "chunk_dir": str(chunk_dir),
            "gam_dir": str(gam_dir),
            "chunk_count": 0,
            "gam_actions": [],
            "error": None,
            "message": "",
        }

        gam_cache: Dict = current_app.config["GAM_CACHE"]

        def run_pipeline():
            try:
                from ...agents.text_gam_agent import TextGAMAgent

                task_results[task_id]["stage"] = "processing"
                task_results[task_id]["message"] = "Starting TextGAMAgent incremental add..."

                workspace = LocalWorkspace(
                    root_path=str(gam_dir),
                    name="incremental_add",
                    description=f"Incremental add from {source_name}",
                )

                tree = GAMTree.from_disk(gam_dir, workspace)

                agent = TextGAMAgent(
                    generator=generator,
                    tree=tree,
                    workspace=workspace,
                    use_chunking=split_within_file,
                    window_size=window_size,
                    overlap_size=overlap_size,
                    auto_save=True,
                    verbose=False,
                    memory_workers=4,
                )

                task_results[task_id]["stage"] = "chunking"
                task_results[task_id]["message"] = "Chunking and generating memories..."

                def progress_callback(event_type, data):
                    if event_type == "memory_generated":
                        task_results[task_id]["stage"] = "chunking"
                        task_results[task_id]["message"] = f'Generated memory: {data.get("title", "")}'
                        task_results[task_id]["chunk_count"] += 1
                    elif event_type == "decision_made":
                        task_results[task_id]["stage"] = "gam_add"
                        belongs = data.get("belongs_to_existing", True)
                        task_results[task_id]["message"] = (
                            "Adding to existing GAM structure..."
                            if belongs else "Expanding GAM with new topic..."
                        )

                if file_entries and not split_within_file:
                    content_list = [
                        entry["content"].strip()
                        for entry in file_entries
                        if entry["content"].strip()
                    ]
                    if not content_list:
                        raise ValueError("No content provided")

                    add_result = agent.add(
                        content=content_list,
                        context=f"Source: {source_name}",
                        callback=progress_callback,
                        output_dir=chunk_dir,
                    )
                else:
                    final_content = content
                    if not final_content:
                        raise ValueError("No content provided")

                    temp_input_file = chunk_dir / f"{source_name}.txt"
                    with open(temp_input_file, "w", encoding="utf-8") as f:
                        f.write(final_content)

                    add_result = agent.add(
                        input_file=temp_input_file,
                        context=f"Source: {source_name}",
                        callback=progress_callback,
                        output_dir=chunk_dir,
                    )

                tree = tree.reload(workspace)

                gam_actions = []
                chunk_count = 0

                if hasattr(add_result, "created_files"):
                    gam_actions.append(f"Created {len(add_result.created_files)} files")
                if hasattr(add_result, "new_directories"):
                    gam_actions.append(f"Created {len(add_result.new_directories)} directories")
                if hasattr(add_result, "affected_paths"):
                    gam_actions.append(f"Updated {len(add_result.affected_paths)} READMEs")

                if hasattr(add_result, "chunks_added"):
                    chunk_count = add_result.chunks_added
                elif hasattr(add_result, "created_files"):
                    chunk_count = len(add_result.created_files)

                gam_cache[str(gam_dir)] = tree

                task_results[task_id]["status"] = "completed"
                task_results[task_id]["chunk_count"] = chunk_count
                task_results[task_id]["gam_actions"] = gam_actions
                task_results[task_id]["message"] = "Incremental add completed successfully"

            except Exception as e:
                import traceback
                traceback.print_exc()
                task_results[task_id]["status"] = "error"
                task_results[task_id]["error"] = str(e)

        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "task_id": task_id,
            "chunk_dir": str(chunk_dir),
            "chunk_base_dir": str(chunk_dir.parent),
            "gam_dir": str(gam_dir),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/pipeline_status
# ---------------------------------------------------------------------------

@pipeline_bp.route("/api/pipeline_status")
def api_pipeline_status():
    """获取 pipeline 任务状态"""
    task_id = request.args.get("task_id", "").strip()

    if not task_id:
        return jsonify({"error": "task_id is required"})

    if task_id not in task_results:
        return jsonify({"error": "Task not found"})

    result = task_results[task_id]
    return jsonify(result)


# ---------------------------------------------------------------------------
# /api/process  (legacy)
# ---------------------------------------------------------------------------

@pipeline_bp.route("/api/process", methods=["POST"])
def api_process():
    """处理新内容，构建/更新 GAM（保留旧 API）"""
    try:
        gam_path = request.form.get("gam_path", "").strip()
        if not gam_path:
            return jsonify({"success": False, "error": "GAM path is required"})

        gam_path = Path(gam_path)
        content = ""

        if "file" in request.files:
            file = request.files["file"]
            if file.filename:
                import tempfile as _tempfile

                from werkzeug.utils import secure_filename as _secure

                filename = _secure(file.filename)
                with _tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                    file.save(tmp.name)
                    tmp_path = tmp.name

                try:
                    if filename.lower().endswith(".pdf"):
                        from ...readers import PdfReader as _PdfReader
                        reader = _PdfReader()
                        content = reader.read(tmp_path)
                    else:
                        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                finally:
                    os.unlink(tmp_path)
        else:
            content = request.form.get("content", "").strip()

        if not content:
            return jsonify({"success": False, "error": "No content provided"})

        workspace = LocalWorkspace(
            root_path=str(gam_path),
            name=gam_path.name,
            description=f"GAM for {gam_path.name}",
        )

        if gam_path.exists():
            tree = GAMTree.from_disk(gam_path, workspace)
        else:
            tree = GAMTree.create_empty(gam_path, name=gam_path.name)

        actions = []
        generator = current_app.config.get("GENERATOR")

        if generator:
            from ...agents.text_gam_agent import TextGAMAgent
            import time

            window_size = parse_int_param(request.form.get("window_size"), 8000, 1000, 100000)
            overlap_size = parse_int_param(request.form.get("overlap_size"), 1000, 100, 20000)

            agent = TextGAMAgent(
                generator=generator,
                tree=tree,
                workspace=workspace,
                use_chunking=True,
                window_size=window_size,
                overlap_size=overlap_size,
                auto_save=True,
                verbose=False,
                memory_workers=4,
            )

            temp_input_file = gam_path / "_temp_input.txt"
            with open(temp_input_file, "w", encoding="utf-8") as f:
                f.write(content)

            try:
                result = agent.add(input_file=temp_input_file, context="")
                tree = tree.reload(workspace)

                chunk_count = 0
                if hasattr(result, "memorized_chunks") and result.memorized_chunks:
                    chunk_count = len(result.memorized_chunks)

                actions.append(f"Batch processed {chunk_count} chunk(s)")

                if hasattr(result, "memorized_chunks") and result.memorized_chunks:
                    for mc in result.memorized_chunks[:5]:
                        actions.append(f"  Memorized: {mc.title}")
                    if len(result.memorized_chunks) > 5:
                        actions.append(f"  ... and {len(result.memorized_chunks) - 5} more")

                if hasattr(result, "organization_plan") and result.organization_plan:
                    for d in result.organization_plan.directories[:5]:
                        if d.chunk_indices:
                            actions.append(f"  Created: {d.path} (chunks: {d.chunk_indices})")
                        else:
                            actions.append(f"  Created: {d.path} (parent)")
                    if len(result.organization_plan.directories) > 5:
                        actions.append(f"  ... and {len(result.organization_plan.directories) - 5} more")

                if hasattr(result, "created_files") and result.created_files:
                    actions.append(f"  Total files created: {len(result.created_files)}")
            finally:
                if temp_input_file.exists():
                    os.unlink(temp_input_file)
        else:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = gam_path / f"content_{timestamp}.md"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            tree = tree.reload(workspace)
            actions.append(f"Created file: {file_path}")

        gam_cache: Dict = current_app.config["GAM_CACHE"]
        gam_cache[str(gam_path)] = tree

        return jsonify({"success": True, "actions": actions, "path": str(gam_path)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})
