# -*- coding: utf-8 -*-
"""
GAM Workflows

High-level entry point: ``Workflow("text", ...)`` or ``Workflow("video", ...)``.
"""

from .base import BaseWorkflow, WorkflowType
from .text_workflow import TextWorkflow
from .video_workflow import VideoWorkflow


def Workflow(workflow_type: str = "text", **kwargs) -> BaseWorkflow:
    """
    Create a GAM workflow — the simplest way to use GAM.

    This factory function returns either a :class:`TextWorkflow` or a
    :class:`VideoWorkflow` based on *workflow_type*.  Users do **not**
    need to import any other classes.

    Args:
        workflow_type: ``"text"`` or ``"video"``.
        **kwargs: Forwarded to the corresponding workflow constructor.
                  Common options: ``gam_dir``, ``model``, ``api_key``,
                  ``api_base``, ``verbose``, etc.

    Returns:
        A :class:`TextWorkflow` or :class:`VideoWorkflow` instance.

    Examples::

        from gam import Workflow

        # ---- Text workflow ----
        wf = Workflow(
            "text",
            gam_dir="./my_gam",
            model="gpt-4o-mini",
            api_key="sk-xxx",
        )
        wf.add("paper.pdf")
        wf.add(content="Extra notes about the paper")
        result = wf.request("What is the main conclusion?")
        print(result.answer)

        # ---- Video workflow ----
        wf = Workflow(
            "video",
            gam_dir="./my_video_gam",
            model="gpt-4o",
            api_key="sk-xxx",
        )
        wf.add("./video_dir")
        result = wf.request("What happens in the video?")
        print(result.answer)
    """
    wtype = workflow_type.lower() if isinstance(workflow_type, str) else str(workflow_type)

    if wtype in ("text", WorkflowType.TEXT):
        return TextWorkflow(**kwargs)
    elif wtype in ("video", WorkflowType.VIDEO):
        return VideoWorkflow(**kwargs)
    else:
        supported = ", ".join(f"'{t.value}'" for t in WorkflowType if t.value in ("text", "video"))
        raise ValueError(
            f"Unsupported workflow_type={workflow_type!r}. "
            f"Supported types: {supported}"
        )


__all__ = [
    "Workflow",
    "BaseWorkflow",
    "WorkflowType",
    "TextWorkflow",
    "VideoWorkflow",
]
