from __future__ import annotations

"""
Audit script for O-Mem adapter compliance and runtime feasibility.
O-Mem 适配器合规性与可运行性审计脚本。
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
O_MEM_ROOT = PROJECT_ROOT / "system" / "O-Mem"
ADAPTER_ROOT = PROJECT_ROOT / "src" / "memory_eval" / "adapters"

REQUIRED_METHODS = {
    "EncodingAdapterProtocol": ["export_full_memory", "find_memory_records"],
    "RetrievalAdapterProtocol": ["retrieve_original"],
    "GenerationAdapterProtocol": ["generate_oracle_answer"],
}


def _scan_classes(py_files: List[Path]) -> Dict[str, List[str]]:
    cls_methods: Dict[str, List[str]] = {}
    for f in py_files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                cls_methods[f"{f.name}:{node.name}"] = methods
    return cls_methods


def _runtime_import_check() -> Dict[str, str]:
    cmd = [sys.executable, "-c", "import sys; sys.path.insert(0, 'system/O-Mem'); import memory_chain; print('ok')"]
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    return {
        "return_code": str(p.returncode),
        "stdout": p.stdout.strip(),
        "stderr": p.stderr.strip(),
    }


def main() -> int:
    if not O_MEM_ROOT.exists():
        print(json.dumps({"status": "error", "reason": "system/O-Mem not found"}, ensure_ascii=False, indent=2))
        return 1

    omem_py_files = list(O_MEM_ROOT.rglob("*.py"))
    adapter_py_files = list(ADAPTER_ROOT.rglob("*.py")) if ADAPTER_ROOT.exists() else []
    classes = _scan_classes(omem_py_files + adapter_py_files)

    # static compliance check
    protocol_hits = {k: [] for k in REQUIRED_METHODS}
    for cls, methods in classes.items():
        for proto, required in REQUIRED_METHODS.items():
            if all(m in methods for m in required):
                protocol_hits[proto].append(cls)

    runtime = _runtime_import_check()
    report = {
        "o_mem_root": str(O_MEM_ROOT),
        "adapter_root": str(ADAPTER_ROOT),
        "python_files": {
            "system_o_mem": len(omem_py_files),
            "adapters": len(adapter_py_files),
        },
        "protocol_compliance_hits": protocol_hits,
        "runtime_import_check": runtime,
        "conclusion": {
            "has_full_eval_adapter": all(len(v) > 0 for v in protocol_hits.values()),
            "importable_in_current_env": runtime["return_code"] == "0",
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
