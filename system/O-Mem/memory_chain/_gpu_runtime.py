from __future__ import annotations

import ctypes
import os
import site
import sys
from pathlib import Path


_BOOTSTRAPPED = False


def _candidate_nvidia_roots() -> list[Path]:
    roots: list[Path] = []
    for sp in site.getsitepackages():
        p = Path(sp) / "nvidia"
        if p.exists():
            roots.append(p)
    env_root = Path(sys.executable).resolve().parents[1]
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    fallback = env_root / "lib" / pyver / "site-packages" / "nvidia"
    if fallback.exists() and fallback not in roots:
        roots.append(fallback)
    return roots


def bootstrap_cuda_wheel_runtime() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED or sys.platform != "linux":
        return

    libdirs: list[str] = []
    preload_paths: list[str] = []
    preferred = [
        ("nvjitlink", "libnvJitLink.so.12"),
        ("cublas", "libcublas.so.12"),
        ("cudnn", "libcudnn.so.9"),
        ("cusparse", "libcusparse.so.12"),
        ("cusolver", "libcusolver.so.11"),
        ("cufft", "libcufft.so.11"),
        ("curand", "libcurand.so.10"),
        ("nccl", "libnccl.so.2"),
    ]

    for root in _candidate_nvidia_roots():
        for child in root.iterdir():
            libdir = child / "lib"
            if libdir.is_dir():
                libdirs.append(str(libdir))
        for pkg, libname in preferred:
            candidate = root / pkg / "lib" / libname
            if candidate.exists():
                preload_paths.append(str(candidate))

    if libdirs:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        merged = ":".join(libdirs + ([current] if current else []))
        os.environ["LD_LIBRARY_PATH"] = merged

    for lib in preload_paths:
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue

    _BOOTSTRAPPED = True
