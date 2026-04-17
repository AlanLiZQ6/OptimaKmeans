#!/usr/bin/env python3
"""Hardware probe helper for GPU K-means auto-tuning.

This script parses host and GPU hardware metadata and prints JSON.
"""

import argparse
import ctypes
import json
import os
import platform
import socket
import subprocess
import sys
from typing import Any, Dict, List, Tuple


def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"


def _safe_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except Exception:
        return default


def probe_gpu() -> Dict[str, Any]:
    """Probe GPU metadata from nvidia-smi.

    Returns a dict with conservative fields; missing values are empty strings.
    """
    query_fields = [
        "name",
        "uuid",
        "driver_version",
        "memory.total",
        "clocks.max.sm",
        "clocks.max.memory",
        "power.limit",
        "temperature.gpu",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]
    code, out, err = _run_cmd(cmd)

    result: Dict[str, Any] = {
        "tool": "nvidia-smi",
        "available": code == 0,
        "raw_error": err if code != 0 else "",
        "gpus": [],
    }

    if code != 0 or not out:
        return result

    lines = [line.strip() for line in out.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(query_fields):
            continue
        gpu = {
            "index": idx,
            "name": parts[0],
            "uuid": parts[1],
            "driver_version": parts[2],
            "memory_total_mib": _safe_int(parts[3]),
            "sm_clock_mhz": _safe_int(parts[4]),
            "mem_clock_mhz": _safe_int(parts[5]),
            "power_limit_w": _safe_int(parts[6]),
            "temperature_c": _safe_int(parts[7]),
        }
        result["gpus"].append(gpu)

    # Enrich with CUDA runtime limits when available.
    rt = probe_cuda_runtime_limits()
    if rt.get("available"):
        attrs = rt.get("gpus", [])
        for i, gpu in enumerate(result["gpus"]):
            if i < len(attrs):
                gpu["max_threads_per_block"] = attrs[i].get("max_threads_per_block", 0)
                gpu["warp_size"] = attrs[i].get("warp_size", 0)
    else:
        for gpu in result["gpus"]:
            gpu["max_threads_per_block"] = 0
            gpu["warp_size"] = 0

    return result


def _load_cudart():
    candidates = [
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def probe_cuda_runtime_limits() -> Dict[str, Any]:
    """Probe selected CUDA runtime attributes (including maxThreadsPerBlock)."""
    lib = _load_cudart()
    if lib is None:
        return {
            "tool": "cudart",
            "available": False,
            "raw_error": "libcudart not found",
            "gpus": [],
        }

    # CUDA device attribute enum values.
    CUDA_DEV_ATTR_MAX_THREADS_PER_BLOCK = 1
    CUDA_DEV_ATTR_WARP_SIZE = 10

    count = ctypes.c_int(0)
    rc = lib.cudaGetDeviceCount(ctypes.byref(count))
    if rc != 0:
        return {
            "tool": "cudart",
            "available": False,
            "raw_error": "cudaGetDeviceCount failed with rc={}".format(rc),
            "gpus": [],
        }

    gpus = []
    for idx in range(count.value):
        max_tpb = ctypes.c_int(0)
        warp_size = ctypes.c_int(0)
        rc_max = lib.cudaDeviceGetAttribute(
            ctypes.byref(max_tpb),
            ctypes.c_int(CUDA_DEV_ATTR_MAX_THREADS_PER_BLOCK),
            ctypes.c_int(idx),
        )
        rc_warp = lib.cudaDeviceGetAttribute(
            ctypes.byref(warp_size),
            ctypes.c_int(CUDA_DEV_ATTR_WARP_SIZE),
            ctypes.c_int(idx),
        )
        gpus.append(
            {
                "index": idx,
                "max_threads_per_block": int(max_tpb.value) if rc_max == 0 else 0,
                "warp_size": int(warp_size.value) if rc_warp == 0 else 0,
            }
        )

    return {
        "tool": "cudart",
        "available": True,
        "raw_error": "",
        "gpus": gpus,
    }


def probe_system() -> Dict[str, Any]:
    """Probe basic host metadata used in cache keying."""
    nvcc_code, nvcc_out, _ = _run_cmd(["nvcc", "--version"])
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cuda_compiler_available": nvcc_code == 0,
        "nvcc_version_text": nvcc_out.splitlines()[-1] if nvcc_out else "",
    }


def save_json(path: str, obj: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware probe helper for auto-tuning.")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "autotune_probe.json"),
        help="Output JSON file path (default: scripts/autotune_probe.json)",
    )
    args = parser.parse_args()

    hw = {
        "system": probe_system(),
        "gpu": probe_gpu(),
    }

    save_json(args.out, hw)
    print(json.dumps(hw, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
