"""Minimal audit logger used by every phase of the pipeline.

Writes timestamped lines to ``audit.log`` at the project root. Format matches
the historical log so prior entries remain readable.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any


_LOG_PATH = "audit.log"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write(line: str) -> None:
    with open(_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def setup_audit(project_root: str = ".") -> None:
    """Initialize the audit log in project_root. Appends a session header."""
    global _LOG_PATH
    _LOG_PATH = os.path.join(project_root, "audit.log")
    bar = "=" * 60
    _write(f"{_now()} | INFO | {bar}")
    _write(f"{_now()} | INFO | AUDIT LOG INITIALIZED")
    _write(f"{_now()} | INFO | Python version: {sys.version}")
    _write(f"{_now()} | INFO | Working directory: {os.path.abspath(project_root)}")
    _write(f"{_now()} | INFO | {bar}")


def log_phase_start(phase: str) -> None:
    _write(f"{_now()} | INFO | PHASE START: {phase}")


def log_phase_end(phase: str, status: str = "PASS", details: dict | None = None) -> None:
    tail = ""
    if details:
        try:
            tail = f" | DETAILS: {json.dumps(details, default=str)}"
        except Exception:
            tail = f" | DETAILS: {details}"
    _write(f"{_now()} | INFO | PHASE END: {phase} | STATUS: {status}{tail}")


def log_gate_check(name: str, passed: Any, expected: Any, actual: Any) -> None:
    ok = bool(passed)
    status = "PASS" if ok else "FAIL"
    level = "INFO" if ok else "WARN"
    _write(f"{_now()} | {level} | GATE CHECK: {name} | {status} | "
           f"expected={expected} | actual={actual}")


def log_metric(name: str, value: Any) -> None:
    _write(f"{_now()} | INFO | METRIC: {name} = {value}")


def log_warning(msg: str) -> None:
    _write(f"{_now()} | WARN | {msg}")


def log_file_created(path: str) -> None:
    if os.path.exists(path):
        size = os.path.getsize(path)
        _write(f"{_now()} | INFO | FILE CREATED: {path} | size={size} bytes")
    else:
        _write(f"{_now()} | WARN | FILE MISSING (expected created): {path}")
