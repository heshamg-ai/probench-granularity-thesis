# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

import yaml
import os
import json
import numpy as np
from pathlib import Path

def save_yaml(data, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def load_yaml(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[warn] File not found: {filepath}")
        return {}

def load_json(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {filepath}: {e}")
        return {}

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]  # JSON has no tuple, use list
    elif isinstance(obj, set):
        return [make_json_safe(v) for v in obj]  # convert sets to lists
    elif isinstance(obj, (np.generic,)):  # np.int64, np.float64, etc.
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # fallback: try str() to be safe for unknown classes
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def save_json(filepath: str, data) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=2, ensure_ascii=False)

def get_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Project root not found")