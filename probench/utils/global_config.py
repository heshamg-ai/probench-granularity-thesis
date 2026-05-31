# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from pathlib import Path
from typing import Dict, Any
from probench.utils.io_utils import get_project_root

# This module holds config values that can be set once and used globally
seed: int = 123
measure: Dict[str, Any] = {}
analysis: Dict[str, Any] = {}
docker: Dict[str, Any] = {}
libraries: Dict[str, Any] = {}
root: Path = Path.cwd()

def initialize(config: Dict[str, Any]) -> None:
    global seed, measure, analysis, docker, libraries, root
    seed = int(config.get("global", {}).get("seed", 42))
    measure = config.get("measure", {}) or {}
    analysis = config.get("analysis", {}) or {}
    docker = config.get("docker", {}) or {}
    libraries = config.get("libraries", {}) or {}
    root = get_project_root() 
