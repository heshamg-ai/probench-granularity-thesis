# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from __future__ import annotations
from typing import Dict, Any, Iterator, Tuple, Union
from pathlib import Path
import os
import json

from datasets import load_dataset
from probench.prep.param_schema import ParamDef, ParamGrid

class DatasetLoader:
    """
    Lightweight wrapper around local/remote datasets used by ProBench.

    Supports two iteration modes:

    - ``raw``: yields a single row dictionary per sample.
    - ``parametrized``: yields ``(row, ParamGrid)`` where ``ParamGrid`` is
      constructed from the row's ``params`` field.

    The loader transparently handles:
    - Local JSON/JSONL files (list-of-dicts)
    - HuggingFace datasets (via ``datasets.load_dataset``)
    - Normalization of ``oracle_context`` when stored as a JSON string
    """

    def __init__(
        self,
        path_or_name: str,
        mode: str = "raw",
        split: str = "test"
    ):
        self.meta = {}
        self.mode = mode
        self.name = self._extract_name(path_or_name)
        self.data = self._load_dataset(path_or_name, split)

    # Data loading
    def _extract_name(self, path_or_name: str) -> str:
        path = Path(path_or_name)

        # Case 1: Local file exists → use filename without extension
        if path.exists():
            return path.stem  # "/x/y/EffiBench.json" → "EffiBench"

        # Case 2: HF dataset name → take last part after slash if present
        if "/" in path_or_name:
            return path_or_name.split("/")[-1]

        # Case 3: Already a clean name
        return path_or_name

    def _normalize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a single dataset row in-place (and return it).

        - If `oracle_context` is a JSON string, decode it to a dict.
        - If it's already a dict, leave as is.
        - If decoding fails, fallback to {}.
        """
        raw_ctx = row.get("oracle_context")

        if isinstance(raw_ctx, str):
            try:
                parsed = json.loads(raw_ctx)
                if isinstance(parsed, dict):
                    row["oracle_context"] = parsed
                else:
                    row["oracle_context"] = {}
            except Exception:
                row["oracle_context"] = {}

        return row

    def _load_dataset(self, path_or_name: str, split: str):
        path = Path(path_or_name)

        try:
            # Local JSON / JSONL
            if path.exists():
                if path.suffix not in {".json", ".jsonl"}:
                    raise ValueError("Only .json or .jsonl supported")

                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict) and isinstance(data.get("samples"), list):
                    self.meta = data.get("meta") or {}
                    data = data["samples"]
                else:
                    self.meta = {}

                if not isinstance(data, list):
                    raise ValueError("JSON dataset must be a list or {'meta':..., 'samples':[...]}")

                data = [self._normalize_row(row) for row in data]
                return data

            # HuggingFace dataset
            token = (
                os.getenv("HUGGINGFACE_TOKEN")
                or os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACE_HUB_TOKEN")
            )
            return load_dataset(path_or_name, split=split, token=token)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset from '{path_or_name}': {e}"
            ) from e

    def _to_paramgrid(self, row: Dict[str, Any]) -> ParamGrid:
        raw_params = row.get("params", [])
        paramdefs = []

        for p in raw_params:
            paramdefs.append(
                ParamDef(
                    name=p.get("name"),
                    key=p.get("short_indicator"),
                    description=p.get("explanation"),
                    values=p.get("values"),
                )
            )

        return ParamGrid(params=paramdefs)

    def __iter__(self) -> Iterator[Union[Dict[str, Any], Tuple[Dict[str, Any], ParamGrid]]]:
        for row in self.data:
            if self.mode == "parametrized":
                yield row, self._to_paramgrid(row)
            else:
                yield row

    def __getitem__(self, idx: int):
        row = self.data[int(idx)]
        if self.mode == "parametrized":
            return row, self._to_paramgrid(row)
        return row

    def __len__(self) -> int:
        return len(self.data)
