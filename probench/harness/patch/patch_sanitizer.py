from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

class MatchMethod(str, Enum):
    """How the search text was located in the source file."""
    EXACT       = "exact"
    DEDENTED    = "dedented"
    NORMALIZED  = "normalized"
    WS_COLLAPSED = "ws_collapsed" 
    FUZZY       = "fuzzy"
    NOT_FOUND   = "not_found"

class ApplyStatus(str, Enum):
    FULLY_APPLIED     = "fully_applied"
    PARTIALLY_APPLIED = "partially_applied"
    FAILED            = "failed"

@dataclass
class RawBlock:
    """One parsed search/replace block (before validation)"""
    file:    str
    search:  str
    replace: str

@dataclass
class ValidBlock:
    """A RawBlock that passed all validation checks."""
    file:    str
    search:  str
    replace: str


@dataclass
class BlockResult:
    """Outcome of applying one ValidBlock."""
    file:         str
    match_method: MatchMethod
    ok:           bool
    error:        Optional[str] = None


@dataclass
class PatchResult:
    """
    Final result of the full apply_patch() pipeline.
    Call .to_dict() for a plain dict compatible with legacy schemas.
    """
    ok:            bool
    apply_status:  ApplyStatus
    error:         Optional[str]
    updated_files: List[str]
    applied_any:   bool
    block_results: List[BlockResult]
    match_methods: Dict[str, int]       

    # metadata
    kind:        str                   = "search_replace"
    dry_run:     bool                  = False
    stage:       str                   = "apply"

    # sanitizer passthrough
    sanitizer_reason:         Optional[str] = None
    sanitizer_dropped_blocks: List[Any]     = field(default_factory=list)

    # warnings
    warnings:       List[str] = field(default_factory=list)
    warnings_count: int       = 0

    # legacy compat
    applied_with: Optional[str] = None
    tried:        List[str]     = field(default_factory=list)
    stdout_tail:  str           = ""
    stderr_tail:  str           = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok":             self.ok,
            "kind":           self.kind,
            "stage":          self.stage,
            "dry_run":        self.dry_run,
            "apply_status":   self.apply_status.value,
            "error":          self.error,
            "updated_files":  self.updated_files,
            "applied_any":    self.applied_any,
            "warnings":       self.warnings,
            "warnings_count": self.warnings_count,
            "block_results": [
                {
                    "file":         r.file,
                    "match_method": r.match_method.value,
                    "ok":           r.ok,
                    "error":        r.error,
                }
                for r in self.block_results
            ],
            "match_methods":            self.match_methods,
            "sanitizer_reason":         self.sanitizer_reason,
            "sanitizer_dropped_blocks": len(self.sanitizer_dropped_blocks),
            "applied_with":             self.applied_with,
            "tried":                    self.tried,
            "stdout_tail":              self.stdout_tail,
            "stderr_tail":              self.stderr_tail,
        }


# FORMAT CONSTANTS
SEARCH_MARKER  = "[SEARCH]"
DIVIDER        = "[/SEARCH]"
REPLACE_OPEN   = "[REPLACE]"
REPLACE_MARKER = "[/REPLACE]"

_BLOCK_RE = re.compile(
    r"(?:^|\n)"
    r"[ \t]*"
    r"(?P<path>[^\n\[\]<>=`{}'\"]+?\.(?:py))"
    r"[ \t]*"
    r"(?:\n[ \t]*)?"
    r"\[SEARCH\][ \t]*\n"
    r"(?P<search>.*?)\n"
    r"[ \t]*\[/SEARCH\][ \t]*\n"
    r"[ \t]*\[REPLACE\][ \t]*\n"
    r"(?P<replace>.*?)\n"
    r"[ \t]*\[/REPLACE\]",
    re.MULTILINE | re.DOTALL,
)
# SANITIZER

_DANGEROUS_PATTERNS = [
    r"\bos\.system\s*\(",
    r"\bsubprocess\.",
    r"\bopen\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
    r"\bimportlib\.import_module\s*\(",
    r"\bshutil\.rmtree\s*\(",
    r"\brequests\.",
]

def _check_safety(text: str) -> Tuple[bool, Optional[str]]:
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, text):
            return False, f"Dangerous pattern detected: {pattern}"
    return True, None

def _normalize_markers(text: str) -> str:
    """
    Normalize LLM output where markers may appear mid-line.
    Splits on all markers so each ends up on its own line.
    """
    # First normalize the path line — put [SEARCH] on its own line
    # handles: "path/to/file.py [SEARCH] def foo..." 
    text = re.sub(
        r"([^\n\[\]]+?\.py)[ \t]*\[SEARCH\]",
        r"\1\n[SEARCH]",
        text
    )

    # Then normalize all markers — each on its own line
    markers = ["[/SEARCH]", "[/REPLACE]", "[SEARCH]", "[REPLACE]"]
    for marker in markers:
        parts = text.split(marker)
        rejoined = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                rejoined.append(part.rstrip())
                rejoined.append(f"\n{marker}\n")
            else:
                rejoined.append(part.lstrip("\n"))
        text = "".join(rejoined)

    # clean up multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\n", "", (text or "").strip())
    text = re.sub(r"\n```$", "", text)
    return text

class SearchReplaceSanitizer:
    """
    Parse and validate LLM output into ValidBlocks.

    Pipeline:
      1. Strip markdown fences
      2. Extract [SEARCH]/[REPLACE] blocks via regex
      3. Per-block: validate path, search anchor, Python syntax, safety
      4. Return valid blocks + dropped blocks with reasons

    Returns a plain dict:
      {
        "is_valid":       bool,
        "blocks":         List[ValidBlock],
        "error":          str | None,
        "reason":         str,
        "dropped_blocks": [{"raw": RawBlock, "reason": str}],
      }
    """

    @classmethod
    def _parse_blocks(cls, text: str) -> List[RawBlock]:

        text = _strip_fences(text)
        normalized_format = _normalize_markers(text)   

        return [
            RawBlock(
                file    = m.group("path").strip(),
                search  = m.group("search"),
                replace = m.group("replace"),
            )
            for m in _BLOCK_RE.finditer(normalized_format)
        ]

    @classmethod
    def _validate_block(cls, block: RawBlock, idx: int) -> Tuple[bool, str]:
        if not block.file.strip():
            return False, f"Block[{idx}]: empty file path"

        if not block.search.strip():
            return False, (
                f"Block[{idx}] ({block.file}): empty search"
            )

        if block.replace.strip() and block.file.endswith((".py")):
            ok, safety_err = _check_safety(block.replace)
            if not ok:
                return False, f"Block[{idx}] ({block.file}): unsafe replace: {safety_err}"

        return True, ""

    @classmethod
    def sanitize(cls, llm_output: str) -> Dict[str, Any]:
        if not llm_output or not llm_output.strip():
            return {
                "is_valid": False, 
                "blocks": [], 
                "dropped_blocks": [],
                "error": "empty_output", 
                "reason": "LLM output is empty",
            }

        raw_blocks = cls._parse_blocks(llm_output)

        if not raw_blocks:
            return {
                "is_valid": False, 
                "blocks": [], 
                "dropped_blocks": [],
                "error": "no_blocks_found",
                "reason": (
                    f"No blocks found. Expected:\n"
                    f"  path/to/file.py\n"
                    f"  {SEARCH_MARKER}\n  <code>\n  {DIVIDER}\n"
                    f"  {REPLACE_OPEN}\n  <replacement>\n  {REPLACE_MARKER}"
                ),
            }

        valid_blocks:   List[ValidBlock] = []
        dropped_blocks: List[Dict]       = []

        for idx, raw in enumerate(raw_blocks):
            ok, reason = cls._validate_block(raw, idx)
            if ok:
                valid_blocks.append(ValidBlock(
                    file=raw.file, 
                    search=raw.search,
                    replace=raw.replace
                ))
            else:
                dropped_blocks.append({"raw": raw, "reason": reason})

        if not valid_blocks:
            return {
                "is_valid": False, 
                "blocks": [], 
                "dropped_blocks": dropped_blocks,
                "error": "all_blocks_invalid",
                "reason": f"All {len(raw_blocks)} block(s) failed validation",
            }

        reason = f"{len(valid_blocks)} valid block(s)"
        if dropped_blocks:
            reason += f", {len(dropped_blocks)} dropped"

        return {
            "is_valid": True, 
            "blocks": valid_blocks,
            "dropped_blocks": dropped_blocks, 
            "error": None, 
            "reason": reason
        }








