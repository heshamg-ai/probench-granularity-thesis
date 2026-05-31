"""
patch_applier.py
----------------
Apply validated search/replace blocks to files on disk.

on_block_fail strategy
----------------------
  abort     — stop at first failure, write nothing for that file
  skip      — skip failed block, apply the rest, write what succeeded
  skip_file — if any block in a file fails, skip the entire file

ApplyStatus
-----------
  all blocks ok   → fully_applied    (ok=True)
  some blocks ok  → partially_applied (ok=False)
  nothing applied → failed           (ok=False)

Match strategy order (strictest → loosest)
------------------------------------------
  EXACT        — literal string match, unique
  DEDENTED     — textwrap.dedent applied, unique
  NORMALIZED   — strip trailing whitespace + normalize line endings, unique
  WS_COLLAPSED — collapse all whitespace runs to one space, line-based fuzzy,
                 ambiguity checked via collapsed string find()
  FUZZY        — sliding line window with variable size, margin-guarded
"""

from __future__ import annotations

import difflib
import re
import textwrap
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from probench.harness.patch.patch_sanitizer import (
    ApplyStatus,
    BlockResult,
    MatchMethod,
    PatchResult,
    SearchReplaceSanitizer,
    ValidBlock,
)


class OnBlockFail(str, Enum):
    ABORT     = "abort"
    SKIP      = "skip"
    SKIP_FILE = "skip_file"


_DEFAULT_FUZZY_THRESHOLD = 0.80
_WS_COLLAPSED_THRESHOLD  = 0.92
_FUZZY_MARGIN            = 0.05
_FUZZY_LINE_SLACK        = 6

Span = Tuple[int, int]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    return "\n".join(
        line.rstrip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    )

def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _span_for_lines(src_lines_ke: List[str], start: int, end: int) -> Span:
    s = sum(len(l) for l in src_lines_ke[:start])
    e = sum(len(l) for l in src_lines_ke[:end])
    return s, e

# ── Match strategies ──────────────────────────────────────────────────────────

def _try_exact(src: str, search: str) -> Optional[Span]:
    idx = src.find(search)
    if idx == -1:
        return None
    if src.find(search, idx + 1) != -1:
        return None  # ambiguous
    return idx, idx + len(search)

def _try_dedented(src: str, search: str) -> Optional[Span]:
    dedented = textwrap.dedent(search)
    if dedented == search:
        return None
    idx = src.find(dedented)
    if idx == -1:
        return None
    if src.find(dedented, idx + 1) != -1:
        return None  # ambiguous
    return idx, idx + len(dedented)

def _try_normalized(src: str, search: str) -> Optional[Span]:
    norm_src    = _normalize(src)
    norm_search = _normalize(search)
    idx = norm_src.find(norm_search)
    if idx == -1:
        return None
    if norm_src.find(norm_search, idx + 1) != -1:
        return None  # ambiguous
    start_line    = norm_src[:idx].count("\n")
    matched_lines = norm_search.count("\n") + 1
    return _span_for_lines(src.splitlines(keepends=True), start_line, start_line + matched_lines)

def _try_ws_collapsed(src: str, search: str) -> Optional[Span]:
    """
    Collapse all whitespace runs to a single space, then do a
    line-based sliding window match.

    Handles:
      - multiline <-> single-line drift (Black wrapping)
      - spacing differences around = and ()
      - quote style differences (ratio-based, not exact)

    Ambiguity is checked once on fully-collapsed strings.
    No margin check — the find() ambiguity guard is sufficient.
    """
    collapsed_src    = _collapse(src)
    collapsed_search = _collapse(search)

    if not collapsed_search:
        return None

    # Ambiguity check on fully collapsed strings
    first = collapsed_src.find(collapsed_search)
    if first != -1 and collapsed_src.find(collapsed_search, first + 1) != -1:
        return None

    src_lines_ke = src.splitlines(keepends=True)
    src_lines    = src.splitlines()
    search_cl    = [_collapse(l) for l in search.splitlines() if l.strip()]

    if not search_cl:
        return None

    n          = len(search_cl)
    best_ratio = 0.0
    best_start = None
    best_end   = None

    for i in range(len(src_lines)):
        candidate = []
        j = i
        while j < len(src_lines) and len(candidate) < n:
            c = _collapse(src_lines[j])
            if c:
                candidate.append(c)
            j += 1

        if len(candidate) < n:
            break

        ratio = difflib.SequenceMatcher(None, search_cl, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            best_end   = j

    if best_ratio < _WS_COLLAPSED_THRESHOLD or best_start is None:
        return None

    return _span_for_lines(src_lines_ke, best_start, best_end)


def _try_fuzzy(src: str, search: str, threshold: float) -> Optional[Span]:
    """
    Sliding line-window fuzzy match.
    Tries window sizes n +/- _FUZZY_LINE_SLACK to handle line-count drift.
    Requires best-vs-second margin to avoid wrong-location matches.
    """
    search_lines = search.splitlines()
    src_lines    = src.splitlines()
    src_lines_ke = src.splitlines(keepends=True)
    n            = len(search_lines)

    if n == 0 or n > len(src_lines):
        return None

    best_ratio  = 0.0
    best_start  = None
    best_n      = n
    second_best = 0.0

    for window in range(
        max(1, n - _FUZZY_LINE_SLACK),
        min(len(src_lines), n + _FUZZY_LINE_SLACK) + 1,
    ):
        for i in range(len(src_lines) - window + 1):
            ratio = difflib.SequenceMatcher(
                None, search_lines, src_lines[i : i + window]
            ).ratio()
            if ratio > best_ratio:
                second_best = best_ratio
                best_ratio  = ratio
                best_start  = i
                best_n      = window
            elif ratio > second_best:
                second_best = ratio

    if best_ratio < threshold or best_start is None:
        return None
    if best_ratio - second_best < _FUZZY_MARGIN:
        return None

    return _span_for_lines(src_lines_ke, best_start, best_start + best_n)


def _find_match(
    src: str,
    search: str,
    *,
    fuzzy: bool = True,
    fuzzy_threshold: float = _DEFAULT_FUZZY_THRESHOLD,
) -> Tuple[Optional[Span], MatchMethod]:

    span = _try_exact(src, search)
    if span:
        return span, MatchMethod.EXACT

    span = _try_dedented(src, search)
    if span:
        return span, MatchMethod.DEDENTED

    span = _try_normalized(src, search)
    if span:
        return span, MatchMethod.NORMALIZED

    span = _try_ws_collapsed(src, search)
    if span:
        return span, MatchMethod.WS_COLLAPSED

    if fuzzy:
        span = _try_fuzzy(src, search, fuzzy_threshold)
        if span:
            return span, MatchMethod.FUZZY

    return None, MatchMethod.NOT_FOUND


# ── Validation + formatting ───────────────────────────────────────────────────

def _validate_syntax(code: str) -> Optional[str]:
    try:
        import ast
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"

def _format_with_black(code: str) -> Tuple[str, Optional[str]]:
    try:
        import black
        return black.format_str(code, mode=black.Mode()), None
    except ImportError:
        return code, "black not installed — skipping formatting"
    except Exception as e:
        return code, f"black: {e}"


# ── Indentation ───────────────────────────────────────────────────────────────

def _reindent_replace(replace: str, base_indent_str: str) -> str:
    """
    Shift all lines in `replace` so that its base indentation aligns with
    `base_indent_str`. Internal relative indentation is preserved.
 
    Example:
        replace base = 4 spaces, base_indent_str = 8 spaces → delta = +4
        a line at 4  → 8
        a line at 8  → 12
        a line at 12 → 16
    """
    lines = replace.splitlines(True)
 
    # Find the base indent of the replace block itself
    replace_base_len = 0
    for line in lines:
        if line.strip():
            replace_base_len = len(line) - len(line.lstrip())
            break
 
    base_len = len(base_indent_str)
    delta    = base_len - replace_base_len  # can be negative, zero, or positive
 
    # Nothing to do
    if delta == 0:
        return replace
 
    out = []
    for line in lines:
        if not line.strip():
            out.append(line)  # preserve blank lines as-is
            continue
        current_indent = len(line) - len(line.lstrip())
        new_indent     = max(0, current_indent + delta)
        out.append(" " * new_indent + line.lstrip())
 
    return "".join(out)

# ── Single block ──────────────────────────────────────────────────────────────
def _apply_one_block(
    src: str,
    block: ValidBlock,
    *,
    fuzzy: bool,
    fuzzy_threshold: float,
) -> Tuple[Optional[str], MatchMethod, Optional[str]]:

    span, method = _find_match(
        src, block.search, fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold
    )

    if span is None:
        return None, method, f"search block have an error in {block.file}"

    start, end = span
    matched_text = src[start:end]

    base_indent_str = ""
    for line in matched_text.splitlines():
        if line.strip():
            base_indent_str = line[: len(line) - len(line.lstrip())]
            break

    replace_base_len = 0
    for line in block.replace.splitlines():
        if line.strip():
            replace_base_len = len(line) - len(line.lstrip())
            break
    
    replace = block.replace
    if replace_base_len != len(base_indent_str):
        replace = _reindent_replace(replace, base_indent_str)
            
    if matched_text.endswith("\n") and not replace.endswith("\n"):
        replace += "\n"
    elif not matched_text.endswith("\n") and replace.endswith("\n"):
        replace = replace.rstrip("\n")

    new_src = src[:start] + replace + src[end:]

    err = _validate_syntax(new_src)
    if err:
        return None, method, f"syntax error after apply in {block.file}: {err}"

    return new_src, method, None

# ── File-level apply ──────────────────────────────────────────────────────────

def _apply_file_blocks(
    target: Path,
    blocks: List[ValidBlock],
    *,
    fuzzy: bool,
    fuzzy_threshold: float,
    dry_run: bool,
    run_black: bool,
    on_block_fail: OnBlockFail,
) -> Tuple[List[BlockResult], List[str], bool]:

    block_results: List[BlockResult] = []
    warnings:      List[str]         = []
    original = target.read_text(encoding="utf-8")
    current  = original

    for i, block in enumerate(blocks):
        new_src, method, error = _apply_one_block(
            current, block, fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold
        )
        ok = new_src is not None
        block_results.append(BlockResult(file=block.file, match_method=method, ok=ok, error=error))

        if not ok:
            warnings.append(error or f"block failed in {block.file}")

            if on_block_fail in (OnBlockFail.ABORT, OnBlockFail.SKIP_FILE):
                mode = "abort" if on_block_fail == OnBlockFail.ABORT else "skip_file"
                for remaining in blocks[i + 1:]:
                    block_results.append(BlockResult(
                        file=remaining.file,
                        match_method=MatchMethod.NOT_FOUND,
                        ok=False,
                        error=f"skipped — earlier block failed ({mode} mode)",
                    ))
                return block_results, warnings, False

        else:
            current = new_src

    if current == original:
        return block_results, warnings, False

    if run_black and target.suffix == ".py":
        formatted, fmt_err = _format_with_black(current)
        if fmt_err:
            warnings.append(fmt_err)
        else:
            current = formatted

    if not dry_run:
        target.with_suffix(target.suffix + ".bak").write_text(original, encoding="utf-8")
        target.write_text(current, encoding="utf-8")

    return block_results, warnings, True


# ── Applier ───────────────────────────────────────────────────────────────────

class Applier:
    def __init__(
        self,
        *,
        repo_dir:        Path,
        dry_run:         bool,
        fuzzy:           bool        = True,
        fuzzy_threshold: float       = _DEFAULT_FUZZY_THRESHOLD,
        run_black:       bool        = True,
        on_block_fail:   OnBlockFail = OnBlockFail.ABORT,
    ):
        self.repo_dir        = repo_dir.resolve()
        self.dry_run         = dry_run
        self.fuzzy           = fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.run_black       = run_black
        self.on_block_fail   = on_block_fail

    def _resolve(self, file_rel: str) -> Tuple[Optional[Path], Optional[str]]:
        target = (self.repo_dir / file_rel).resolve()
        try:
            target.relative_to(self.repo_dir)
        except ValueError:
            return None, f"path traversal detected: {file_rel}"
        if not target.exists():
            return None, f"file not found: {file_rel}"
        return target, None

    def apply(self, blocks: List[ValidBlock], dropped_count: int = 0) -> PatchResult:
        if not blocks:
            status = ApplyStatus.FAILED if dropped_count == 0 else ApplyStatus.PARTIALLY_APPLIED
            return PatchResult(
                ok=False, apply_status=status, error="no blocks to apply",
                updated_files=[], applied_any=False, block_results=[],
                match_methods={}, dry_run=self.dry_run,
            )

        by_file: Dict[str, List[ValidBlock]] = defaultdict(list)
        for block in blocks:
            by_file[block.file].append(block)

        all_results:   List[BlockResult] = []
        all_warnings:  List[str]         = []
        updated_files: List[str]         = []

        for file_rel, file_blocks in by_file.items():
            target, err = self._resolve(file_rel)
            if target is None:
                for block in file_blocks:
                    all_results.append(BlockResult(
                        file=file_rel, match_method=MatchMethod.NOT_FOUND,
                        ok=False, error=err,
                    ))
                all_warnings.append(err or f"could not resolve {file_rel}")
                continue

            results, warnings, written = _apply_file_blocks(
                target, file_blocks,
                fuzzy=self.fuzzy,
                fuzzy_threshold=self.fuzzy_threshold,
                dry_run=self.dry_run,
                run_black=self.run_black,
                on_block_fail=self.on_block_fail,
            )
            all_results.extend(results)
            all_warnings.extend(warnings)
            if written:
                updated_files.append(file_rel)

        method_counts: Dict[str, int] = defaultdict(int)
        for r in all_results:
            method_counts[r.match_method.value] += 1

        blocks_ok     = sum(1 for r in all_results if r.ok)
        blocks_failed = sum(1 for r in all_results if not r.ok) + dropped_count

        if blocks_failed == 0:
            status, ok, error = ApplyStatus.FULLY_APPLIED, True, None
        elif blocks_ok == 0:
            status, ok = ApplyStatus.FAILED, False
            error = all_warnings[0] if all_warnings else "apply_failed"
        else:
            status, ok = ApplyStatus.PARTIALLY_APPLIED, False
            error = all_warnings[0] if all_warnings else "partially_applied"

        return PatchResult(
            ok=ok, apply_status=status, error=error,
            updated_files=updated_files, applied_any=len(updated_files) > 0,
            block_results=all_results, match_methods=dict(method_counts),
            dry_run=self.dry_run, warnings=all_warnings,
            warnings_count=len(all_warnings),
        )


# ── Public entry point ────────────────────────────────────────────────────────

def apply_patch(
    *,
    repo_dir:        Path,
    llm_output:      str,
    dry_run:         bool        = False,
    fuzzy:           bool        = True,
    fuzzy_threshold: float       = _DEFAULT_FUZZY_THRESHOLD,
    run_black:       bool        = True,
    on_block_fail:   OnBlockFail = OnBlockFail.SKIP,
) -> PatchResult:
    """
    Full pipeline: sanitize LLM output -> apply search/replace blocks.

    Args:
        repo_dir:        Root of the repository.
        llm_output:      Raw LLM output with search/replace blocks.
        dry_run:         Validate without writing files.
        fuzzy:           Allow fuzzy matching as last resort.
        fuzzy_threshold: Minimum difflib ratio for fuzzy (default 0.80).
        run_black:       Format modified .py files with black.
        on_block_fail:   ABORT | SKIP (default) | SKIP_FILE

    Returns:
        PatchResult — call .to_dict() for legacy-compatible plain dict.
    """
    san = SearchReplaceSanitizer.sanitize(llm_output)

    if not san["is_valid"]:
        dropped      = san.get("dropped_blocks", [])
        drop_reasons = [d["reason"] for d in dropped]
        return PatchResult(
            ok=False, apply_status=ApplyStatus.FAILED,
            error=san.get("error") or "sanitizer_rejected",
            updated_files=[], applied_any=False,
            block_results=[], match_methods={},
            dry_run=dry_run, stage="sanitize",
            sanitizer_reason=san.get("reason"),
            sanitizer_dropped_blocks=dropped,
            warnings=drop_reasons, warnings_count=len(drop_reasons),
        )

    dropped = san.get("dropped_blocks", [])
    result  = Applier(
        repo_dir=repo_dir, dry_run=dry_run,
        fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold,
        run_black=run_black, on_block_fail=on_block_fail,
    ).apply(san["blocks"], dropped_count=len(dropped))

    drop_reasons = [d["reason"] for d in dropped]
    result.sanitizer_reason         = san.get("reason")
    result.sanitizer_dropped_blocks = dropped
    if drop_reasons:
        result.warnings       = drop_reasons + result.warnings
        result.warnings_count = len(result.warnings)

    return result