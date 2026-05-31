from typing import Dict, List, Any, Optional
import ast
import logging
import re

logger = logging.getLogger(__name__)

class OracleContextValidator:
    """
    Validate that truncated oracle context still contains baseline code
    for every patch-relevant entity.

    This is mainly used for full-file oracle prompts where the original
    files can be very large and must be truncated to fit a token budget.
    The validator checks that, after truncation, the baseline code blocks
    corresponding to modified/removed classes, methods, and functions are
    still present in the prompt.

    AST is used only to locate the start/end lines of those entities in
    the original file; the actual membership check is done via normalized
    text matching.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger

    def is_valid(self, row: Dict[str, Any], prompt: str) -> bool:
        """
        Check that every modified/removed baseline entry appears in the final prompt.

        - Uses oracle_context["file_context"] to collect change summaries.
        - For each modified/removed entity, extracts the exact class/method/
          function block from the baseline file via AST.
        - Normalizes both block and prompt and checks that the block text
          is still contained in `prompt`.
        """

        oracle_context = row.get("oracle_context") or {}
        file_context = oracle_context.get("file_context") or []

        if not isinstance(file_context, list) or not file_context:
            if self.logger:
                self.logger.warning("oracle_context.file_context is missing or empty; skipping validation")
            return True

        baseline_entries = self._collect_baseline_entries(row, file_context)
        if not baseline_entries:
            # No modified/removed entries to enforce → valid by definition.
            return True

        for entry in baseline_entries:
            ok, debug = self._entry_present(entry, file_context, prompt)   # <-- changed
            if not ok:
                if self.logger:
                    self.logger.warning(
                        f"Missing baseline {entry.get('category')} `{entry.get('qualified_name')}` in prompt. "
                        f"block_chars={debug.get('block_chars')} block_tokens≈{debug.get('block_tokens')} "
                        f"file={debug.get('file_path')}"
                    )
                return False

        return True

    # Change summary handling
    def _collect_baseline_entries(
        self,
        row: Dict[str, Any],
        file_context: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Collect modified and removed baseline entities (non-test) from file_context.

        For now this aggregates `change_summary.modified` and
        `change_summary.removed` from each file unit under
        `oracle_context.file_context`.
        """
        entries: List[Dict[str, Any]] = []

        for fu in file_context:
            cs = fu.get("change_summary") or {}
            if not isinstance(cs, dict):
                continue
            for key in ("modified", "removed"): #only qnames that is available under baseline
                items = cs.get(key) or []
                entries.extend(items)

        # Deduplicate by (qualified_name, category, parent_class)
        dedup: Dict[tuple, Dict[str, Any]] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            key = (
                e.get("qualified_name"),
                e.get("category"),
                e.get("parent_class"),
            )
            if key[0] is None:
                continue
            dedup[key] = e

        return list(dedup.values())

    def _normalize_for_match(self, text: str) -> str:
        """
        Normalize text for robust substring matching.

        - Normalize line endings to `\\n`.
        - Strip trailing whitespace per line.
        - Collapse all whitespace runs into single spaces.
        """

        # normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # strip trailing spaces per line
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines)

        # collapse all whitespace to single spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _entry_present(self, entry, file_context, prompt):
        """
        Check whether the baseline code block for `entry` is present in `prompt`.

        Extracts the block from each file_unit.raw_code via AST and tests
        normalized block text against the normalized prompt.
        """

        norm_prompt = self._normalize_for_match(prompt)
        best_debug = {"block_chars": None, "block_tokens": None, "file_path": None}

        for fu in file_context:
            raw_code = fu.get("raw_code")
            if not isinstance(raw_code, str):
                continue

            block = self._extract_entry_block_ast(raw_code, entry)
            if not block:
                continue

            norm_block = self._normalize_for_match(block)

            # fill debug from the first extracted candidate
            if best_debug["block_chars"] is None:
                best_debug["block_chars"] = len(block)
                best_debug["file_path"] = fu.get("file_path")

            if norm_block and norm_block in norm_prompt:
                return True, best_debug

        return False, best_debug

    # AST extraction 
    def _extract_entry_block_ast(
        self,
        source: str,
        entry: Dict[str, Any]) -> Optional[str]:
        """
        Given full file source and a change_summary entry,
        return the exact class/method/function block text.
        """

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()
        category = entry.get("category")

        # ----- Class-level change -----
        if category == "class":
            class_name = self._extract_class_name(entry)
            if not class_name:
                return None

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return self._slice(lines, node)

            return None

        # ----- Method-level change -----
        if category == "method":
            class_name = self._extract_class_name(entry)
            method_name = self._extract_method_name(entry)
            if not class_name or not method_name:
                return None

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if child.name == method_name:
                                return self._slice(lines, child)

            return None

        # ----- Function-level change -----
        if category == "function":
            func_name = self._extract_function_name(entry)
            if not func_name:
                return None

            # Only look at module-level functions
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                    return self._slice(lines, node)

            return None

        # Unknown category → skip
        return None

    # Name parsing utilities
    def _extract_class_name(self, entry: Dict[str, Any]) -> Optional[str]:
        """Determine the class name associated with a change entry."""

        if entry.get("category") == "class":
            qname = entry.get("qualified_name")
            if isinstance(qname, str):
                return qname.split(".")[-1]

        if entry.get("category") == "method":
            parent = entry.get("parent_class")
            if isinstance(parent, str):
                return parent.split(".")[-1]

        return None

    def _extract_method_name(self, entry: Dict[str, Any]) -> Optional[str]:
        """Determine the method name associated with a change entry."""

        if entry.get("category") == "method":
            qname = entry.get("qualified_name")
            if isinstance(qname, str):
                return qname.split(".")[-1]
        return None

    def _extract_function_name(self, entry: Dict[str, Any]) -> Optional[str]:
        """Determine the function name associated with a change entry."""

        if entry.get("category") == "function":
            qname = entry.get("qualified_name")
            if isinstance(qname, str):
                return qname.split(".")[-1]
        return None

    def _slice(self, lines: List[str], node: ast.AST) -> str:
        """Return the exact source span for the given AST node."""

        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return ""
        
        start = node.lineno - 1
        end = node.end_lineno
        return "\n".join(lines[start:end])
