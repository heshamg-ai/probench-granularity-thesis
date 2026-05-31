from typing import Dict, List, Any
from pathlib import Path
import json
import logging
from argparse import ArgumentParser

from probench.prep.data_loader import *
from probench.prep.prompt.prompt_tokenizer import get_tokenizer, count_tokens, truncate_to_token_budget
from probench.prep.prompt.oracle_context_validator import OracleContextValidator
from probench.prep.utils import normalize_prompt_dataset_meta

from probench.prep.prompt.prompt_frame import (
    build_prompt_prefix,
    build_prompt_suffix,
    build_code_signatures,
    build_module_globals,
    build_imports,
)

logger = logging.getLogger(__name__) 
logging.basicConfig(level=logging.INFO)

class OraclePromptBuilder:
    """
        OraclePromptBuilder assembles model prompts from precomputed oracle context.
        It performs no retrieval and no static analysis at prompt-build time.

        Core contract
        -------------
        - Input: one dataset row containing ``oracle_context``.
        - Output: a single prompt string composed of:
        1) a fixed prefix (task + instructions),
        2) per-file context sections, and
        3) a fixed suffix (response format + constraints),
        all subject to a global token budget.

        Non-goals (by design)
        ---------------------
        - No file retrieval, search, or ranking.
        - No call-graph construction, dependency inference, or cross-file linking.
        - No code parsing beyond what is already present in ``oracle_context``.

        Granularity modes
        -----------------
        1) ``full_file``
        - Injects the complete oracle file dump from ``raw_code``.
        - No structural slicing, filtering, or extraction.
        - File-level augmentations (imports, module globals, signatures) are disabled.

        2) ``class_and_function_level``
        - Structured, scoped context built from extracted ``code_units``.
        - If changed classes are available in the patch, their full class bodies are included.
        - If changed functions are available in the patch, their full function bodies are included.
        - May prepend optional per-file metadata:
            imports, module globals, change summary, and code signatures.

        3) ``method_and_function_level``
        - Minimal, change-focused context with lightweight class scaffolding.
        - If changed methods are available in the patch, their method bodies are
            included together with parent-class scaffolding (class signature, class
            attributes, and ``__init__``).
        - If changed functions are available in the patch, their full function
            bodies are included.
        - May prepend optional per-file metadata:
            imports, module globals, and code signatures.
        - Full class bodies are not included, even when a class itself changed;
            only selected methods plus minimal surrounding structure are shown.

        Fallbacks
        ---------
        - For ``class_and_function_level`` and ``method_and_function_level`` only:
        if baseline code for a changed entity is not available because it was
        newly added in the patch, the builder falls back to listing the added
        entities from ``change_summary.added`` (per class or per file) instead
        of including code bodies.
        
        Notes
        -----
        - “Changed” / “impacted” entities are determined upstream via
        ``oracle_context`` and ``change_summary``.
        - This builder only formats and packs the provided context under
        the specified token budget.
    """

    def __init__(
        self,
        row: Dict[str, Any],
        *,
        granularity: str = "method_and_function_level",            # "full_file" | "class_and_function_level" | "method_and_function_level"
        max_tokens: int = 8000,
        include_module_globals: bool = False,
        include_change_summary: bool = False,
        include_code_signatures: bool = False,
        include_imports: bool = False
    ):
        self.row = row
        self.granularity = granularity if granularity in ("full_file", "class_and_function_level", "method_and_function_level") else "method_and_function_level"
        self.max_tokens = max_tokens

        # Metadata only matters for class_and_function_level/method_and_function_level
        self.include_module_globals = include_module_globals
        self.include_code_signatures = include_code_signatures
        self.include_imports = include_imports

        # PR-level metadata
        self.include_change_summary = include_change_summary

        # Full_file shows entire file, metadata blocks would be redundant.
        if self.granularity == "full_file":
            self.include_imports = False
            self.include_module_globals = False
            self.include_code_signatures = False

        baseline_ctx = row.get("oracle_context") or {}
        self.file_contexts = baseline_ctx.get("file_context", [])

    def _format_change_item(self, item: Any) -> str:
        """Format entry from change_summary.{added,removed,modified}."""

        if isinstance(item, str):
            return f"- `{item}`"

        if isinstance(item, dict):
            qn = (item.get("qualified_name") or "").strip()
            cat = (item.get("category") or "").strip()
            parent = (item.get("parent_class") or "").strip()

            if cat == "method" and parent:
                return f"- `{qn}` (method of `{parent}`)"
            if cat:
                return f"- `{qn}` ({cat})"
            return f"- `{qn}`"

        return f"- `{str(item)}`"


    def _added_methods_for_class(self, file_unit: Dict[str, Any], class_qname: str) -> List[str]:
        """Return qualified names of methods added for a given parent class."""

        cs = file_unit.get("change_summary") or {}
        if not isinstance(cs, dict):
            return []

        added = cs.get("added") or []
        if not isinstance(added, list):
            return []

        out: List[str] = []
        for it in added:
            if not isinstance(it, dict):
                continue
            if it.get("category") != "method":
                continue
            if (it.get("parent_class") or "") != class_qname:
                continue

            qn = it.get("qualified_name")
            if qn:
                out.append(qn)

        # de-dup
        return list(dict.fromkeys(out))

    def build_change_summary(self, file_unit: Dict[str, Any]) -> str:
        """Build a markdown summary of added/removed/modified entities for a file."""

        summary = file_unit.get("change_summary")
        if not isinstance(summary, dict) or not summary:
            return ""

        parts: List[str] = ["#### Change Summary"]

        for key in ("added", "removed", "modified"):
            items = summary.get(key) or []
            if not isinstance(items, list) or not items:
                continue

            lines = [s for s in (self._format_change_item(i) for i in items) if s]
            if lines:
                parts.append(f"**{key.capitalize()}:**\n" + "\n".join(lines))
        return "\n".join(parts)

    def build_optional_metadata(self, file_unit: Dict[str, Any]) -> str:
        """
        Build optional file-level metadata blocks (imports, globals, signatures) with PR-level change summary.
        """
        parts: List[str] = []

        if self.include_change_summary:
            b = self.build_change_summary(file_unit)
            if b:
                parts.append(b)

        if self.include_imports:
            b = build_imports(file_unit)
            if b:
                parts.append(b)

        if self.include_module_globals:
            b = build_module_globals(file_unit)
            if b:
                parts.append(b)

        if self.include_code_signatures:
            b = build_code_signatures(file_unit)
            if b:
                parts.append(b)

        return "\n".join(parts)

    def build_full_file_granularity(self) -> List[Dict[str, Any]]:
        """
        Full_file granularity

        Includes:
        - raw_code
        - optional change_summary
        """
        sections: List[Dict[str, Any]] = []

        for file_unit in self.file_contexts:
            file_path = file_unit.get("file_path")
            raw_code = file_unit.get("raw_code")

            if not isinstance(raw_code, str) or not raw_code:
                continue
            raw_code = raw_code.rstrip("\n")

            parts: List[str] = []

            if self.include_change_summary:
                cs = self.build_change_summary(file_unit)
                if cs:
                    parts.append(cs)

            # IMPORTANT: in full_file mode we do NOT include imports/module_globals/signatures.
            parts.append("### Full file context")
            parts.append(f"```python\n{raw_code}\n```")

            sections.append({"file_path": file_path, "text": "\n".join(parts)})

        return sections

    def build_method_and_function_level_granularity(self) -> List[Dict[str, Any]]:
        """
        Method and function level granularity:

        Includes:
        - class scaffolding (signature, attributes, __init__) plus changed methods
        - changed functions
        - fallbacks when only added entities are available
        """
        sections: List[Dict[str, Any]] = []

        for file_unit in self.file_contexts:
            file_path = file_unit.get("file_path")
            code_units = file_unit.get("code_units", []) or []

            # Build method_and_function_level blocks (methods + functions)
            blocks: List[str] = []

            for unit in code_units:
                category = unit.get("category")
                qname = unit.get("qualified_name")
                if not category or not qname:
                    continue

                # ----- class meta for methods -----
                if category == "class":
                    class_sig = unit.get("class_signature")
                    class_init = unit.get("init_code")
                    class_globals = unit.get("class_vars")
                    methods = unit.get("methods", []) or []

                    scaffold_parts: List[str] = []

                    if class_sig:
                        scaffold_parts.append(
                            "**Class Signature**\n"
                            "```python\n"
                            f"{str(class_sig).rstrip()}\n"
                            "```"
                        )

                    if class_globals:
                        scaffold_parts.append(
                            "**Class Attributes**\n"
                            "```python\n"
                            f"{str(class_globals).rstrip()}\n"
                            "```"
                        )

                    if class_init:
                        scaffold_parts.append(
                            "**Class __init__ method**\n"
                            "```python\n"
                            f"{str(class_init).rstrip()}\n"
                            "```"
                        )

                    scaffold = "\n".join(scaffold_parts).strip()
                    method_blocks: List[str] = []
                    for m in methods:
                        m_code = (m.get("code") or "").strip()
                        m_qname = m.get("qualified_name")
                        if not m_code or not m_qname:
                            continue

                        method_blocks.append(
                            f"##### Method `{m_qname}`\n"
                            f"```python\n{m_code}\n```"
                        )

                    method_blocks = list(dict.fromkeys(method_blocks))

                    if method_blocks:
                        class_header = f"#### Class `{qname}`"
                        if scaffold:
                            blocks.append(class_header + "\n" + scaffold + "\n" + "\n".join(method_blocks))
                        else:
                            blocks.append(class_header + "\n" + "\n".join(method_blocks))

                    else:
                        # fallback: no method bodies, but maybe patch added methods
                        added_methods = self._added_methods_for_class(file_unit, qname)
                        if added_methods:
                            class_header = f"#### Class `{qname}`"
                            fallback = "**Added methods (baseline missing):**\n" + "\n".join(f"- `{m}`" for m in added_methods)
                            if scaffold:
                                blocks.append(class_header + "\n" + scaffold + "\n" + fallback)
                            else:
                                blocks.append(class_header + "\n" + fallback)

                # ----- functions -----
                elif category == "function":
                    code = (unit.get("code") or "").strip()
                    if code:
                        blocks.append(
                            f"#### Function `{qname}`\n```python\n{code}\n```"
                        )

            seen = set()
            blocks = [x for x in blocks if not (x in seen or seen.add(x))]
            
            if not blocks:
                cs = file_unit.get("change_summary") or {}
                added = cs.get("added") or []
                if isinstance(added, list) and added:
                    lines = [self._format_change_item(x) for x in added]
                    blocks = [
                        "**New code only:** No baseline bodies or inlined source are available for this file in this record.\n"
                        "Added items (summary):\n"
                        + "\n".join(lines)]
                else:
                    continue

            context_parts: List[str] = ["### Relevant context"]

            meta = self.build_optional_metadata(file_unit)
            if meta:
                context_parts.append(meta)
            context_parts.append("\n".join(blocks))

            sections.append({
                "file_path": file_path,
                "text": "\n".join(context_parts),
            })

        return sections

    def build_class_and_function_level_granularity(self) -> List[Dict[str, Any]]:
        """
        Class and function level granularity:

        Includes:
        - full bodies of relevant top-level classes and functions
        - fallbacks when only added entities are available
        """
        sections: List[Dict[str, Any]] = []

        for file_unit in self.file_contexts:
            file_path = file_unit.get("file_path")
            code_units = file_unit.get("code_units", []) or []

            # Build class_and_function_level blocks (top-level classes/functions)
            blocks: List[str] = []

            for unit in code_units:
                category = unit.get("category")
                qname = unit.get("qualified_name")
                if not category or not qname:
                    continue

                code = (unit.get("code") or "").strip()
                if not code:
                    continue

                if category == "class":
                    header = f"#### Class `{qname}`"
                elif category == "function":
                    header = f"#### Function `{qname}`"
                else:
                    continue

                blocks.append(f"{header}\n```python\n{code}\n```")

            seen = set()
            blocks = [x for x in blocks if not (x in seen or seen.add(x))]

            if not blocks:
                cs = file_unit.get("change_summary") or {}
                added = cs.get("added") or []
                if isinstance(added, list) and added:
                    lines = [self._format_change_item(x) for x in added]
                    blocks = [
                        "**New code only:** No baseline bodies or inlined source are available for this file in this record.\n"
                        "Added items (summary):\n"
                        + "\n".join(lines)]
                else:
                    continue

            # Assemble per-file section
            context_parts: List[str] = ["### Relevant context"]

            meta = self.build_optional_metadata(file_unit)
            if meta:
                context_parts.append(meta)
            context_parts.append("\n".join(blocks))

            sections.append({
                "file_path": file_path,
                "text": "\n".join(context_parts),
            })

        return sections

    def build_context(self) -> List[Dict[str, Any]]:
        """Dispatch to the appropriate context builder based on granularity."""

        if self.granularity == "full_file":
            return self.build_full_file_granularity()
        elif self.granularity == "class_and_function_level":
            return self.build_class_and_function_level_granularity()
        elif self.granularity == "method_and_function_level":
            return self.build_method_and_function_level_granularity()
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")

    def build_prompt_with_budget(
        self,
        *,
        tokenizer_spec,
    ) -> str:
        """
        Build a full prompt (prefix + context + suffix).

        Strategy:
        1. Build the complete prompt and return it immediately if it already fits.
        2. Otherwise, reserve budget for prefix/suffix and pack context chunks incrementally.
        3. If the last chunk does not fully fit, truncate only that chunk.
        """

        sections = self.build_context()
        if not sections:
            raise ValueError("no context sections generated")

        prefix = build_prompt_prefix(self.row)
        suffix = build_prompt_suffix()
        sep = "\n\n"

        prepared_chunks: List[Dict[str, str]] = []

        for sec in sections:
            path = sec.get("file_path")
            code = sec.get("text")

            if not path or code is None:
                continue

            code_str = str(code).rstrip()
            if not code_str:
                continue

            start_marker = f"[start of {path}]\n"
            end_marker = f"\n[end of {path}]\n"
            full_chunk = start_marker + code_str + end_marker

            prepared_chunks.append(
                {
                    "file_path": path,
                    "start_marker": start_marker,
                    "end_marker": end_marker,
                    "code_str": code_str,
                    "full_chunk": full_chunk,
                }
            )

        if not prepared_chunks:
            raise ValueError("No code fits into prompt")

        # Phase 1: try the full prompt first

        full_context = sep.join(chunk["full_chunk"] for chunk in prepared_chunks)
        full_prompt = sep.join([prefix, full_context, suffix])
        tokens = count_tokens(full_prompt, tokenizer_spec)
        if tokens <= self.max_tokens:
            return full_prompt

        # Phase 2: budget-aware packing
        sep_cost = count_tokens(sep, tokenizer_spec)

        # prefix <sep> context <sep> suffix  => 2 fixed separators
        fixed_tokens = (
            count_tokens(prefix, tokenizer_spec)
            + count_tokens(suffix, tokenizer_spec)
            + 2 * sep_cost
        )

        remaining = self.max_tokens - fixed_tokens
        if remaining <= 0:
            raise ValueError("Token budget too small even for prefix+suffix")

        selected_chunks: List[str] = []
        used = 0

        for chunk in prepared_chunks:
            full_chunk = chunk["full_chunk"]
            join_cost = sep_cost if selected_chunks else 0
            chunk_cost = count_tokens(full_chunk, tokenizer_spec)
            extra = join_cost + chunk_cost

            # Case 1: whole chunk fits
            if used + extra <= remaining:
                selected_chunks.append(full_chunk)
                used += extra
                continue

            # Case 2: only part of this final chunk can fit
            budget_for_this_chunk = remaining - used - join_cost
            if budget_for_this_chunk <= 0:
                break

            start_marker = chunk["start_marker"]
            end_marker = chunk["end_marker"]
            code_str = chunk["code_str"]

            marker_cost = (
                count_tokens(start_marker, tokenizer_spec)
                + count_tokens(end_marker, tokenizer_spec)
            )
            budget_for_code = budget_for_this_chunk - marker_cost
            if budget_for_code <= 0:
                break

            truncated_code = truncate_to_token_budget(
                code_str,
                tokenizer_spec,
                budget_for_code,
            ).rstrip()

            # avoid ending on a broken last line
            if "\n" in truncated_code:
                truncated_code = truncated_code.rsplit("\n", 1)[0].rstrip()

            if truncated_code:
                selected_chunks.append(start_marker + truncated_code + end_marker)

            break

        if not selected_chunks:
            raise ValueError("No code fits into token budget")

        context = sep.join(selected_chunks)
        prompt = sep.join([prefix, context, suffix])

        # Final safety check
        if count_tokens(prompt, tokenizer_spec) > self.max_tokens:
            prompt = truncate_to_token_budget(prompt, tokenizer_spec, self.max_tokens).rstrip()

        final_tokens = count_tokens(prompt, tokenizer_spec)
        logger.debug(
            "pr=%s prompt_tokens=%d budget=%d utilization=%.1f%%",
            self.row.get("pr_number", "?"),
            final_tokens,
            self.max_tokens,
            100 * final_tokens / self.max_tokens,
        )

        return prompt

def build_output_filename(
    *,
    granularity: str,
    tokenizer_name: str,
    max_tokens: int,
    include_imports: bool,
    include_module_globals: bool,
    include_code_signatures: bool,
    include_change_summary: bool,
    prefix: str = "dataset") -> str:

    """Return an output filename that encodes the prompt configuration."""

    return (
        f"{prefix}"
        f"__gran={granularity}"
        f"__tok={tokenizer_name}"
        f"__max={int(max_tokens)}"
        f"__imports={bool(include_imports)}"
        f"__globals={bool(include_module_globals)}"
        f"__sigs={bool(include_code_signatures)}"
        f"__changes={bool(include_change_summary)}"
        f".json"
    )

def build_prompt_dataset(
    *,
    dataset_name_or_path: str,
    output_path: str,
    tokenizer_name: str = "cl100k_base",
    max_tokens: int = 8000,
    granularity: str = "method_and_function_level",
    include_module_globals: bool = False,
    include_imports: bool = False,
    include_code_signatures: bool = False,
    include_change_summary: bool = False):

    """Build oracle prompts for each dataset row and write them to a JSON file."""

    tokenizer_spec = get_tokenizer(tokenizer_name)
    loader = DatasetLoader(dataset_name_or_path)
    validator = OracleContextValidator() if granularity == "full_file" else None

    rows_out: List[Dict[str, Any]] = []
    truncation_filtered = []

    for row in loader:
        scenario_id = row.get("scenario_id")
        ctx = row.get("oracle_context") or {}
        if not ctx:
            logger.warning(f"{scenario_id}: no usable context")
            continue

        prompter = OraclePromptBuilder(
            row,
            granularity=granularity,
            max_tokens=max_tokens,
            include_module_globals=include_module_globals,
            include_change_summary=include_change_summary,
            include_code_signatures=include_code_signatures,
            include_imports=include_imports
        )
        try:
            prompt = prompter.build_prompt_with_budget(
                        tokenizer_spec=tokenizer_spec)

        except Exception as e:
            logger.warning(f"{scenario_id}: {e}")
            continue

        prompt_tokens = count_tokens(prompt, tokenizer_spec)
        logger.info(f"{scenario_id}: prompt_tokens={prompt_tokens}")

        if granularity == "full_file":
            if not validator.is_valid(row, prompt):
                logger.warning(f"{scenario_id}: oracle context missing patch-relevant code")
                truncation_filtered.append(scenario_id)
                continue


        out_row = dict(row)
        out_row["text"] = prompt

        # Remove heavy oracle/baseline context from final dataset
        out_row.pop("oracle_context", None)

        rows_out.append(out_row)
    
    meta = normalize_prompt_dataset_meta({
        "builder": "oracle",
        "granularity": granularity,
        "tokenizer": tokenizer_name,
        "max_tokens": int(max_tokens),
        "include_module_globals": bool(include_module_globals),
        "include_imports": bool(include_imports),
        "include_code_signatures": bool(include_code_signatures),
        "include_change_summary": bool(include_change_summary),
        "dataset_path": dataset_name_or_path,
        "truncation_filtered_count": len(truncation_filtered),
        "truncation_filtered": truncation_filtered, 
    })

    output_obj = {"meta": meta, "samples": rows_out}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(rows_out)} prompts → {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate oracle prompts with token budget")

    # Core arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/dataset_25_04_finalized.json",
        help="Path to base dataset")
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="prompt_oracle_gpt52.json",
        help="Output JSON path. If omitted, it is auto-generated under <dataset_parent>/prompt_dataset/.")
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="cl100k",
        help="Tokenizer name")
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200_000,
        help="Max total tokens per prompt")

    # Granularity
    parser.add_argument(
        "--granularity",
        type=str,
        default="full_file",
        choices=["method_and_function_level", "class_and_function_level", "full_file"],
        help="Granularity level for prompt building")

    # PR-level augmentations
    parser.add_argument(
        "--include_change_summary",
        action="store_true",
        help="Include per-file change summary block in prompt")

    # File-level augmentations (non-full_file)
    parser.add_argument(
        "--include_module_globals",
        action="store_true",
        help="(method_and_function_level/class_and_function_level only) Add module-level globals to file context")
    
    parser.add_argument(
        "--include_imports",
        action="store_true",
        help="(method_and_function_level/class_and_function_level only) Add top-of-file imports to file context")
    
    parser.add_argument(
        "--include_code_signatures",
        action="store_true",
        help="(method_and_function_level/class_and_function_level only) Add code signatures to file context")

    args = parser.parse_args()

    # --- auto output naming in the same style as BM25 + fixed folder name ---
    if args.output_path is None:
        ds_path = Path(args.dataset_path).resolve()
        parent = ds_path.parent

        out_dir = parent / "prompt_dataset_oracle"
        out_dir.mkdir(parents=True, exist_ok=True)

        fname = build_output_filename(
            granularity=args.granularity,
            tokenizer_name=args.tokenizer,
            max_tokens=args.max_tokens,
            include_imports=args.include_imports,
            include_module_globals=args.include_module_globals,
            include_code_signatures=args.include_code_signatures,
            include_change_summary=args.include_change_summary,
            prefix="dataset",
        )
        args.output_path = str(out_dir / fname)

    build_prompt_dataset(
        dataset_name_or_path=args.dataset_path,
        output_path=args.output_path,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens,
        granularity=args.granularity,
        include_module_globals=args.include_module_globals,
        include_imports=args.include_imports,
        include_code_signatures=args.include_code_signatures,
        include_change_summary=args.include_change_summary,
    )

    """
    Example usage:
    python probench/prep/prompt/prompt_oracle_builder.py \
    --dataset_path data/merged_dataset_test_09_02.json \
    --output_path data/prompt_dataset/out.json \
    --tokenizer cl100k_base \
    --max_tokens 200000 \
    --granularity full_file \
    --include_change_summary \
    --include_module_globals \
    --include_imports \
    --include_code_signatures
    """