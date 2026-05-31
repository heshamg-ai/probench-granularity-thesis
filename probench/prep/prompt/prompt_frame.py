from __future__ import annotations
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Output format ─────────────────────────────────────────────────────────────

_FORMAT_SPEC = """\
OUTPUT FORMAT — SEARCH/REPLACE BLOCKS:

For each change emit exactly one block:

path/to/file.py
[SEARCH]
<code to find>
[/SEARCH]
[REPLACE]
<replacement code>
[/REPLACE]

RULES:
- Output ONLY the blocks. No prose, no explanations, no markdown fences.
- Multiple blocks are allowed — one per logical change, in any order.
- SEARCH must be copied VERBATIM from the file:
    * exact line breaks — never collapse a multiline signature to one line
    * exact indentation — do not add or remove leading whitespace
    * exact quotes — do not change ' to " or vice versa
    * exact spacing — do not add or remove spaces around = or ()
- SEARCH must be the SMALLEST snippet that uniquely identifies the location.
  Never include whole functions or docstrings unless the whole thing changes.
- REPLACE may be empty to delete the matched code.
- Multiple blocks on the same file are applied top-to-bottom.
"""


# ── Anchoring guidance ────────────────────────────────────────────────────────

_ANCHORING = """\
ANCHORING GUIDANCE:
- REPLACE: SEARCH = the lines to change, REPLACE = the new lines.
- ADD:     SEARCH = nearest unique surrounding line(s), REPLACE = those same
           lines plus the new code inserted before or after.
- DELETE:  SEARCH = the lines to remove, REPLACE = (empty).
- If a function signature spans multiple lines in the file, your SEARCH must
  span those exact same lines — do not rewrite it as a single line.
"""


# ── Examples ──────────────────────────────────────────────────────────────────

_EXAMPLES = """\
EXAMPLES:

# 1. Replace a function, method, or class body
mypackage/module.py
[SEARCH]
    def compute(self, x):
        return x * 2
[/SEARCH]
[REPLACE]
    def compute(self, x: int) -> int:
        return x * 3
[/REPLACE]

# 2. Replace a multiline signature
mypackage/module.py
[SEARCH]
    def process(
        self,
        value: int,
    ) -> int:
[/SEARCH]
[REPLACE]
    def process(
        self,
        value: int,
        scale: float = 1.0,
    ) -> int:
[/REPLACE]

# 3. Add a function, method, or class after an existing one
mypackage/module.py
[SEARCH]
    def existing_method(self):
        pass
[/SEARCH]
[REPLACE]
    def existing_method(self):
        pass

    def new_method(self):
        return True
[/REPLACE]

# 4. Delete a function, method, or class
mypackage/module.py
[SEARCH]
    def deprecated_method(self):
        do_old_thing()
        return result
[/SEARCH]
[REPLACE]
[/REPLACE]
"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_prompt_prefix(row: Dict[str, Any]) -> str:
    task        = (row.get("task")        or "").strip()
    entry_point = (row.get("entry_point") or "").strip()

    parts: List[str] = [
        "You are an expert Python performance engineer. "
        f"Your objective is to optimize {entry_point} for speed and memory "
        "efficiency without altering its behavior, public APIs, or test expectations.",
        "",
        "CONSTRAINTS:",
        "- Preserve exact semantics and outputs.",
        "- Preserve all public APIs and return types.",
        "- Do not modify tests or test expectations."
    ]

    if task:
        parts.append("")
        parts.append("OPTIMIZATION REQUEST:")
        if task:
            parts.append(task)
        if entry_point:
            parts.append(f"Optimization function: {entry_point}")

    return "\n".join(parts).rstrip() + "\n"


def build_prompt_suffix() -> str:
    return "\n\n".join([_FORMAT_SPEC, _ANCHORING, _EXAMPLES])

def build_imports(file_meta: Dict[str, Any]) -> str:
    """Build a fenced imports block from file-level metadata, or empty string if none."""

    imports_list = file_meta.get("imports") or []
    if not isinstance(imports_list, (list, tuple)):
        return ""
    imports_list = [str(i).strip() for i in imports_list if str(i).strip()]
    if not imports_list:
        return ""
    return "#### Imports\n```python\n" + "\n".join(imports_list) + "\n```"

def build_module_globals(file_meta: Dict[str, Any]) -> str:
    """Build a fenced module-globals block from file-level metadata, or empty string if none."""

    module_globals = (file_meta.get("module_globals") or "").strip()
    if not module_globals:
        return ""
    return f"#### Module Globals\n```python\n{module_globals}\n```"

def build_code_signatures(file_meta: Dict[str, Any]) -> str:
    """Build fenced code-signature blocks from file-level metadata, or empty string if none."""

    signatures = file_meta.get("code_signatures") or []
    if not isinstance(signatures, list) or not signatures:
        return ""

    blocks: List[str] = ["#### Code Signatures"]
    for sig in signatures:
        if not isinstance(sig, dict):
            continue

        qname = (sig.get("qualified_name") or "").strip()
        signature = (sig.get("signature") or "").rstrip()
        category = (sig.get("category") or "").strip()

        if not qname or not signature:
            continue

        cat = f" ({category})" if category else ""
        blocks.append(
            f"##### `{qname}`{cat}\n```python\n{signature}\n```"
        )

    return "" if len(blocks) == 1 else "\n".join(blocks)
