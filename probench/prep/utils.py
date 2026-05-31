from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict
import logging
from pathlib import Path
from git import Repo
from datetime import datetime, timezone
import hashlib
import json


logger = logging.getLogger(__name__)

class ContextManager:
    """
    Context manager to reset a cloned Git repo to a specific commit.
    """

    def __init__(self, repo_path: str, base_commit: str, verbose: bool = False):
        self.repo_path = Path(repo_path).resolve().as_posix()
        self.base_commit = base_commit
        self.verbose = verbose
        self.repo = Repo(self.repo_path)

    def __enter__(self):
        """Reset the repo to `base_commit` and clean untracked files."""

        if self.verbose:
            logger.info(f"Switching to {self.base_commit} in {self.repo_path}")
        try:
            self.repo.git.reset("--hard", self.base_commit)
            self.repo.git.clean("-fdq")
        except Exception as e:
            logger.error(f"Failed to switch to {self.base_commit}")
            logger.error(e)
            raise e
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and self.verbose:
            logger.error(f"Error inside context: {exc_val}")
        return False

def clone_repo(repo_slug: str, root_dir: Path, token: Optional[str] = None, base_url: str = "https://github.com") -> Path:
    """
    Clone a GitHub repo into `root_dir` if it does not already exist.
    """
    repo_dir = root_dir / f"repo__{repo_slug.replace('/', '__')}"
    if repo_dir.exists():
        return repo_dir

    if token and token != "git":
        repo_url = f"{base_url.rstrip('/')}/{repo_slug}.git"
        # if you need token in URL, adjust here
    else:
        repo_url = f"{base_url.rstrip('/')}/{repo_slug}.git"

    logger.info(f"Cloning {repo_slug} into {repo_dir} from {repo_url}")
    Repo.clone_from(repo_url, repo_dir)
    return repo_dir

EXCLUDED_DIRS = {
    "ci",
    "example", "examples",
    "benchmark", "benchmarks", "bench",
    "benchmarks", "bench",
    "scripts",
    "doc", "docs"
}
EXCLUDED_FILENAMES = {
    # packaging / tooling
    "setup.py",
    "conftest.py",    
    "noxfile.py",
    "tasks.py",
    "invoke.py",
}
def is_optimization_candidate(path) -> bool:
    """
    Return True if the file is a plausible target for performance optimization.

    - Only .py files
    - Excluded packaging / tooling files
    - Excludes example/benchmark/script/doc/ci dirs (by directory part)
    - Excludes tests by directory part AND by filename pattern:
        - test_*.py
        - *_test.py
        - *_tests.py
    """
    if not isinstance(path, Path):
        path = Path(path)

    # Only .py
    if path.suffix.lower() != ".py":
        return False

    name = path.name.lower()
    parts = [p.lower() for p in path.parts]

    # Exclude known filenames
    if name in EXCLUDED_FILENAMES:
        return False

    # Exclude if any directory part is in excluded set
    if set(parts) & EXCLUDED_DIRS:
        return False

    # Exclude typical pytest file patterns
    # (pytest defaults: test_*.py and *_test.py)
    if name.startswith("test_"):
        return False
    if name.endswith(("_test.py", "_tests.py")):
        return False

    return True

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def normalize_prompt_dataset_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(meta)
    out.setdefault("schema_version", 1)
    out.setdefault("created_at", _utc_now_iso())

    output_meta = {
        "builder": out.get("builder"),
        "granularity": out.get("granularity"),
        "include_change_summary": out.get("include_change_summary"),
        "prompt_style": out.get("prompt_style"),
        "bm25_retrievals_path": out.get("bm25_retrievals_path"),
        #"patch_kind": out.get("patch_kind"),
        "tokenizer": out.get("tokenizer"),
        "max_tokens": out.get("max_tokens"),
        "include_imports": out.get("include_imports"),
        "include_module_globals": out.get("include_module_globals"),
        "include_code_signatures": out.get("include_code_signatures"),
        "dataset_path": out.get("dataset_path"),
    }
    s = json.dumps(output_meta, sort_keys=True, ensure_ascii=False)
    out["prompt_dataset_id"] = "pd_" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
    return out

def prompt_dataset_filename(meta: Dict[str, Any], prefix: str = "dataset") -> str:
    """
    Key=value filename style.

    Oracle:
      dataset__gran=...__tok=...__max=...__imports=...__globals=...__sigs=...__changes=....json

    BM25:
      dataset__bm25=...__tok=...__max=...__imports=...__globals=...__sigs=....json
      (no __changes because it's always off / not applicable)
    """
    builder = (meta.get("builder") or "").strip().lower()

    tok = meta.get("tokenizer") or meta.get("tokenizer_name") or "unknown"
    max_tokens = int(meta.get("max_tokens") or 0)

    include_imports = bool(meta.get("include_imports"))
    include_module_globals = bool(meta.get("include_module_globals"))
    include_code_signatures = bool(meta.get("include_code_signatures"))
    granularity = meta.get("granularity") or "unknown"

    parts = [prefix]

    if builder == "oracle":
        parts.append(f"changes={bool(meta.get('include_change_summary'))}")

    parts.extend([
        f"gran={granularity}",
        f"tok={tok}",
        f"max={max_tokens}",
        f"imports={include_imports}",
        f"globals={include_module_globals}",
        f"sigs={include_code_signatures}",
    ])

    return "__".join(parts) + ".json"

def prompt_preview(input_json, selected_scenario_id=None):
    input_json = Path(input_json)
    rows = json.loads(input_json.read_text(encoding="utf-8"))
    samples = rows.get("samples", [])

    row = None

    if selected_scenario_id:
        for r in samples:
            if str(r.get("scenario_id")) == str(selected_scenario_id):
                row = r
                break

    if row is None:
        row = samples[0]

    sid = str(row.get("scenario_id"))
    text = row.get("text", "")

    return sid, text

   

    