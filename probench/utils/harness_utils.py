import importlib
import re
import json
from pathlib import Path
from typing import Any, Dict

_TAG_RE = re.compile(r"^(?P<repo>[a-zA-Z0-9_]+)-(?P<pr>\d+)$")

def id_to_scenario_class(tag: str) -> type:
    """
    Resolve a scenario tag to its corresponding scenario class.

    Expected tag format
    -------------------
    '<repo>-<pr>' → module 'probench.scenarios.<repo>.pr<pr>'
    with a class named 'PR<pr>Scenario' inside that module.

    Example:
      'pandas-50778' -> probench.scenarios.pandas.pr50778.PR50778Scenario
    """
    m = _TAG_RE.match(tag)
    if not m:
        raise ValueError(f"Invalid scenario_id '{tag}', expected '<repo>-<pr>'.")

    repo = m.group("repo")
    pr = m.group("pr")

    module_name = f"probench.scenarios.{repo}.pr{pr}"
    class_name = f"PR{pr}Scenario"

    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import module '{module_name}' for tag '{tag}'. "
            f"Expected file at 'probench/scenarios/{repo}/pr{pr}.py'."
        ) from e

    try:
        return getattr(mod, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' loaded but class '{class_name}' not found. "
            f"Ensure the class name matches '{class_name}'."
        ) from e

def load_experiment_meta(inference_run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(inference_run_dir)
    meta_path = run_dir / "INFERENCE_META.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing INFERENCE_META.json in: {run_dir}")

    meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))

    # Minimal fields harness needs:
    run_id = meta.get("run_id") or run_dir.name
    return {
        "run_id": run_id,
        "provider": meta.get("provider"),
        "model": meta.get("model"),
        "temperature": meta.get("temperature"),
        "top_p": meta.get("top_p"),
        "max_tokens": meta.get("max_tokens"),
        "stop": meta.get("stop"),
        "prompt_dataset_path": meta.get("prompt_dataset_path"),
        "prompt_dataset_meta": meta.get("prompt_dataset_meta") or {},
        "raw": meta,
        "meta_source": str(meta_path),
    }