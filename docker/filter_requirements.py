# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from __future__ import annotations
import argparse, os, sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Filter a requirements.txt by excluding lines containing keywords.")
    p.add_argument("--in", dest="input_file",
                   default=os.getenv("INPUT_FILE", "/pandas/requirements-dev.txt"),
                   help="Input requirements file path (default: env INPUT_FILE or /pandas/requirements-dev.txt)")
    p.add_argument("--out", dest="output_file",
                   default=os.getenv("OUTPUT_FILE", "/pandas/requirements-light.txt"),
                   help="Output requirements file path (default: env OUTPUT_FILE or /pandas/requirements-light.txt)")
    p.add_argument("--exclude", dest="exclude",
                   default=os.getenv("EXCLUDE_KEYWORDS", "pyqt,qt,gui"),
                   help="Comma-separated keywords to exclude (case-insensitive).")
    return p.parse_args()

def should_exclude(line: str, keywords: list[str]) -> bool:
    lower = line.strip().lower()
    if not lower or lower.startswith("#"):
        return True
    return any(kw in lower for kw in keywords if kw)

def main():
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    keywords = [x.strip().lower() for x in (args.exclude or "").split(",") if x.strip()]

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 2

    lines = input_path.read_text(encoding="utf-8").splitlines(True)
    kept = [l for l in lines if not should_exclude(l, keywords)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(kept), encoding="utf-8")

    print(f"[INFO] Filtered {len(lines) - len(kept)} lines → {output_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
