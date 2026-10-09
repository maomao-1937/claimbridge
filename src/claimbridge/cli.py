"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import audit
from .model import LocalNLI


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an answer's inline [source-id] citations locally")
    parser.add_argument("input", type=Path, help="JSON file containing answer and sources")
    parser.add_argument("--out", type=Path, help="write JSON report here (default: stdout)")
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        report = audit(payload, LocalNLI())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
