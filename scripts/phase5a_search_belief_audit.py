from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from search_belief_audit import audit_search_belief_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a Phase5A VLM-guided search-belief run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument(
        "--require-both-support-outcomes",
        action="store_true",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    report = audit_search_belief_run(
        args.run_dir,
        require_api=bool(args.require_api),
        require_both_support_outcomes=bool(
            args.require_both_support_outcomes
        ),
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).expanduser().resolve().write_text(
            output,
            encoding="utf-8",
        )
    print(output)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
