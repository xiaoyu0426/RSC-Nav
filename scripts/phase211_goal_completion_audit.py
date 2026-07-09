from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_ORACLE_DIR = Path("outputs/phase210_live_oracle_visuals/mp3d_oracle_visuals_20260622-204418/eval")
LEGACY_ORACLE_DIR = Path("outputs/phase30_live_oracle_visuals/mp3d_oracle_visuals_20260622-204418/eval")
DEFAULT_REUSE_DIR = Path("outputs/phase29_live_memory_reuse/mp3d_reuse_pass_20260622-203909/eval")

REQUIRED_ORACLE_IMAGES = [
    "step_0036_final_bev.png",
    "step_0036_final_oracle.png",
    "step_0036_final_oracle_diff.png",
    "step_0036_final_semantic_bev.png",
]

REQUIRED_REUSE_IMAGES = [
    "037_saved_memory_step_0036_mem_0084_semantic_bev.png",
    "039_loaded_memory_step_0000_mem_0084_semantic_bev.png",
    "052_replay_saved_memory_step_0012_mem_0096_semantic_bev.png",
]

REQUIRED_LOG_MARKERS = [
    "Phase 2.4",
    "Phase 2.7 Live Path-Step Auto Evaluation",
    "Phase 2.8 Live Oracle Geometry Gate",
    "Phase 2.9 Live Memory Reuse / Reload Evaluation",
    "Phase 2.10 Live Oracle Visual Evidence",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit evidence for the live semantic-spatial memory goal.")
    parser.add_argument("--oracle-dir", default=str(DEFAULT_ORACLE_DIR))
    parser.add_argument("--reuse-dir", default=str(DEFAULT_REUSE_DIR))
    parser.add_argument("--log-path", default="IMPLEMENTATION_LOG.md")
    parser.add_argument("--out-dir", default="outputs/phase211_goal_audit")
    parser.add_argument("--min-oracle-free-iou", type=float, default=0.2)
    parser.add_argument("--min-oracle-occupied-f1", type=float, default=0.05)
    parser.add_argument("--min-classes", type=int, default=4)
    parser.add_argument("--min-memory-items", type=int, default=4)
    parser.add_argument("--max-tail-drift", type=float, default=0.8)
    args = parser.parse_args()

    oracle_dir = Path(args.oracle_dir)
    if oracle_dir == DEFAULT_ORACLE_DIR and not oracle_dir.exists() and LEGACY_ORACLE_DIR.exists():
        oracle_dir = LEGACY_ORACLE_DIR
    reuse_dir = Path(args.reuse_dir)
    log_path = Path(args.log_path)
    out_root = Path(args.out_dir)
    out_dir = out_root / f"goal_audit_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle_metrics = _read_json(oracle_dir / "metrics.json")
    reuse_metrics = _read_json(reuse_dir / "metrics.json")
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    checks = []
    checks.extend(_oracle_checks(oracle_metrics, args))
    checks.extend(_semantic_memory_checks(oracle_metrics, args))
    checks.extend(_reuse_checks(reuse_metrics))
    checks.extend(_image_checks(oracle_dir, REQUIRED_ORACLE_IMAGES, "oracle_visual"))
    checks.extend(_image_checks(reuse_dir, REQUIRED_REUSE_IMAGES, "reuse_visual"))
    checks.extend(_log_checks(log_text))
    checks.extend(_source_checks())

    passed = all(check["passed"] for check in checks)
    report = {
        "passed": passed,
        "oracle_dir": str(oracle_dir),
        "reuse_dir": str(reuse_dir),
        "log_path": str(log_path),
        "checks": checks,
        "summary": _summary(checks),
        "key_metrics": {
            "oracle": {
                "passed": oracle_metrics.get("passed"),
                "free_iou_observed": oracle_metrics.get("final_geometry_oracle", {}).get("free_iou_observed"),
                "occupied_f1_observed": oracle_metrics.get("final_geometry_oracle", {}).get("occupied_f1_observed"),
                "occupied_boundary_chamfer_m": oracle_metrics.get("final_geometry_oracle", {}).get("occupied_boundary_chamfer_m"),
            },
            "semantic_memory": {
                "class_coverage": oracle_metrics.get("class_coverage", {}),
                "num_items": oracle_metrics.get("final_memory", {}).get("num_items"),
                "mean_confidence": oracle_metrics.get("final_memory", {}).get("mean_confidence"),
                "mean_freshness": oracle_metrics.get("final_memory", {}).get("mean_freshness"),
                "max_tail_drift_m": oracle_metrics.get("object_stability", {}).get("max_tail_drift_m"),
            },
            "reuse": {
                "passed": reuse_metrics.get("passed"),
                "retained_after_load": reuse_metrics.get("retained_after_load", {}),
                "retained_after_replay": reuse_metrics.get("retained_after_replay", {}),
                "duplicate_item_ids": reuse_metrics.get("duplicate_item_ids", []),
                "updated_ids": reuse_metrics.get("updated_ids", []),
            },
        },
    }
    (out_dir / "goal_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary_html(out_dir, report)
    print(json.dumps(report, indent=2))


def _oracle_checks(metrics: dict, args) -> list[dict]:
    oracle = metrics.get("final_geometry_oracle", {})
    return [
        _check("oracle_metrics_passed", bool(metrics.get("passed")), metrics.get("passed")),
        _check("oracle_geometry_gate", bool(metrics.get("criteria", {}).get("geometry_ok")), metrics.get("criteria", {})),
        _check("oracle_enabled", bool(oracle.get("enabled")), oracle.get("enabled")),
        _check(
            "oracle_free_iou_threshold",
            float(oracle.get("free_iou_observed", 0.0)) >= args.min_oracle_free_iou,
            oracle.get("free_iou_observed"),
        ),
        _check(
            "oracle_occupied_f1_threshold",
            float(oracle.get("occupied_f1_observed", 0.0)) >= args.min_oracle_occupied_f1,
            oracle.get("occupied_f1_observed"),
        ),
    ]


def _semantic_memory_checks(metrics: dict, args) -> list[dict]:
    coverage = metrics.get("class_coverage", {})
    covered_classes = [name for name, count in coverage.items() if int(count) > 0]
    memory = metrics.get("final_memory", {})
    stability = metrics.get("object_stability", {})
    return [
        _check("semantic_all_target_classes_observed", len(covered_classes) >= args.min_classes, coverage),
        _check("semantic_gt_gate", bool(metrics.get("criteria", {}).get("semantic_ok")), metrics.get("final_semantic", {})),
        _check("object_memory_gate", bool(metrics.get("criteria", {}).get("memory_ok")), memory),
        _check("object_memory_item_count", int(memory.get("num_items", 0)) >= args.min_memory_items, memory.get("num_items")),
        _check("object_memory_has_confidence", float(memory.get("mean_confidence", 0.0)) > 0.0, memory.get("mean_confidence")),
        _check("object_memory_has_freshness", float(memory.get("mean_freshness", 0.0)) > 0.0, memory.get("mean_freshness")),
        _check("object_stability_gate", bool(metrics.get("criteria", {}).get("stability_ok")), stability),
        _check(
            "object_stability_tail_drift",
            float(stability.get("max_tail_drift_m", 999.0)) <= args.max_tail_drift,
            stability.get("max_tail_drift_m"),
        ),
    ]


def _reuse_checks(metrics: dict) -> list[dict]:
    return [
        _check("memory_reuse_passed", bool(metrics.get("passed")), metrics.get("criteria", {})),
        _check("memory_load_retained", bool(metrics.get("criteria", {}).get("load_retained_ok")), metrics.get("retained_after_load", {})),
        _check("memory_replay_retained", bool(metrics.get("criteria", {}).get("replay_retained_ok")), metrics.get("retained_after_replay", {})),
        _check("memory_no_duplicate_ids", bool(metrics.get("criteria", {}).get("duplicate_ok")), metrics.get("duplicate_item_ids", [])),
        _check("memory_replay_updated_items", bool(metrics.get("criteria", {}).get("update_ok")), metrics.get("updated_ids", [])),
        _check("memory_step_monotonic", bool(metrics.get("criteria", {}).get("memory_step_monotonic")), metrics.get("replay", {})),
    ]


def _image_checks(root: Path, names: list[str], label: str) -> list[dict]:
    checks = []
    for name in names:
        path = root / name
        detail: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        passed = False
        if path.exists():
            try:
                image = Image.open(path)
                detail.update({"size": list(image.size), "nonempty": image.getbbox() is not None, "bytes": path.stat().st_size})
                passed = bool(image.getbbox() is not None and path.stat().st_size > 0)
            except Exception as exc:
                detail["error"] = str(exc)
        checks.append(_check(f"{label}:{name}", passed, detail))
    return checks


def _log_checks(log_text: str) -> list[dict]:
    return [
        _check(f"log_contains:{marker}", marker in log_text, marker)
        for marker in REQUIRED_LOG_MARKERS
    ]


def _source_checks() -> list[dict]:
    paths = [
        "scripts/phase23_habitat_control_server.py",
        "scripts/phase24_bev_geometry_eval.py",
        "scripts/phase27_live_control_eval.py",
        "scripts/phase28_live_memory_reuse_eval.py",
        "src/dense_bev_mapper.py",
        "src/semantic_bev_memory.py",
        "src/object_memory_store.py",
    ]
    return [_check(f"source_exists:{path}", Path(path).exists(), path) for path in paths]


def _summary(checks: list[dict]) -> dict:
    return {
        "total": len(checks),
        "passed": sum(1 for check in checks if check["passed"]),
        "failed": [check["name"] for check in checks if not check["passed"]],
    }


def _check(name: str, passed: bool, detail: Any) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary_html(out_dir: Path, report: dict) -> None:
    rows = "\n".join(
        f"<tr><td>{check['name']}</td><td>{check['passed']}</td><td><pre>{json.dumps(check['detail'], ensure_ascii=False, indent=2)}</pre></td></tr>"
        for check in report["checks"]
    )
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Phase 2.11 Goal Completion Audit</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;line-height:1.4}}
pre{{white-space:pre-wrap;background:#f4f4f4;padding:8px}}
td,th{{vertical-align:top;padding:8px;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;width:100%}}
</style></head>
<body>
<h1>Phase 2.11 Goal Completion Audit</h1>
<p>Passed: <strong>{report['passed']}</strong></p>
<pre>{json.dumps(report['key_metrics'], indent=2)}</pre>
<table>
<tr><th>Check</th><th>Passed</th><th>Detail</th></tr>
{rows}
</table>
</body></html>
"""
    (out_dir / "summary.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
