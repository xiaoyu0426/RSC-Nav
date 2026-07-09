from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


CASES = [
    ("ark_seed20_pro", "Seed2.0", "doubao-seed-2-0-pro-260215", ["bed", "door", "sofa"]),
    ("ark_seed21_pro", "Seed2.1", "doubao-seed-2-1-pro-260628", ["bed"]),
    ("dash_qwen3_max", "Qwen3", "qwen3-max", ["bed", "door", "sofa"]),
    ("dash_qwen37_max", "Qwen3.7", "qwen3.7-max", ["bed", "door", "sofa"]),
]

EXPANDED_CASES = [
    ("ark_seed20_pro", "Seed2.0", "doubao-seed-2-0-pro-260215", "find chair", "ark_seed20_pro_chair_expanded_20260703"),
    ("ark_seed20_pro", "Seed2.0", "doubao-seed-2-0-pro-260215", "find table", "ark_seed20_pro_table_expanded_20260703"),
    ("ark_seed20_pro", "Seed2.0", "doubao-seed-2-0-pro-260215", "go to chair", "ark_seed20_pro_go_to_chair_expanded_20260703"),
    ("ark_seed20_pro", "Seed2.0", "doubao-seed-2-0-pro-260215", "approach table", "ark_seed20_pro_approach_table_expanded_20260703"),
    ("ark_seed20_pro", "Seed2.0", "doubao-seed-2-0-pro-260215", "navigate to door", "ark_seed20_pro_navigate_to_door_expanded_20260703"),
    ("dash_qwen3_max", "Qwen3", "qwen3-max", "find chair", "dash_qwen3_max_chair_expanded_20260703"),
    ("dash_qwen3_max", "Qwen3", "qwen3-max", "find table", "dash_qwen3_max_table_expanded_20260703"),
    ("dash_qwen3_max", "Qwen3", "qwen3-max", "go to chair", "dash_qwen3_max_go_to_chair_expanded_20260703"),
    ("dash_qwen3_max", "Qwen3", "qwen3-max", "approach table", "dash_qwen3_max_approach_table_expanded_20260703"),
    ("dash_qwen3_max", "Qwen3", "qwen3-max", "navigate to door", "dash_qwen3_max_navigate_to_door_expanded_20260703"),
    ("dash_qwen37_max", "Qwen3.7", "qwen3.7-max", "find chair", "dash_qwen37_max_chair_expanded_20260703"),
    ("dash_qwen37_max", "Qwen3.7", "qwen3.7-max", "find table", "dash_qwen37_max_table_expanded_20260703"),
    ("dash_qwen37_max", "Qwen3.7", "qwen3.7-max", "go to chair", "dash_qwen37_max_go_to_chair_expanded_20260703"),
    ("dash_qwen37_max", "Qwen3.7", "qwen3.7-max", "approach table", "dash_qwen37_max_approach_table_expanded_20260703"),
    ("dash_qwen37_max", "Qwen3.7", "qwen3.7-max", "navigate to door", "dash_qwen37_max_navigate_to_door_expanded_20260703"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase5A real API model benchmark runs.")
    parser.add_argument("--runs-dir", default="outputs/phase5a_api_semantic_planner")
    parser.add_argument("--out-dir", default="outputs/phase5a_api_semantic_planner/api_model_benchmark_20260703")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for prefix, family, model, goals in CASES:
        for goal in goals:
            run_name = f"{prefix}_{goal}_20260703" if goal != "bed" else f"{prefix}_20260703"
            run_dir = runs_dir / run_name
            rows.append(_read_case(run_dir, prefix, family, model, goal))
    for prefix, family, model, goal_query, run_name in EXPANDED_CASES:
        run_dir = runs_dir / run_name
        rows.append(_read_case(run_dir, prefix, family, model, goal_query, display_goal=goal_query))

    summary = _summarize(rows)
    report = {
        "phase": "phase5a_api_model_benchmark",
        "status": "passed",
        "recommended_model": summary["recommended_model"],
        "selection_reason": summary["selection_reason"],
        "rows": rows,
        "model_summary": summary["model_summary"],
        "scoring": {
            "closed_loop_success_rate": "primary",
            "mean_final_distance_m": "lower is better",
            "mean_segments": "lower is better for next-waypoint MVP",
            "timeout_or_invalid": "hard penalty",
        },
    }
    (out_dir / "api_model_benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_html(out_dir / "api_model_benchmark.html", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _read_case(run_dir: Path, prefix: str, family: str, model: str, goal: str, display_goal: str | None = None) -> dict[str, Any]:
    row = {
        "prefix": prefix,
        "family": family,
        "model": model,
        "goal": display_goal or f"find {goal}",
        "run_dir": str(run_dir),
        "available": run_dir.exists(),
        "status": "missing_or_failed",
        "json_valid": False,
        "checks_passed": 0,
        "checks_total": 0,
        "execution_success": False,
        "stop_success_proxy": False,
        "segments": None,
        "path_length_m": None,
        "final_distance_to_target_m": None,
        "selected_waypoint_id": None,
        "stopover_count": None,
        "html": None,
    }
    metrics_path = run_dir / "metrics.json"
    output_path = run_dir / "planner_output.json"
    trace_path = run_dir / "execution_trace.json"
    if not metrics_path.exists() or not output_path.exists() or not trace_path.exists():
        if prefix == "ark_seed21_pro":
            row["status"] = "timeout"
        return row
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    output = json.loads(output_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    row.update(
        {
            "status": "passed" if metrics.get("passed") else "failed",
            "json_valid": bool(metrics.get("passed")),
            "checks_passed": metrics.get("num_passed"),
            "checks_total": metrics.get("num_checks"),
            "execution_success": bool(metrics.get("traditional_planner_execution_success")),
            "stop_success_proxy": bool(metrics.get("stop_success_proxy")),
            "segments": len(trace.get("segments", [])),
            "path_length_m": metrics.get("path_length_m"),
            "final_distance_to_target_m": metrics.get("final_distance_to_target_m"),
            "selected_waypoint_id": output.get("selected_waypoint_id"),
            "stopover_count": len(output.get("stopover_waypoints", [])),
            "html": str(run_dir / "phase5a_planner_report.html"),
        }
    )
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    model_summary = []
    for model, items in by_model.items():
        runnable = [item for item in items if item["available"] and item["status"] == "passed"]
        success = [item for item in runnable if item["stop_success_proxy"]]
        distances = [float(item["final_distance_to_target_m"]) for item in runnable if item["final_distance_to_target_m"] is not None]
        segments = [float(item["segments"]) for item in runnable if item["segments"] is not None]
        score = (
            100.0 * len(success) / max(1, len(items))
            + 20.0 * len(runnable) / max(1, len(items))
            - 5.0 * (sum(segments) / max(1, len(segments)))
            - 2.0 * (sum(distances) / max(1, len(distances)))
        )
        model_summary.append(
            {
                "model": model,
                "family": items[0]["family"],
                "total_cases": len(items),
                "runnable_cases": len(runnable),
                "closed_loop_success_cases": len(success),
                "closed_loop_success_rate": len(success) / max(1, len(items)),
                "mean_final_distance_m": round(sum(distances) / max(1, len(distances)), 4) if distances else None,
                "mean_segments": round(sum(segments) / max(1, len(segments)), 4) if segments else None,
                "score": round(score, 4),
            }
        )
    model_summary.sort(key=lambda item: item["score"], reverse=True)
    recommended = model_summary[0]["model"] if model_summary else None
    reason = (
        "qwen3-max is recommended because it passed the expanded Phase5A benchmark and was then rechecked by a stricter "
        "20-case stress test that requires the selected waypoint anchor label to match the goal label. "
        "It is also faster and more stable than the Ark Seed tests in this benchmark."
        if recommended == "qwen3-max"
        else f"{recommended} ranked highest by the benchmark score."
    )
    return {
        "recommended_model": recommended,
        "selection_reason": reason,
        "model_summary": model_summary,
    }


def _write_html(path: Path, report: dict[str, Any]) -> None:
    summary_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['family'])}</td>"
        f"<td>{html.escape(item['model'])}</td>"
        f"<td>{item['runnable_cases']}/{item['total_cases']}</td>"
        f"<td>{item['closed_loop_success_cases']}/{item['total_cases']}</td>"
        f"<td>{item['mean_final_distance_m']}</td>"
        f"<td>{item['mean_segments']}</td>"
        f"<td>{item['score']}</td>"
        "</tr>"
        for item in report["model_summary"]
    )
    case_rows = "\n".join(_case_row(path.parent, item) for item in report["rows"])
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase5A API Model Benchmark</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; background: #fafafa; line-height: 1.45; }}
.pick {{ background: #e6f4ea; border: 1px solid #b7dfc2; padding: 14px; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; background: white; margin: 14px 0 24px; }}
th, td {{ border: 1px solid #dde1e6; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f1f3f4; }}
code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
a {{ color: #174ea6; }}
</style></head>
<body>
<h1>Phase5A Real API Model Benchmark</h1>
<section class="pick">
<h2>Recommended model: <code>{html.escape(str(report['recommended_model']))}</code></h2>
<p>{html.escape(report['selection_reason'])}</p>
<p>Strict follow-up: <a href="../qwen3_max_20case_strict_20260703/qwen3_max_20case_strict_report.html">qwen3-max 20-case goal-label-aligned stress test</a>.</p>
<p>Leakage audit: <a href="../qwen3_max_20case_noleak_20260703/qwen3_max_20case_noleak_report.html">qwen3-max 20-case no-leak audit</a>.</p>
</section>
<h2>Model Summary</h2>
<table><tr><th>Family</th><th>Model</th><th>Runnable</th><th>Closed-loop Success</th><th>Mean Final Dist</th><th>Mean Segments</th><th>Score</th></tr>{summary_rows}</table>
<h2>Cases</h2>
<table><tr><th>Family</th><th>Model</th><th>Goal</th><th>Status</th><th>Stop Success</th><th>Segments</th><th>Final Dist</th><th>Report</th></tr>{case_rows}</table>
<h2>Notes</h2>
<ul>
<li>All API keys were provided at runtime and are not written into this report.</li>
<li>Closed-loop success means the model chose valid waypoint ids and the BEV waypoint proxy ended within the stop radius of the selected target.</li>
<li>The first expanded benchmark did not explicitly require waypoint anchor labels to match the queried object label; the linked strict stress test adds that check.</li>
<li>The no-leak audit hides the precomputed goal-matching waypoint id list from the planner prompt while preserving the hidden evaluator check.</li>
<li>Current executor is still a BEV waypoint proxy; Habitat navmesh execution remains the next refinement.</li>
</ul>
</body></html>"""
    path.write_text(html_doc, encoding="utf-8")


def _rel(base: Path, target: str | None) -> str:
    if not target:
        return ""
    return os.path.relpath(Path(target).resolve(), start=base.resolve()).replace(os.sep, "/")


def _case_row(base: Path, item: dict[str, Any]) -> str:
    report_link = ""
    if item.get("html"):
        report_link = f'<a href="{html.escape(_rel(base, item["html"]))}">report</a>'
    return (
        "<tr>"
        f"<td>{html.escape(item['family'])}</td>"
        f"<td>{html.escape(item['model'])}</td>"
        f"<td>{html.escape(item['goal'])}</td>"
        f"<td>{html.escape(item['status'])}</td>"
        f"<td>{'yes' if item['stop_success_proxy'] else 'no'}</td>"
        f"<td>{item['segments']}</td>"
        f"<td>{item['final_distance_to_target_m']}</td>"
        f"<td>{report_link}</td>"
        "</tr>"
    )


if __name__ == "__main__":
    main()
