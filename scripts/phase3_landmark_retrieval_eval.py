from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Keep the M1 report runnable in lightweight envs.
    plt = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landmark_retrieval import build_landmark_nodes, nodes_to_json, results_to_json, retrieve_landmarks


FIXTURE_OBJECTS = [
    {"id": "chair_active", "label": "chair", "bev_position": [0.0, 0.0], "confidence": 0.9, "freshness": 0.95, "status": "active", "context_id": "context_A", "source": "live", "last_seen_step": 10},
    {"id": "chair_stale", "label": "chair", "bev_position": [3.0, 0.0], "confidence": 0.8, "freshness": 0.4, "status": "stale", "context_id": "context_A", "source": "prior", "last_seen_step": 4},
    {"id": "table_active", "label": "table", "bev_position": [1.0, 1.5], "confidence": 0.7, "freshness": 0.85, "status": "active", "context_id": "context_A", "source": "live", "last_seen_step": 9},
    {"id": "bed_missing", "label": "bed", "bev_position": [-2.0, 1.0], "confidence": 0.4, "freshness": 0.2, "status": "missing", "context_id": "context_A", "source": "prior", "last_seen_step": 2},
    {"id": "door_active", "label": "door", "bev_position": [-1.5, -1.0], "confidence": 0.6, "freshness": 0.9, "status": "active", "context_id": "context_A", "source": "live", "last_seen_step": 12},
    {"id": "sofa_context_a", "label": "sofa", "bev_position": [2.0, 2.0], "confidence": 0.85, "freshness": 0.9, "status": "active", "context_id": "context_A", "source": "live", "last_seen_step": 11},
    {"id": "sofa_context_b", "label": "sofa", "bev_position": [8.0, 8.0], "confidence": 0.95, "freshness": 0.95, "status": "active", "context_id": "context_B", "source": "imported", "last_seen_step": 12},
]

FIXTURE_IMPORTED_OBJECTS = [
    {"id": "nlmap_cup_001", "label": "cup", "bev_position": [1.8, -1.7], "confidence": 0.78, "freshness": 1.0, "status": "active", "context_id": "context_A", "source": "imported", "last_seen_step": 13},
    {"id": "live_mug_alias_near", "label": "mug", "bev_position": [1.95, -1.65], "confidence": 0.64, "freshness": 0.9, "status": "active", "context_id": "context_A", "source": "live", "last_seen_step": 14},
]

QUERIES = [
    {"query": "chair", "context_id": "context_A", "expected_top_label": "chair", "expected_top_status": "active"},
    {"query": "seat", "context_id": "context_A", "expected_any_labels": ["chair", "sofa"]},
    {"query": "table", "context_id": "context_A", "expected_top_label": "table"},
    {"query": "door", "context_id": "context_A", "expected_top_label": "door"},
    {"query": "bed", "context_id": "context_A", "expected_top_label": "bed"},
    {"query": "sofa", "context_id": "context_A", "expected_top_context": "context_A"},
    {"query": "sofa", "context_id": "context_B", "expected_top_context": "context_B"},
    {"query": "mug", "context_id": "context_A", "expected_any_labels": ["cup"]},
    {"query": "microwave", "context_id": "context_A", "expected_empty": True},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 M1 landmark retrieval evaluator.")
    parser.add_argument("--objects-json", help="Optional object memory / imported objects JSON.")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "phase3_landmark_retrieval" / "m1_fixture"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-id", default="context_A")
    parser.add_argument("--merge-radius-m", type=float, default=0.75)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = _load_objects(Path(args.objects_json)) if args.objects_json else [*FIXTURE_OBJECTS, *FIXTURE_IMPORTED_OBJECTS]
    landmarks = build_landmark_nodes(objects, context_id=args.context_id, merge_radius_m=args.merge_radius_m)

    retrieval_report = []
    checks = []
    for query in QUERIES if not args.objects_json else _queries_from_objects(objects, args.context_id):
        results = retrieve_landmarks(query["query"], landmarks, context_id=query.get("context_id", args.context_id), top_k=args.top_k)
        result_rows = results_to_json(results)
        retrieval_report.append(
            {
                "query": query["query"],
                "context_id": query.get("context_id", args.context_id),
                "results": result_rows,
            }
        )
        if not args.objects_json:
            checks.append(_check_query(query, result_rows))
        else:
            checks.append(_check_query(query, result_rows))

    metrics = _metrics(checks)
    outputs = {
        "landmark_nodes": str(out_dir / "landmark_nodes.json"),
        "topk_retrieval": str(out_dir / "topk_retrieval.json"),
        "retrieval_score_breakdown": str(out_dir / "retrieval_score_breakdown.json"),
        "retrieval_bev": str(out_dir / ("retrieval_bev.png" if plt is not None else "retrieval_bev.svg")),
        "retrieval_report": str(out_dir / "retrieval_report.html"),
        "metrics": str(out_dir / "metrics.json"),
    }
    report = {
        "phase": "phase3_landmark_retrieval_m1",
        "status": "passed" if metrics["passed"] else "failed",
        "top_k": args.top_k,
        "context_id": args.context_id,
        "merge_radius_m": args.merge_radius_m,
        "num_objects": len(objects),
        "num_landmarks": len(landmarks),
        "metrics": metrics,
        "checks": checks,
        "outputs": outputs,
    }

    (out_dir / "landmark_nodes.json").write_text(json.dumps(nodes_to_json(landmarks), indent=2), encoding="utf-8")
    (out_dir / "topk_retrieval.json").write_text(json.dumps(retrieval_report, indent=2), encoding="utf-8")
    (out_dir / "retrieval_score_breakdown.json").write_text(json.dumps(retrieval_report, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_path = Path(outputs["retrieval_bev"])
    _plot_retrieval(landmarks, retrieval_report, plot_path)
    _write_html(out_dir, report, landmarks, retrieval_report, plot_path.name)
    print(json.dumps(report, indent=2))


def _load_objects(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "items" in data:
            return list(data["items"])
        if "objects" in data:
            return list(data["objects"])
    return list(data)


def _queries_from_objects(objects: list[dict], context_id: str) -> list[dict]:
    labels = sorted({str(obj.get("label") or obj.get("category") or "").lower() for obj in objects if obj.get("label") or obj.get("category")})
    return [{"query": label, "context_id": context_id, "expected_top_label": label} for label in labels[:8]]


def _check_query(query: dict, results: list[dict]) -> dict:
    top = results[0] if results else {}
    passed = True
    reasons = []
    if query.get("expected_empty"):
        passed = not results
        if results:
            reasons.append(f"expected empty results, got top label {top.get('label')}")
        return {
            "query": query["query"],
            "context_id": query.get("context_id"),
            "passed": passed,
            "reasons": reasons,
            "top": top,
        }
    expected_top_label = query.get("expected_top_label")
    if expected_top_label and top.get("label") != expected_top_label:
        passed = False
        reasons.append(f"expected top label {expected_top_label}, got {top.get('label')}")
    expected_top_status = query.get("expected_top_status")
    if expected_top_status and top.get("status") != expected_top_status:
        passed = False
        reasons.append(f"expected top status {expected_top_status}, got {top.get('status')}")
    expected_top_context = query.get("expected_top_context")
    if expected_top_context and top.get("context_id") != expected_top_context:
        passed = False
        reasons.append(f"expected top context {expected_top_context}, got {top.get('context_id')}")
    expected_any_labels = query.get("expected_any_labels", [])
    if expected_any_labels and not any(row.get("label") in expected_any_labels for row in results):
        passed = False
        reasons.append(f"expected any labels {expected_any_labels}")
    return {
        "query": query["query"],
        "context_id": query.get("context_id"),
        "passed": passed,
        "reasons": reasons,
        "top": top,
    }


def _metrics(checks: list[dict]) -> dict:
    if not checks:
        return {"passed": True, "num_checks": 0, "num_passed": 0, "pass_rate": 1.0}
    num_passed = sum(1 for check in checks if check["passed"])
    return {
        "passed": num_passed == len(checks),
        "num_checks": len(checks),
        "num_passed": num_passed,
        "pass_rate": num_passed / len(checks),
    }


def _plot_retrieval(landmarks, retrieval_report: list[dict], path: Path) -> None:
    if plt is None:
        _write_retrieval_svg(landmarks, retrieval_report, path)
        return
    status_colors = {"active": "#2ca02c", "stale": "#d8a700", "missing": "#8a8a8a", "relocated": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(7, 7))
    for node in landmarks:
        color = status_colors.get(node.status, "#9467bd")
        size = 50 + 180 * node.confidence
        ax.scatter(node.bev_position[0], node.bev_position[1], s=size, color=color, alpha=max(0.25, node.freshness), edgecolor="#222")
        ax.text(node.bev_position[0] + 0.05, node.bev_position[1] + 0.05, f"{node.label}\\n{node.context_id}", fontsize=7)
    if retrieval_report:
        first_results = retrieval_report[0]["results"][:3]
        for rank, row in enumerate(first_results, start=1):
            ax.scatter(row["bev_position"][0], row["bev_position"][1], s=260 - rank * 40, facecolors="none", edgecolors="#d62728", linewidths=2)
            ax.text(row["bev_position"][0], row["bev_position"][1] - 0.25, f"#{rank}", color="#d62728", fontsize=9, fontweight="bold")
    ax.set_title("Phase 3 M1 Landmark Retrieval")
    ax.set_xlabel("world x / BEV x")
    ax.set_ylabel("world z / BEV y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_retrieval_svg(landmarks, retrieval_report: list[dict], path: Path) -> None:
    status_colors = {"active": "#2ca02c", "stale": "#d8a700", "missing": "#8a8a8a", "relocated": "#ff7f0e"}
    xs = [node.bev_position[0] for node in landmarks] or [0.0]
    ys = [node.bev_position[1] for node in landmarks] or [0.0]
    min_x, max_x = min(xs) - 1.0, max(xs) + 1.0
    min_y, max_y = min(ys) - 1.0, max(ys) + 1.0
    width = height = 760

    def project(x: float, y: float) -> tuple[float, float]:
        px = 40 + (x - min_x) / max(1e-6, max_x - min_x) * (width - 80)
        py = height - (40 + (y - min_y) / max(1e-6, max_y - min_y) * (height - 80))
        return px, py

    top_ids = {}
    if retrieval_report:
        for rank, row in enumerate(retrieval_report[0]["results"][:3], start=1):
            top_ids[row["landmark_id"]] = rank
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        '<text x="24" y="30" font-family="Arial" font-size="20" font-weight="700">Phase 3 M1 Landmark Retrieval</text>',
    ]
    for i in range(0, 8):
        x = 40 + i * (width - 80) / 7
        y = 40 + i * (height - 80) / 7
        parts.append(f'<line x1="{x:.1f}" y1="40" x2="{x:.1f}" y2="{height-40}" stroke="#e3e0d8" stroke-width="1"/>')
        parts.append(f'<line x1="40" y1="{y:.1f}" x2="{width-40}" y2="{y:.1f}" stroke="#e3e0d8" stroke-width="1"/>')
    for node in landmarks:
        px, py = project(node.bev_position[0], node.bev_position[1])
        radius = 8 + 14 * float(node.confidence)
        color = status_colors.get(node.status, "#9467bd")
        opacity = max(0.3, float(node.freshness))
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="{opacity:.2f}" stroke="#222" stroke-width="1"/>')
        if node.id in top_ids:
            rank = top_ids[node.id]
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius+10:.1f}" fill="none" stroke="#d62728" stroke-width="3"/>')
            parts.append(f'<text x="{px-8:.1f}" y="{py+4:.1f}" font-family="Arial" font-size="13" fill="#d62728" font-weight="700">#{rank}</text>')
        label = f"{node.label} ({node.context_id})"
        parts.append(f'<text x="{px+10:.1f}" y="{py-10:.1f}" font-family="Arial" font-size="12" fill="#1f2933">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_html(out_dir: Path, report: dict, landmarks, retrieval_report: list[dict], plot_name: str) -> None:
    checks = "\n".join(
        f"<tr><td>{check['query']}</td><td>{check['passed']}</td><td><pre>{json.dumps(check.get('top'), ensure_ascii=False, indent=2)}</pre></td><td>{'; '.join(check.get('reasons', []))}</td></tr>"
        for check in report["checks"]
    )
    retrieval_rows = "\n".join(
        f"<tr><td>{item['query']}</td><td>{item['context_id']}</td><td><pre>{json.dumps(item['results'], ensure_ascii=False, indent=2)}</pre></td></tr>"
        for item in retrieval_report
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase 3 M1 Landmark Retrieval</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
td, th {{ border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }}
table {{ border-collapse: collapse; margin-bottom: 24px; }}
img {{ max-width: 760px; border: 1px solid #ccc; }}
pre {{ margin: 0; white-space: pre-wrap; }}
.pass {{ color: #137333; font-weight: 700; }}
.fail {{ color: #b3261e; font-weight: 700; }}
</style></head><body>
<h1>Phase 3 M1 Landmark Retrieval</h1>
<p>Status: <span class="{ 'pass' if report['status'] == 'passed' else 'fail' }">{report['status']}</span></p>
<img src="{plot_name}">
<h2>Metrics</h2>
<pre>{json.dumps(report['metrics'], ensure_ascii=False, indent=2)}</pre>
<h2>Checks</h2>
<table><tr><th>Query</th><th>Passed</th><th>Top</th><th>Reasons</th></tr>{checks}</table>
<h2>Retrieval Results</h2>
<table><tr><th>Query</th><th>Context</th><th>Top-K</th></tr>{retrieval_rows}</table>
<h2>Landmarks</h2>
<pre>{json.dumps(nodes_to_json(landmarks), ensure_ascii=False, indent=2)}</pre>
</body></html>"""
    (out_dir / "retrieval_report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
