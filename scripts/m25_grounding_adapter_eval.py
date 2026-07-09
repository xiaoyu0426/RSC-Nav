from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_grounding_adapter import (
    candidates_from_habitat_memory,
    compare_candidates,
    load_grounding_candidates,
    write_grounding_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="M2.5 grounding adapter and Habitat oracle validation.")
    parser.add_argument("--backend", choices=["habitat-oracle", "external-json", "model-stub"], default="habitat-oracle")
    parser.add_argument("--habitat-memory-json", help="Habitat semantic-oracle object memory JSON.")
    parser.add_argument("--predicted-objects-json", help="External model grounding candidates JSON.")
    parser.add_argument("--gold-memory-json", help="Gold Habitat object memory JSON. Defaults to --habitat-memory-json.")
    parser.add_argument("--metrics-json", help="Optional source episode metrics.json for report context.")
    parser.add_argument("--image-dir", help="Optional source episode image directory for report thumbnails.")
    parser.add_argument("--overlay-dir", help="Optional detector overlay image directory for visual grounding evidence.")
    parser.add_argument("--grounding-export-report", help="Optional detector/export report HTML to link from this validation report.")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "m25_open_vocab_grounding_adapter" / "habitat_oracle_validation"))
    parser.add_argument("--context-id", default="habitat_mp3d_A")
    parser.add_argument("--scene-id", default="habitat_mp3d_A")
    parser.add_argument("--horizontal-axes", choices=["xz", "xy"], default="xz")
    parser.add_argument("--resolution-m", type=float, default=0.10)
    parser.add_argument("--padding-m", type=float, default=1.0)
    parser.add_argument("--semantic-radius-m", type=float, default=0.25)
    parser.add_argument("--distance-threshold-m", type=float, default=0.75)
    parser.add_argument("--min-f1", type=float, default=0.95)
    parser.add_argument("--min-candidates", type=int, default=5)
    parser.add_argument("--preserve-candidate-context", action="store_true", help="Keep context_id from input candidates instead of assigning --context-id.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, backend_note = _load_candidates(args)
    if not args.preserve_candidate_context:
        for candidate in candidates:
            candidate.context_id = args.context_id
    candidates_path = out_dir / "grounding_candidates.json"
    write_grounding_candidates(
        candidates_path,
        candidates,
        metadata={
            "backend": args.backend,
            "context_id": args.context_id,
            "backend_note": backend_note,
        },
    )

    gold_candidates = _load_gold(args)
    validation = compare_candidates(candidates, gold_candidates, distance_threshold_m=args.distance_threshold_m)

    bridge_dir = out_dir / "bridge"
    bridge_metadata = _run_bridge(args, candidates_path, bridge_dir)
    source_metrics = _read_json(args.metrics_json) if args.metrics_json else {}

    passed = (
        len(candidates) >= int(args.min_candidates)
        and bridge_metadata.get("status") == "passed"
        and validation.get("f1", 0.0) >= float(args.min_f1)
        and args.backend != "model-stub"
    )
    metrics = {
        "phase": "m25_open_vocab_grounding_adapter",
        "status": "passed" if passed else "failed",
        "backend": args.backend,
        "backend_note": backend_note,
        "context_id": args.context_id,
        "num_candidates": len(candidates),
        "validation": validation,
        "bridge": bridge_metadata,
        "source_episode": _source_episode_summary(source_metrics),
        "outputs": {
            "grounding_candidates": str(candidates_path),
            "bridge_dir": str(bridge_dir),
            "bridge_report": str(bridge_dir / "bridge_report.html"),
            "report": str(out_dir / "m25_grounding_report.html"),
            "metrics": str(out_dir / "grounding_metrics.json"),
        },
    }
    (out_dir / "grounding_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_report(out_dir, metrics, candidates, args)
    print(json.dumps(metrics, indent=2))


def _load_candidates(args) -> tuple[list, str]:
    if args.backend == "habitat-oracle":
        if not args.habitat_memory_json:
            raise SystemExit("--habitat-memory-json is required for --backend habitat-oracle")
        return (
            candidates_from_habitat_memory(args.habitat_memory_json, context_id=args.context_id),
            "Uses Habitat semantic sensor/object memory as an oracle grounding backend.",
        )
    if args.backend == "external-json":
        if not args.predicted_objects_json:
            raise SystemExit("--predicted-objects-json is required for --backend external-json")
        return (
            load_grounding_candidates(args.predicted_objects_json, context_id=args.context_id),
            "Uses an externally generated open-vocabulary grounding JSON.",
        )
    return (
        [],
        "Model backend placeholder. Install/attach OWLv2 or GroundingDINO+SAM inference code, then run as external-json.",
    )


def _load_gold(args) -> list:
    gold_path = args.gold_memory_json or args.habitat_memory_json
    if not gold_path:
        return []
    return candidates_from_habitat_memory(gold_path, context_id=args.context_id)


def _run_bridge(args, candidates_path: Path, bridge_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "m2_nlmap_bev_bridge_eval.py"),
        "--objects-json",
        str(candidates_path),
        "--out-dir",
        str(bridge_dir),
        "--context-id",
        args.context_id,
        "--scene-id",
        args.scene_id,
        "--horizontal-axes",
        args.horizontal_axes,
        "--resolution-m",
        str(args.resolution_m),
        "--padding-m",
        str(args.padding_m),
        "--semantic-radius-m",
        str(args.semantic_radius_m),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True, text=True, capture_output=True)
    return _read_json(bridge_dir / "bridge_metadata.json")


def _source_episode_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    if not metrics:
        return {}
    return {
        "passed": metrics.get("passed"),
        "class_coverage": metrics.get("class_coverage", {}),
        "final_semantic": metrics.get("final_semantic", {}),
        "final_geometry_oracle": metrics.get("final_geometry_oracle", {}),
    }


def _write_report(out_dir: Path, metrics: dict[str, Any], candidates: list, args) -> None:
    validation = metrics["validation"]
    bridge = metrics["bridge"]
    source_images = _sample_images(Path(args.image_dir)) if args.image_dir else []
    overlay_images = _sample_images(Path(args.overlay_dir), pattern="*_overlay.jpg") if args.overlay_dir else []
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item.id)}</td>"
        f"<td>{html.escape(item.label)}</td>"
        f"<td>{item.confidence:.3f}</td>"
        f"<td>{item.position_3d[0]:.2f}, {item.position_3d[2]:.2f}</td>"
        f"<td>{html.escape(item.source)}</td>"
        "</tr>"
        for item in candidates[:80]
    )
    label_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{row['predicted']}</td>"
        f"<td>{row['gold']}</td>"
        f"<td>{row['matched']}</td>"
        f"<td>{row['precision']:.3f}</td>"
        f"<td>{row['recall']:.3f}</td>"
        "</tr>"
        for label, row in sorted(validation.get("by_label", {}).items())
    )
    image_tags = "\n".join(
        f'<img src="{html.escape(path)}" alt="{html.escape(Path(path).name)}">'
        for path in source_images
    )
    overlay_tags = "\n".join(
        f'<img src="{html.escape(path)}" alt="{html.escape(Path(path).name)}">'
        for path in overlay_images
    )
    export_report_link = ""
    if args.grounding_export_report:
        export_report_link = f'<p><a href="{html.escape(_relative_report_path(out_dir, Path(args.grounding_export_report)))}">Detector Export Report</a></p>'
    backend_explanation = (
        "This run consumes externally generated open-vocabulary grounding candidates. "
        "When the external JSON is produced by OWLv2/GroundingDINO, this page validates the downstream contract: "
        "detector candidates become object inventory, project into BEV/RSC memory, and feed Phase 3 retrieval."
        if metrics["backend"] == "external-json"
        else "This report validates the M2.5 adapter contract: RGB/RGB-D semantic grounding candidates can be represented as object inventory, projected through the M2 BEV bridge, exported as RSC object memory, and consumed by Phase 3 landmark retrieval. The current run uses Habitat semantic oracle as gold."
    )
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>M2.5 Grounding Adapter Validation</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
section {{ margin: 24px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #d7dce2; padding: 6px 8px; text-align: left; }}
th {{ background: #f4f6f8; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; align-items: start; }}
.metric {{ background: #f7f9fb; border: 1px solid #dfe5ec; border-radius: 6px; padding: 12px; }}
img {{ max-width: 100%; border: 1px solid #d7dce2; border-radius: 6px; }}
code {{ background: #f1f3f4; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>M2.5 Grounding Adapter Validation</h1>
<section class="grid">
  <div class="metric"><strong>Status</strong><br>{html.escape(metrics["status"])}</div>
  <div class="metric"><strong>Backend</strong><br>{html.escape(metrics["backend"])}</div>
  <div class="metric"><strong>Candidates</strong><br>{metrics["num_candidates"]}</div>
  <div class="metric"><strong>Validation F1</strong><br>{validation.get("f1", 0.0):.3f}</div>
  <div class="metric"><strong>Mean Centroid Error</strong><br>{_fmt(validation.get("mean_centroid_error_m"))}</div>
  <div class="metric"><strong>Bridge Status</strong><br>{html.escape(str(bridge.get("status")))}</div>
</section>
<section>
<h2>What This Validates</h2>
<p>{html.escape(backend_explanation)}</p>
</section>
<section class="grid">
  <div><h2>Occupancy BEV</h2><img src="bridge/occupancy_bev.png" alt="Occupancy BEV"></div>
  <div><h2>Semantic BEV</h2><img src="bridge/semantic_bev.png" alt="Semantic BEV"></div>
</section>
<section>
<h2>Per-Label Oracle Comparison</h2>
<table><tr><th>Label</th><th>Predicted</th><th>Gold</th><th>Matched</th><th>Precision</th><th>Recall</th></tr>{label_rows}</table>
</section>
<section>
<h2>Grounding Candidates</h2>
<table><tr><th>ID</th><th>Label</th><th>Confidence</th><th>BEV x,z</th><th>Source</th></tr>{rows}</table>
</section>
<section>
<h2>Detector Overlays</h2>
<div class="grid">{overlay_tags}</div>
</section>
<section>
<h2>Source RGB Samples</h2>
<div class="grid">{image_tags}</div>
</section>
<section>
<h2>Linked Outputs</h2>
{export_report_link}
<p><a href="bridge/bridge_report.html">M2 Bridge Report</a></p>
<p><a href="bridge/phase3_retrieval/retrieval_report.html">Phase 3 Retrieval Report</a></p>
</section>
</body>
</html>
"""
    (out_dir / "m25_grounding_report.html").write_text(report, encoding="utf-8")


def _sample_images(image_dir: Path, pattern: str = "*_rgb.jpg") -> list[str]:
    if not image_dir.is_absolute():
        image_dir = (ROOT / image_dir).resolve()
    if not image_dir.exists():
        return []
    images = sorted(image_dir.glob(pattern))
    if not images:
        return []
    picks = [images[0], images[len(images) // 2], images[-1]] if len(images) >= 3 else images
    outputs_root = (ROOT / "outputs").resolve()
    return [
        str(Path("..") / ".." / path.resolve().relative_to(outputs_root))
        if _is_under(path, outputs_root)
        else str(path)
        for path in picks
    ]


def _relative_report_path(out_dir: Path, report_path: Path) -> str:
    if not report_path.is_absolute():
        report_path = (ROOT / report_path).resolve()
    try:
        return str(report_path.resolve().relative_to(out_dir.resolve()))
    except ValueError:
        try:
            return str(Path("..") / report_path.resolve().relative_to(out_dir.parent.resolve()))
        except ValueError:
            return str(report_path)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):.3f} m"


if __name__ == "__main__":
    main()
