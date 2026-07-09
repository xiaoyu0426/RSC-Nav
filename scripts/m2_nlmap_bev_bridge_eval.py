from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ModuleNotFoundError:
    plt = None
    ListedColormap = None

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlmap_bev_bridge import (
    bridge_mock_scene,
    export_rsc_memory,
    load_ascii_pcd,
    load_nlmap_objects,
    load_pointcloud_json,
    make_bridge_grid,
    rasterize_occupancy,
    rasterize_semantic,
    semantic_label_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="M2 NLMap-style semantic BEV bridge evaluator.")
    parser.add_argument("--objects-json", help="NLMap-style object inventory JSON. Uses deterministic mock if omitted.")
    parser.add_argument("--pointcloud-json", help="Optional point cloud JSON list or {'points': ...}.")
    parser.add_argument("--pointcloud-pcd", help="Optional sampled ASCII/binary PCD point cloud.")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "m2_nlmap_semantic_bev_bridge" / "mock_bridge"))
    parser.add_argument("--context-id", default="nlmap_mock_A")
    parser.add_argument("--scene-id", default="m2_nlmap_mock_scene")
    parser.add_argument("--resolution-m", type=float, default=0.25)
    parser.add_argument("--padding-m", type=float, default=0.75)
    parser.add_argument("--semantic-radius-m", type=float, default=0.35)
    parser.add_argument("--horizontal-axes", choices=["xz", "xy"], default="xz")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.objects_json:
        objects = load_nlmap_objects(args.objects_json, context_id=args.context_id, horizontal_axes=args.horizontal_axes)
        pointcloud = _load_pointcloud(args)
        input_mode = "external_objects"
    else:
        objects, pointcloud = bridge_mock_scene(context_id=args.context_id, horizontal_axes=args.horizontal_axes)
        input_mode = "deterministic_mock"
    if pointcloud is None:
        pointcloud = np.asarray([item.position_3d for item in objects], dtype=float)

    label_to_id = semantic_label_ids(objects)
    grid = make_bridge_grid(
        objects,
        pointcloud,
        resolution_m=args.resolution_m,
        padding_m=args.padding_m,
        horizontal_axes=args.horizontal_axes,
    )
    occupancy = rasterize_occupancy(pointcloud, grid)
    semantic, semantic_confidence = rasterize_semantic(
        objects,
        grid,
        label_to_id=label_to_id,
        radius_m=args.semantic_radius_m,
    )
    rsc_memory = export_rsc_memory(objects, label_to_id=label_to_id, scene_id=args.scene_id)

    objects_path = out_dir / "objects.json"
    rsc_memory_path = out_dir / "rsc_memory_init.json"
    occupancy_path = out_dir / "occupancy_bev.npz"
    semantic_path = out_dir / "semantic_bev.npz"
    metadata_path = out_dir / "bridge_metadata.json"

    objects_path.write_text(json.dumps([item.to_dict() for item in objects], indent=2), encoding="utf-8")
    rsc_memory_path.write_text(json.dumps(rsc_memory, indent=2), encoding="utf-8")
    np.savez_compressed(occupancy_path, occupancy=occupancy, grid=grid.to_dict(), point_count=len(pointcloud))
    np.savez_compressed(
        semantic_path,
        semantic=semantic,
        confidence=semantic_confidence,
        label_to_id=label_to_id,
        grid=grid.to_dict(),
    )

    occupancy_png = out_dir / "occupancy_bev.png"
    semantic_png = out_dir / "semantic_bev.png"
    _plot_occupancy(occupancy, objects, grid, occupancy_png)
    _plot_semantic(semantic, semantic_confidence, objects, grid, label_to_id, semantic_png)

    retrieval_dir = out_dir / "phase3_retrieval"
    retrieval_report = _run_phase3_retrieval(rsc_memory_path, retrieval_dir, args.context_id)
    metadata = {
        "phase": "m2_nlmap_semantic_bev_bridge",
        "status": "passed" if _passed(objects, occupancy, semantic, retrieval_report) else "failed",
        "input_mode": input_mode,
        "context_id": args.context_id,
        "scene_id": args.scene_id,
        "horizontal_axes": args.horizontal_axes,
        "num_objects": len(objects),
        "num_labels": len(label_to_id),
        "point_count": int(len(pointcloud)),
        "occupancy_cells": int((occupancy > 0).sum()),
        "semantic_cells": int((semantic > 0).sum()),
        "grid": grid.to_dict(),
        "label_to_id": label_to_id,
        "outputs": {
            "objects": str(objects_path),
            "rsc_memory_init": str(rsc_memory_path),
            "occupancy_bev_npz": str(occupancy_path),
            "semantic_bev_npz": str(semantic_path),
            "occupancy_bev_png": str(occupancy_png),
            "semantic_bev_png": str(semantic_png),
            "phase3_retrieval": str(retrieval_dir),
            "bridge_report": str(out_dir / "bridge_report.html"),
            "bridge_metadata": str(metadata_path),
        },
        "phase3_retrieval": retrieval_report,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_bridge_report(out_dir, metadata, objects, rsc_memory, retrieval_report)
    print(json.dumps(metadata, indent=2))


def _load_pointcloud(args) -> np.ndarray | None:
    if args.pointcloud_json:
        return load_pointcloud_json(args.pointcloud_json)
    if args.pointcloud_pcd:
        return load_ascii_pcd(args.pointcloud_pcd)
    return None


def _run_phase3_retrieval(objects_json: Path, out_dir: Path, context_id: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "phase3_landmark_retrieval_eval.py"),
        "--objects-json",
        str(objects_json),
        "--out-dir",
        str(out_dir),
        "--context-id",
        context_id,
        "--top-k",
        "5",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=True)
    metrics_path = out_dir / "metrics.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    report["command"] = cmd
    report["stdout_tail"] = result.stdout[-2000:]
    return report


def _passed(objects, occupancy, semantic, retrieval_report: dict[str, Any]) -> bool:
    return bool(
        len(objects) >= 5
        and (occupancy > 0).sum() > 0
        and (semantic > 0).sum() > 0
        and retrieval_report.get("status") == "passed"
    )


def _plot_occupancy(occupancy: np.ndarray, objects, grid, path: Path) -> None:
    if plt is None:
        _plot_occupancy_pillow(occupancy, objects, grid, path)
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(occupancy, origin="lower", cmap="Greys", vmin=0, vmax=1)
    _draw_objects(ax, objects, grid)
    ax.set_title("M2 Occupancy BEV")
    ax.set_xlabel(f"{grid.horizontal_axes[0]} cells")
    ax.set_ylabel(f"{grid.horizontal_axes[1]} cells")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_semantic(semantic: np.ndarray, confidence: np.ndarray, objects, grid, label_to_id: dict[str, int], path: Path) -> None:
    if plt is None or ListedColormap is None:
        _plot_semantic_pillow(semantic, objects, grid, label_to_id, path)
        return
    palette = ["#f2f2f2", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6", "#9d755d"]
    cmap = ListedColormap(palette[: max(label_to_id.values(), default=0) + 1])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(semantic, origin="lower", cmap=cmap, vmin=0, vmax=max(label_to_id.values(), default=1), alpha=0.86)
    ax.imshow(np.ma.masked_where(confidence <= 0, confidence), origin="lower", cmap="Greys", alpha=0.18, vmin=0, vmax=1)
    _draw_objects(ax, objects, grid)
    legend = ", ".join(f"{idx}:{label}" for label, idx in sorted(label_to_id.items(), key=lambda row: row[1]))
    ax.set_title(f"M2 Semantic BEV | {legend}")
    ax.set_xlabel(f"{grid.horizontal_axes[0]} cells")
    ax.set_ylabel(f"{grid.horizontal_axes[1]} cells")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_occupancy_pillow(occupancy: np.ndarray, objects, grid, path: Path) -> None:
    if Image is None or ImageDraw is None:
        path.with_suffix(".txt").write_text("matplotlib and pillow unavailable", encoding="utf-8")
        return
    scale = max(8, min(18, int(560 / max(grid.width, grid.height))))
    img = Image.new("RGB", (grid.width * scale, grid.height * scale), "#f5f5f5")
    draw = ImageDraw.Draw(img)
    for row in range(grid.height):
        for col in range(grid.width):
            color = "#2f3437" if occupancy[row, col] else "#f5f5f5"
            _draw_cell(draw, col, row, grid.height, scale, color)
    _draw_objects_pillow(draw, objects, grid, scale)
    img.save(path)


def _plot_semantic_pillow(semantic: np.ndarray, objects, grid, label_to_id: dict[str, int], path: Path) -> None:
    if Image is None or ImageDraw is None:
        path.with_suffix(".txt").write_text("matplotlib and pillow unavailable", encoding="utf-8")
        return
    palette = ["#f2f2f2", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6", "#9d755d"]
    scale = max(8, min(18, int(560 / max(grid.width, grid.height))))
    img = Image.new("RGB", (grid.width * scale, grid.height * scale), palette[0])
    draw = ImageDraw.Draw(img)
    for row in range(grid.height):
        for col in range(grid.width):
            label_id = int(semantic[row, col])
            color = palette[label_id % len(palette)] if label_id > 0 else palette[0]
            _draw_cell(draw, col, row, grid.height, scale, color)
    _draw_objects_pillow(draw, objects, grid, scale)
    legend = " | ".join(f"{idx}:{label}" for label, idx in sorted(label_to_id.items(), key=lambda row: row[1]))
    draw.text((8, 8), legend, fill="#111111")
    img.save(path)


def _draw_cell(draw, col: int, row: int, height: int, scale: int, color: str) -> None:
    y = (height - 1 - row) * scale
    x = col * scale
    draw.rectangle([x, y, x + scale - 1, y + scale - 1], fill=color)


def _draw_objects_pillow(draw, objects, grid, scale: int) -> None:
    from nlmap_bev_bridge import world_to_cell

    for item in objects:
        cell = world_to_cell(item.bev_position, grid)
        if cell is None:
            continue
        col, row = cell
        x = col * scale + scale // 2
        y = (grid.height - 1 - row) * scale + scale // 2
        radius = max(3, scale // 3)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline="#111111", width=2)
        draw.line([x - radius, y, x + radius, y], fill="#111111", width=1)
        draw.line([x, y - radius, x, y + radius], fill="#111111", width=1)
        draw.text((x + radius + 2, y - radius), item.label, fill="#111111")


def _draw_objects(ax, objects, grid) -> None:
    from nlmap_bev_bridge import world_to_cell

    for item in objects:
        cell = world_to_cell(item.bev_position, grid)
        if cell is None:
            continue
        col, row = cell
        ax.scatter([col], [row], s=55, facecolors="none", edgecolors="#111111", linewidths=1.4)
        ax.text(col + 0.3, row + 0.3, item.label, fontsize=8, color="#111111")


def _write_bridge_report(out_dir: Path, metadata: dict[str, Any], objects, rsc_memory: dict[str, Any], retrieval_report: dict[str, Any]) -> None:
    object_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item.id)}</td>"
        f"<td>{html.escape(item.label)}</td>"
        f"<td>{item.confidence:.2f}</td>"
        f"<td>{[round(v, 3) for v in item.position_3d]}</td>"
        f"<td>{[round(v, 3) for v in item.bev_position]}</td>"
        f"<td>{html.escape(', '.join(item.source_view_ids))}</td>"
        "</tr>"
        for item in objects
    )
    retrieval_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(check.get('query')))}</td>"
        f"<td>{'passed' if check.get('passed') else 'failed'}</td>"
        f"<td><pre>{html.escape(json.dumps(check.get('top', {}), indent=2))}</pre></td>"
        "</tr>"
        for check in retrieval_report.get("checks", [])
    )
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>M2 NLMap Semantic BEV Bridge</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
h1, h2 {{ margin-bottom: 8px; }}
code, pre {{ background: #f5f5f5; padding: 2px 4px; border-radius: 4px; }}
pre {{ overflow-x: auto; padding: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #f3f6f8; text-align: left; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 18px; }}
.panel img {{ width: 100%; border: 1px solid #ddd; }}
.status {{ font-weight: 700; color: {'#137333' if metadata['status'] == 'passed' else '#b3261e'}; }}
</style></head>
<body>
<h1>M2 NLMap-Style Semantic BEV Bridge</h1>
<p>Status: <span class="status">{html.escape(metadata['status'])}</span></p>
<p><code>NLMap-style semantic candidates / point cloud -> occupancy BEV -> semantic BEV -> RSC object memory -> Phase 3 landmark retrieval</code></p>
<h2>Summary</h2>
<pre>{html.escape(json.dumps({k: v for k, v in metadata.items() if k not in {'phase3_retrieval'}}, indent=2))}</pre>
<div class="grid">
  <section class="panel"><h2>Occupancy BEV</h2><img src="occupancy_bev.png"></section>
  <section class="panel"><h2>Semantic BEV</h2><img src="semantic_bev.png"></section>
</div>
<h2>Input Semantic Candidates</h2>
<table><tr><th>ID</th><th>Label</th><th>Conf</th><th>3D Position</th><th>BEV Position</th><th>Views</th></tr>{object_rows}</table>
<h2>RSC Object Memory</h2>
<pre>{html.escape(json.dumps(rsc_memory, indent=2))}</pre>
<h2>Phase 3 Retrieval Checks</h2>
<table><tr><th>Query</th><th>Status</th><th>Top Result</th></tr>{retrieval_rows}</table>
<p>Phase 3 full report: <a href="phase3_retrieval/retrieval_report.html">phase3_retrieval/retrieval_report.html</a></p>
</body></html>"""
    (out_dir / "bridge_report.html").write_text(html_doc, encoding="utf-8")


if __name__ == "__main__":
    main()
