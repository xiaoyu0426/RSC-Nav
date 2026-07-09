from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phase23_habitat_control_server import HabitatControlSession, ensure_conda_nvidia_egl_vendor  # noqa: E402


DEFAULT_CASES = [
    "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_bed_20260703",
    "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_chair_20260703",
    "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_door_20260703",
    "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_sofa_20260703",
    "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_table_20260703",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase5A selected waypoints against Habitat navmesh shortest paths.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dataset-config")
    parser.add_argument("--out-dir", default="outputs/phase5a_navmesh_validation/qwen3_max_find5_20260704")
    parser.add_argument("--case-dir", action="append", dest="case_dirs", help="Phase5A run dir. Defaults to 5 no-leak find cases.")
    parser.add_argument("--start-xz", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--snap-max-distance-m", type=float, default=2.5)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()

    ensure_conda_nvidia_egl_vendor()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = [Path(item).expanduser().resolve() for item in (args.case_dirs or DEFAULT_CASES)]

    session = HabitatControlSession(
        scene=Path(args.scene).expanduser().resolve(),
        scene_dataset_config=Path(args.scene_dataset_config).expanduser().resolve() if args.scene_dataset_config else None,
        resolution=args.resolution,
        move_amount=0.25,
        turn_amount=15.0,
        semantic_categories=["wall", "door", "table", "chair", "bed", "sofa"],
    )
    try:
        rows = [_validate_case(session, case_dir, tuple(args.start_xz), float(args.snap_max_distance_m)) for case_dir in case_dirs]
    finally:
        session.close()

    summary = {
        "phase": "phase5a_navmesh_validation",
        "scene": str(Path(args.scene).expanduser().resolve()),
        "case_count": len(rows),
        "reachable_count": sum(1 for row in rows if row["reachable"]),
        "snap_ok_count": sum(1 for row in rows if row["snap_ok"]),
        "all_passed": all(row["reachable"] and row["snap_ok"] for row in rows),
        "snap_max_distance_m": float(args.snap_max_distance_m),
        "cases": rows,
    }
    (out_dir / "navmesh_validation_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(out_dir / "navmesh_validation_report.html", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _validate_case(session: HabitatControlSession, case_dir: Path, start_xz: tuple[float, float], snap_max_distance_m: float) -> dict[str, Any]:
    import habitat_sim

    request = _read_json(case_dir / "planner_request.json")
    output = _read_json(case_dir / "planner_output.json")
    metrics = _read_json(case_dir / "metrics.json")
    waypoint_by_id = {str(item.get("id")): item for item in request.get("candidate_waypoints", [])}
    selected_id = str(output.get("selected_waypoint_id", ""))
    selected_wp = waypoint_by_id.get(selected_id)
    pathfinder = session.sim.pathfinder
    if selected_wp is None:
        return {
            "case": case_dir.name,
            "goal_query": request.get("goal_query"),
            "selected_waypoint_id": selected_id,
            "reachable": False,
            "snap_ok": False,
            "failure_reason": "selected waypoint id not found in planner_request candidate_waypoints",
            "phase5a_metrics": metrics,
        }

    start = _snap_xz(pathfinder, start_xz)
    target_xz = _xy(selected_wp.get("bev_position"))
    target = _snap_xz(pathfinder, target_xz)
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start["snapped"]
    shortest_path.requested_end = target["snapped"]
    reachable = bool(pathfinder.find_path(shortest_path))
    geodesic = float(shortest_path.geodesic_distance) if reachable and math.isfinite(float(shortest_path.geodesic_distance)) else None
    snap_ok = bool(start["snap_distance_m"] <= snap_max_distance_m and target["snap_distance_m"] <= snap_max_distance_m)
    return {
        "case": case_dir.name,
        "goal_query": request.get("goal_query"),
        "goal_target_label": request.get("goal_target_label"),
        "selected_waypoint_id": selected_id,
        "anchor_label": selected_wp.get("anchor_label"),
        "anchor_landmark_id": selected_wp.get("anchor_landmark_id"),
        "requested_start_xz": [float(start_xz[0]), float(start_xz[1])],
        "requested_target_xz": [float(target_xz[0]), float(target_xz[1])],
        "snapped_start_xyz": _point_list(start["snapped"]),
        "snapped_target_xyz": _point_list(target["snapped"]),
        "start_snap_distance_m": round(start["snap_distance_m"], 4),
        "target_snap_distance_m": round(target["snap_distance_m"], 4),
        "snap_ok": snap_ok,
        "reachable": reachable,
        "geodesic_distance_m": round(geodesic, 4) if geodesic is not None else None,
        "num_path_points": len(shortest_path.points) if reachable else 0,
        "path_points_xyz": [_point_list(point) for point in shortest_path.points] if reachable else [],
        "phase5a_goal_label_aligned": bool(metrics.get("goal_label_aligned")),
        "phase5a_stop_success_proxy": bool(metrics.get("stop_success_proxy")),
        "failure_reason": None if reachable and snap_ok else _failure_reason(reachable, snap_ok, start, target, snap_max_distance_m),
    }


def _snap_xz(pathfinder, xz: tuple[float, float]) -> dict[str, Any]:
    ref_y = float(np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)[1])
    requested = np.asarray([float(xz[0]), ref_y, float(xz[1])], dtype=np.float32)
    snapped = np.asarray(pathfinder.snap_point(requested), dtype=np.float32)
    return {
        "requested": requested,
        "snapped": snapped,
        "snap_distance_m": float(np.linalg.norm(snapped - requested)),
    }


def _failure_reason(reachable: bool, snap_ok: bool, start: dict[str, Any], target: dict[str, Any], snap_max_distance_m: float) -> str:
    reasons = []
    if not reachable:
        reasons.append("Habitat pathfinder could not find a shortest path")
    if not snap_ok:
        reasons.append(
            "snap distance exceeds threshold "
            f"(start={start['snap_distance_m']:.3f}, target={target['snap_distance_m']:.3f}, max={snap_max_distance_m:.3f})"
        )
    return "; ".join(reasons)


def _write_html(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for item in summary["cases"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['goal_query'] or '')}</td>"
            f"<td>{html.escape(item.get('anchor_label') or '')}</td>"
            f"<td>{html.escape(item['selected_waypoint_id'])}</td>"
            f"<td>{_yes(item['snap_ok'])}</td>"
            f"<td>{_yes(item['reachable'])}</td>"
            f"<td>{item.get('target_snap_distance_m')}</td>"
            f"<td>{item.get('geodesic_distance_m')}</td>"
            f"<td>{html.escape(item.get('failure_reason') or '')}</td>"
            "</tr>"
        )
    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Phase5A Navmesh Validation</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202124;line-height:1.5}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #d8e0ea;padding:8px;text-align:left;vertical-align:top}}th{{background:#eef3f8}}
.ok{{color:#147a3d;font-weight:700}}.bad{{color:#b42318;font-weight:700}}code{{background:#eef2f7;padding:2px 4px;border-radius:4px}}
</style></head><body>
<h1>Phase5A Navmesh Validation</h1>
<p>验证 qwen3-max selected waypoint 是否能回到 Habitat navmesh 中 snap 到可导航点，并查询 shortest path。</p>
<p><b>Result:</b> {summary['reachable_count']}/{summary['case_count']} reachable, {summary['snap_ok_count']}/{summary['case_count']} snap-ok.</p>
<table><tr><th>Goal</th><th>Anchor</th><th>Selected Waypoint</th><th>Snap OK</th><th>Reachable</th><th>Target Snap m</th><th>Geodesic m</th><th>Failure</th></tr>{''.join(rows)}</table>
</body></html>"""
    path.write_text(html_doc, encoding="utf-8")


def _yes(value: bool) -> str:
    return '<span class="ok">yes</span>' if value else '<span class="bad">no</span>'


def _xy(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return 0.0, 0.0


def _point_list(point: Any) -> list[float]:
    arr = np.asarray(point, dtype=np.float32)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
