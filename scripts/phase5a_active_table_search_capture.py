from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from m25_habitat_rgbd_export import _save_frame  # noqa: E402
from phase23_habitat_control_server import HabitatControlSession, ensure_conda_nvidia_egl_vendor  # noqa: E402


SEARCH_LABELS = {"table", "counter", "sink"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture close multi-view tabletop searches after a from-zero exploration pass.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dataset-config")
    parser.add_argument("--exploration-metadata", required=True)
    parser.add_argument("--grounding-candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--step-m", type=float, default=0.30)
    parser.add_argument("--observation-radius-m", type=float, default=1.05)
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument("--viewpoints-per-target", type=int, default=2)
    parser.add_argument("--min-target-views", type=int, default=2)
    parser.add_argument("--min-target-confidence", type=float, default=0.24)
    parser.add_argument("--yaw-increment-deg", type=float, default=15.0)
    parser.add_argument("--scan-sector-deg", type=float, default=120.0)
    parser.add_argument("--look-down-deg", type=float, default=45.0)
    parser.add_argument("--max-frames", type=int, default=320)
    args = parser.parse_args()

    ensure_conda_nvidia_egl_vendor()
    exploration_metadata = _read_json(Path(args.exploration_metadata).expanduser().resolve())
    grounding = _read_json(Path(args.grounding_candidates).expanduser().resolve())
    targets = _select_targets(
        grounding.get("items", []),
        max_targets=int(args.max_targets),
        min_views=int(args.min_target_views),
        min_confidence=float(args.min_target_confidence),
    )
    if not targets:
        raise RuntimeError("No table/counter/sink target candidates were available for active search")

    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    start_index = len(exploration_metadata.get("frames", []))
    final_exploration_frame = exploration_metadata["frames"][-1]

    session = HabitatControlSession(
        scene=Path(args.scene).expanduser().resolve(),
        scene_dataset_config=Path(args.scene_dataset_config).expanduser().resolve() if args.scene_dataset_config else None,
        resolution=int(args.resolution),
        move_amount=float(args.step_m),
        turn_amount=float(args.yaw_increment_deg),
        semantic_categories=["wall", "door", "table", "chair", "bed", "sofa", "sink", "counter", "shower", "toilet"],
    )
    frames: list[dict[str, Any]] = []
    search_events: list[dict[str, Any]] = []
    try:
        current = _set_from_metadata(session, final_exploration_frame)
        targets = _greedy_target_order(targets, current)
        frame_index = start_index
        for target_index, target in enumerate(targets, start=1):
            if len(frames) >= int(args.max_frames):
                break
            target_xyz = np.asarray(target["position_3d"], dtype=np.float32)
            viewpoints = _candidate_viewpoints(
                session.sim.pathfinder,
                target_xyz=target_xyz,
                floor_y=float(current[1]),
                radius_m=float(args.observation_radius_m),
                count=int(args.viewpoints_per_target),
                start=current,
            )
            target_event = {
                "target_index": target_index,
                "target_id": target.get("id"),
                "target_label": target.get("label"),
                "target_position_3d": [float(value) for value in target_xyz],
                "source_views": target.get("source_view_ids", []),
                "viewpoints": [],
            }
            for view_index, viewpoint in enumerate(viewpoints, start=1):
                if len(frames) >= int(args.max_frames):
                    break
                route = _shortest_route(session.sim.pathfinder, current, viewpoint, step_m=float(args.step_m))
                route_position = np.asarray(route[0], dtype=np.float32)
                for point in route[1:]:
                    if len(frames) >= int(args.max_frames):
                        break
                    frame_index = _append_facing_frames(
                        session,
                        frames,
                        frames_dir,
                        frame_index,
                        route_position,
                        point,
                        target,
                        target_index,
                        view_index,
                        max_delta_deg=float(args.yaw_increment_deg),
                        action="target_route_face_turn",
                        search_stage="route_align",
                        force_frame=False,
                    )
                    direction = np.asarray(point, dtype=np.float32) - route_position
                    look_at = np.asarray(point, dtype=np.float32) + direction
                    session._set_agent_pose(point, look_at)
                    state = _light_state(frame_index)
                    record = _save_frame(
                        session,
                        frames_dir,
                        frame_index=frame_index,
                        action="target_route_step",
                        state=state,
                    )
                    record.update(_target_fields(target, target_index, view_index, "navigate"))
                    frames.append(record)
                    frame_index += 1
                    route_position = np.asarray(point, dtype=np.float32)
                current = np.asarray(viewpoint, dtype=np.float32)
                if len(frames) >= int(args.max_frames):
                    break

                frame_index = _append_facing_frames(
                    session,
                    frames,
                    frames_dir,
                    frame_index,
                    current,
                    target_xyz,
                    target,
                    target_index,
                    view_index,
                    max_delta_deg=float(args.yaw_increment_deg),
                    action="target_face_turn",
                    search_stage="face_anchor",
                    force_frame=True,
                )

                pitch_steps = max(1, int(round(float(args.look_down_deg) / max(1e-6, float(args.yaw_increment_deg)))))
                for _ in range(pitch_steps):
                    if len(frames) >= int(args.max_frames):
                        break
                    session.sim.step("look_down")
                    state = _light_state(frame_index)
                    record = _save_frame(
                        session,
                        frames_dir,
                        frame_index=frame_index,
                        action="target_pitch_down",
                        state=state,
                    )
                    record.update(_target_fields(target, target_index, view_index, "pitch_down"))
                    frames.append(record)
                    frame_index += 1

                half_sector_steps = max(
                    1,
                    int(round(0.5 * float(args.scan_sector_deg) / max(1e-6, float(args.yaw_increment_deg)))),
                )
                for turn_action, count in (
                    ("turn_right", half_sector_steps),
                    ("turn_left", 2 * half_sector_steps),
                    ("turn_right", half_sector_steps),
                ):
                    for _ in range(count):
                        if len(frames) >= int(args.max_frames):
                            break
                        session.sim.step(turn_action)
                        state = _light_state(frame_index)
                        record = _save_frame(
                            session,
                            frames_dir,
                            frame_index=frame_index,
                            action=f"target_scan_{turn_action}",
                            state=state,
                        )
                        record.update(_target_fields(target, target_index, view_index, "surface_scan"))
                        frames.append(record)
                        frame_index += 1

                for _ in range(pitch_steps):
                    if len(frames) >= int(args.max_frames):
                        break
                    session.sim.step("look_up")
                    state = _light_state(frame_index)
                    record = _save_frame(
                        session,
                        frames_dir,
                        frame_index=frame_index,
                        action="target_pitch_up",
                        state=state,
                    )
                    record.update(_target_fields(target, target_index, view_index, "pitch_restore"))
                    frames.append(record)
                    frame_index += 1
                target_event["viewpoints"].append([float(value) for value in viewpoint])
            search_events.append(target_event)
    finally:
        session.close()

    active_metadata = {
        "phase": "phase5a_active_table_search_capture",
        "scene": str(Path(args.scene).expanduser().resolve()),
        "scene_dataset_config": str(Path(args.scene_dataset_config).expanduser().resolve()) if args.scene_dataset_config else None,
        "resolution": int(args.resolution),
        "hfov_deg": float(exploration_metadata.get("hfov_deg", 90.0)),
        "num_frames": len(frames),
        "frames": frames,
        "search_events": search_events,
        "task": "自行熟悉房间并找到所有水杯",
    }
    active_path = out_dir / "active_search_frames_metadata.json"
    active_path.write_text(json.dumps(active_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    combined = dict(exploration_metadata)
    combined["phase"] = "phase5a_zero_map_then_active_cup_search"
    combined["task"] = active_metadata["task"]
    combined["frames"] = list(exploration_metadata.get("frames", [])) + frames
    combined["num_frames"] = len(combined["frames"])
    combined["active_search"] = {
        "source_grounding_candidates": str(Path(args.grounding_candidates).expanduser().resolve()),
        "num_targets": len(search_events),
        "events": search_events,
    }
    combined_path = out_dir / "combined_frames_metadata.json"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "active_search_frames": len(frames),
                "combined_frames": combined["num_frames"],
                "targets": len(search_events),
                "combined_metadata": str(combined_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _select_targets(
    items: list[dict[str, Any]],
    max_targets: int,
    min_views: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    selected = [
        item
        for item in items
        if str(item.get("label", "")).lower() in SEARCH_LABELS
        and isinstance(item.get("position_3d"), list)
        and len(item["position_3d"]) == 3
        and len(set(item.get("source_view_ids", []))) >= max(1, min_views)
        and float(item.get("confidence", 0.0)) >= min_confidence
    ]
    selected.sort(
        key=lambda item: (
            {"table": 3, "counter": 2, "sink": 1}.get(str(item.get("label", "")).lower(), 0),
            len(set(item.get("source_view_ids", []))),
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    quotas = {"table": 3, "counter": 2, "sink": 1}
    selected_per_label = {label: 0 for label in quotas}
    for candidate in selected:
        label = str(candidate.get("label", "")).lower()
        if selected_per_label.get(label, 0) >= quotas.get(label, max_targets):
            continue
        position = np.asarray(candidate["position_3d"], dtype=np.float32)
        if any(np.linalg.norm(position[[0, 2]] - np.asarray(item["position_3d"], dtype=np.float32)[[0, 2]]) < 0.7 for item in out):
            continue
        out.append(candidate)
        selected_per_label[label] = selected_per_label.get(label, 0) + 1
        if len(out) >= max(1, int(max_targets)):
            break
    return out


def _candidate_viewpoints(pathfinder, target_xyz: np.ndarray, floor_y: float, radius_m: float, count: int, start: np.ndarray) -> list[np.ndarray]:
    candidates: list[tuple[float, np.ndarray]] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
        requested = np.asarray(
            [
                float(target_xyz[0] + radius_m * math.cos(float(angle))),
                floor_y,
                float(target_xyz[2] + radius_m * math.sin(float(angle))),
            ],
            dtype=np.float32,
        )
        snapped = np.asarray(pathfinder.snap_point(requested), dtype=np.float32)
        if not np.isfinite(snapped).all():
            continue
        target_distance = float(np.linalg.norm(snapped[[0, 2]] - target_xyz[[0, 2]]))
        if not 0.60 <= target_distance <= 1.65:
            continue
        route = _find_path(pathfinder, start, snapped)
        if route is None:
            continue
        score = float(route["distance"]) + abs(target_distance - radius_m)
        candidates.append((score, snapped))
    candidates.sort(key=lambda item: item[0])
    selected: list[np.ndarray] = []
    for _, point in candidates:
        if any(np.linalg.norm(point[[0, 2]] - existing[[0, 2]]) < 0.8 for existing in selected):
            continue
        selected.append(point)
        if len(selected) >= max(1, int(count)):
            break
    return selected


def _greedy_target_order(targets: list[dict[str, Any]], start: np.ndarray) -> list[dict[str, Any]]:
    remaining = list(targets)
    ordered: list[dict[str, Any]] = []
    current = np.asarray(start, dtype=np.float32)
    while remaining:
        best_index = min(
            range(len(remaining)),
            key=lambda index: float(
                np.linalg.norm(
                    np.asarray(remaining[index]["position_3d"], dtype=np.float32)[[0, 2]]
                    - current[[0, 2]]
                )
            ),
        )
        selected = remaining.pop(best_index)
        ordered.append(selected)
        current = np.asarray(selected["position_3d"], dtype=np.float32)
    return ordered


def _shortest_route(pathfinder, start: np.ndarray, end: np.ndarray, step_m: float) -> list[np.ndarray]:
    result = _find_path(pathfinder, start, end)
    if result is None:
        return [np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)]
    return _resample_polyline(result["points"], step_m)


def _find_path(pathfinder, start: np.ndarray, end: np.ndarray) -> dict[str, Any] | None:
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(end, dtype=np.float32)
    if not pathfinder.find_path(path):
        return None
    return {
        "distance": float(path.geodesic_distance),
        "points": [np.asarray(point, dtype=np.float32) for point in path.points],
    }


def _resample_polyline(points: list[np.ndarray], step_m: float) -> list[np.ndarray]:
    if len(points) <= 1:
        return points
    cumulative = [0.0]
    for previous, current in zip(points[:-1], points[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(current - previous)))
    total = cumulative[-1]
    if total <= 1e-6:
        return [points[0]]
    count = max(2, int(math.ceil(total / max(0.05, step_m))) + 1)
    targets = np.linspace(0.0, total, count)
    result: list[np.ndarray] = []
    segment = 0
    for target in targets:
        while segment + 1 < len(cumulative) and cumulative[segment + 1] < target:
            segment += 1
        if segment + 1 >= len(points):
            result.append(points[-1].copy())
            continue
        span = cumulative[segment + 1] - cumulative[segment]
        alpha = 0.0 if span <= 1e-6 else (target - cumulative[segment]) / span
        result.append((1.0 - alpha) * points[segment] + alpha * points[segment + 1])
    return [np.asarray(point, dtype=np.float32) for point in result]


def _set_from_metadata(session: HabitatControlSession, frame: dict[str, Any]) -> np.ndarray:
    position = np.asarray(frame["agent_position_xyz"], dtype=np.float32)
    rotation = np.asarray(frame["agent_rotation_matrix"], dtype=np.float32)
    forward = -rotation[:, 2]
    look_at = position + forward
    session._set_agent_pose(position, look_at)
    return position


def _append_facing_frames(
    session: HabitatControlSession,
    frames: list[dict[str, Any]],
    frames_dir: Path,
    frame_index: int,
    current: np.ndarray,
    target_xyz: np.ndarray,
    target: dict[str, Any],
    target_index: int,
    view_index: int,
    max_delta_deg: float,
    action: str,
    search_stage: str,
    force_frame: bool,
) -> int:
    agent_state = session.sim.get_agent(0).get_state()
    rotation = _rotation_matrix(agent_state.rotation)
    forward = -rotation[:, 2]
    current_yaw = math.atan2(-float(forward[0]), -float(forward[2]))
    direction = np.asarray(target_xyz, dtype=np.float32) - np.asarray(current, dtype=np.float32)
    desired_yaw = math.atan2(-float(direction[0]), -float(direction[2]))
    delta = (desired_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi
    if not force_frame and abs(math.degrees(delta)) < 1.0:
        return frame_index
    steps = max(1, int(math.ceil(abs(math.degrees(delta)) / max(1.0, max_delta_deg))))
    for step in range(1, steps + 1):
        yaw = current_yaw + delta * (step / steps)
        look_direction = np.asarray([-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32)
        session._set_agent_pose(current, np.asarray(current, dtype=np.float32) + look_direction)
        state = _light_state(frame_index)
        record = _save_frame(
            session,
            frames_dir,
            frame_index=frame_index,
            action=action,
            state=state,
        )
        record.update(_target_fields(target, target_index, view_index, search_stage))
        frames.append(record)
        frame_index += 1
    return frame_index


def _target_fields(target: dict[str, Any], target_index: int, view_index: int, search_stage: str) -> dict[str, Any]:
    return {
        "search_target_index": int(target_index),
        "search_target_id": target.get("id"),
        "search_target_label": target.get("label"),
        "search_target_position_3d": target.get("position_3d"),
        "search_viewpoint_index": int(view_index),
        "search_stage": search_stage,
    }


def _light_state(frame_index: int) -> dict[str, Any]:
    return {"memory_step": int(frame_index)}


def _rotation_matrix(rotation) -> np.ndarray:
    if hasattr(rotation, "transform_vector"):
        axes = [
            np.asarray(rotation.transform_vector([1.0, 0.0, 0.0]), dtype=np.float32),
            np.asarray(rotation.transform_vector([0.0, 1.0, 0.0]), dtype=np.float32),
            np.asarray(rotation.transform_vector([0.0, 0.0, 1.0]), dtype=np.float32),
        ]
        return np.stack(axes, axis=1)
    import quaternion as np_quaternion

    return np.asarray(np_quaternion.as_rotation_matrix(rotation), dtype=np.float32)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
