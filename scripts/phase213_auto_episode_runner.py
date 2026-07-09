from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


DEFAULT_ACTIONS = (
    ["move_forward"] * 6
    + ["turn_left"] * 2
    + ["move_forward"] * 6
    + ["turn_right"] * 3
    + ["move_forward"] * 6
    + ["turn_right"] * 2
    + ["move_forward"] * 6
)
DEFAULT_PATH_STEPS = 36

IMAGE_KEYS = {
    "rgb_jpeg": ("rgb", ".jpg"),
    "depth_png": ("depth", ".png"),
    "bev_png": ("bev", ".png"),
    "oracle_png": ("oracle", ".png"),
    "oracle_diff_png": ("oracle_diff", ".png"),
    "semantic_png": ("semantic_bev", ".png"),
}
IMAGE_STATE_KEYS = set(IMAGE_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2.13 automatic Habitat episode runner with local recorder "
            "and offline summary report."
        )
    )
    parser.add_argument("--scene", required=True, help="Path to a Habitat-Sim loadable .glb scene.")
    parser.add_argument("--scene-dataset-config")
    parser.add_argument("--out-dir", help="Run output directory. Defaults to outputs/phase213_episode_runs/<timestamp>.")
    parser.add_argument("--episode-name", default="auto_episode")
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--move-amount", type=float, default=0.25)
    parser.add_argument("--turn-amount", type=float, default=15.0)
    parser.add_argument("--bev-resolution", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--obstacle-dilation-cells", type=int, default=2)
    parser.add_argument("--semantic-categories", default="wall,door,table,chair")
    parser.add_argument(
        "--allow-no-semantic",
        action="store_true",
        help="Treat semantic/object-memory metrics as optional for geometry-only scenes.",
    )
    parser.add_argument("--semantic-confidence-saturation", type=float, default=80.0)
    parser.add_argument("--freshness-tau-steps", type=float, default=20.0)
    parser.add_argument("--negative-evidence-scale", type=float, default=1.0)
    parser.add_argument("--object-memory-missing-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--object-memory-missing-missed-weight-threshold", type=float, default=6.0)
    parser.add_argument("--object-memory-stale-missed-weight-threshold", type=float, default=2.5)
    parser.add_argument("--memory-path", help="Saved object-memory JSON. Defaults inside --out-dir.")
    parser.add_argument("--load-bev-state", help="Load a previously saved BEV/semantic evidence state before recording.")
    parser.add_argument(
        "--load-bev-align",
        choices=("source", "center"),
        default="source",
        help=(
            "How to align loaded BEV state. 'source' preserves the saved world origin; "
            "'center' centers the saved grid on the current route-fitted grid."
        ),
    )
    parser.add_argument(
        "--drop-loaded-trajectory",
        action="store_true",
        help="Discard trajectory points from the loaded BEV state before recording the new episode.",
    )
    parser.add_argument(
        "--load-semantic-as-prior",
        action="store_true",
        help="Load semantic BEV evidence into a prior layer instead of the live semantic layer.",
    )
    parser.add_argument(
        "--semantic-prior-decay",
        action="store_true",
        help="Decay only the loaded semantic prior layer using current depth-confirmed observations.",
    )
    parser.add_argument("--semantic-prior-decay-scale", type=float, default=1.0)
    parser.add_argument("--load-object-memory", help="Load a prior object-memory JSON before recording.")
    parser.add_argument("--load-object-memory-source", default="prior_A", help="Source label for loaded prior object memory.")
    parser.add_argument(
        "--no-align-object-memory-to-loaded-bev",
        action="store_true",
        help="Do not transform loaded object centroids using the loaded BEV source/target origin transform.",
    )
    parser.add_argument(
        "--reset-loaded-object-evidence",
        action="store_true",
        help="Reset negative/not-observable counters on loaded prior object memory.",
    )
    parser.add_argument("--save-bev-state", help="Saved BEV state .npz. Defaults inside --out-dir.")
    parser.add_argument("--start-path-min-distance", type=float, default=3.0)
    parser.add_argument("--start-path-samples", type=int, default=48)
    parser.add_argument("--disable-oracle-metrics", action="store_true")
    parser.add_argument("--trajectory-mode", choices=("path", "actions", "coverage-loop"), default="path")
    parser.add_argument("--path-steps", type=int, default=DEFAULT_PATH_STEPS)
    parser.add_argument("--actions", help="Comma-separated action sequence.")
    parser.add_argument("--coverage-samples", type=int, default=800, help="Navmesh samples used to infer a coverage loop.")
    parser.add_argument("--coverage-waypoints", type=int, default=10, help="Perimeter waypoints in coverage-loop mode.")
    parser.add_argument("--coverage-route-steps", type=int, default=48, help="Recorded movement steps in coverage-loop mode.")
    parser.add_argument("--coverage-route-passes", type=int, default=1, help="Repeat the coverage-loop route this many times.")
    parser.add_argument("--coverage-map-margin-m", type=float, default=7.0, help="Extra BEV margin around coverage-loop route.")
    parser.add_argument("--look-sweep-every", type=int, default=0, help="In coverage-loop mode, run an up/down pitch sweep every N route steps.")
    parser.add_argument("--look-sweep-levels", type=int, default=1, help="Number of look_up/look_down actions per sweep direction.")
    parser.add_argument("--yaw-scan-every", type=int, default=0, help="In coverage-loop mode, rotate in place every N route steps.")
    parser.add_argument("--yaw-scan-steps", type=int, default=0, help="Turn actions per yaw scan. Defaults to 360 / turn amount.")
    parser.add_argument("--image-interval", type=int, default=1, help="Save images every N actions; default saves every step.")
    parser.add_argument("--summary-interval", type=int, default=5, help="Show every Nth image checkpoint in summary.html.")
    parser.add_argument("--save-full-state-jsonl", action="store_true", help="Also persist raw payloads with base64 images.")
    parser.add_argument("--target-classes", default="wall,door,table,chair")
    parser.add_argument("--min-oracle-free-iou", type=float, default=0.2)
    parser.add_argument("--min-oracle-occupied-f1", type=float, default=0.05)
    parser.add_argument("--max-mean-step-drift-m", type=float, default=0.45)
    parser.add_argument("--max-tail-drift-m", type=float, default=0.8)
    parser.add_argument("--stability-window", type=int, default=6)
    parser.add_argument("--min-final-items", type=int, default=4)
    parser.add_argument("--min-active-items", type=int, default=1)
    args = parser.parse_args()

    started_at = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = _run_out_dir(args.out_dir, args.episode_name, started_at)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    scene = Path(args.scene).expanduser().resolve()
    if not scene.exists():
        raise FileNotFoundError(scene)
    scene_dataset_config = Path(args.scene_dataset_config).expanduser().resolve() if args.scene_dataset_config else None
    if scene_dataset_config is not None and not scene_dataset_config.exists():
        raise FileNotFoundError(scene_dataset_config)

    actions = [] if args.trajectory_mode == "coverage-loop" else _parse_actions(args.actions, args.trajectory_mode, args.path_steps)
    semantic_categories = [item.strip().lower() for item in args.semantic_categories.split(",") if item.strip()]
    target_classes = [item.strip().lower() for item in args.target_classes.split(",") if item.strip()]
    memory_path = Path(args.memory_path).expanduser().resolve() if args.memory_path else out_dir / "live_object_memory.json"
    load_bev_state = Path(args.load_bev_state).expanduser().resolve() if args.load_bev_state else None
    load_object_memory = Path(args.load_object_memory).expanduser().resolve() if args.load_object_memory else None
    save_bev_state = Path(args.save_bev_state).expanduser().resolve() if args.save_bev_state else out_dir / "bev_state.npz"

    run_config = {
        "episode_name": args.episode_name,
        "started_at": started_at,
        "scene": str(scene),
        "scene_dataset_config": str(scene_dataset_config) if scene_dataset_config else None,
        "out_dir": str(out_dir),
        "memory_path": str(memory_path),
        "load_bev_state": str(load_bev_state) if load_bev_state else None,
        "load_bev_align": args.load_bev_align,
        "drop_loaded_trajectory": bool(args.drop_loaded_trajectory),
        "load_semantic_as_prior": bool(args.load_semantic_as_prior),
        "semantic_prior_decay": bool(args.semantic_prior_decay),
        "semantic_prior_decay_scale": args.semantic_prior_decay_scale,
        "save_bev_state": str(save_bev_state),
        "trajectory_mode": args.trajectory_mode,
        "num_actions": len(actions),
        "actions": actions,
        "resolution": args.resolution,
        "move_amount": args.move_amount,
        "turn_amount": args.turn_amount,
        "bev_resolution": args.bev_resolution,
        "grid_size": args.grid_size,
        "sample_stride": args.sample_stride,
        "obstacle_dilation_cells": args.obstacle_dilation_cells,
        "semantic_categories": semantic_categories,
        "semantic_required": not args.allow_no_semantic,
        "freshness_tau_steps": args.freshness_tau_steps,
        "negative_evidence_scale": args.negative_evidence_scale,
        "object_memory_missing_confidence_threshold": args.object_memory_missing_confidence_threshold,
        "object_memory_missing_missed_weight_threshold": args.object_memory_missing_missed_weight_threshold,
        "object_memory_stale_missed_weight_threshold": args.object_memory_stale_missed_weight_threshold,
        "oracle_metrics_enabled": not args.disable_oracle_metrics,
        "load_object_memory": str(load_object_memory) if load_object_memory else None,
        "load_object_memory_source": args.load_object_memory_source,
        "align_object_memory_to_loaded_bev": not args.no_align_object_memory_to_loaded_bev,
        "reset_loaded_object_evidence": bool(args.reset_loaded_object_evidence),
        "coverage_route_passes": args.coverage_route_passes,
        "look_sweep_every": args.look_sweep_every,
        "look_sweep_levels": args.look_sweep_levels,
        "yaw_scan_every": args.yaw_scan_every,
        "yaw_scan_steps": args.yaw_scan_steps,
    }
    from phase23_habitat_control_server import HabitatControlSession, ensure_conda_nvidia_egl_vendor

    ensure_conda_nvidia_egl_vendor()
    session = HabitatControlSession(
        scene=scene,
        resolution=args.resolution,
        move_amount=args.move_amount,
        turn_amount=args.turn_amount,
        scene_dataset_config=scene_dataset_config,
        bev_resolution=args.bev_resolution,
        grid_size=args.grid_size,
        sample_stride=args.sample_stride,
        obstacle_dilation_cells=args.obstacle_dilation_cells,
        semantic_categories=semantic_categories,
        semantic_confidence_saturation=args.semantic_confidence_saturation,
        freshness_tau_steps=args.freshness_tau_steps,
        negative_evidence_scale=args.negative_evidence_scale,
        object_memory_missing_confidence_threshold=args.object_memory_missing_confidence_threshold,
        object_memory_missing_missed_weight_threshold=args.object_memory_missing_missed_weight_threshold,
        object_memory_stale_missed_weight_threshold=args.object_memory_stale_missed_weight_threshold,
        semantic_prior_decay=args.semantic_prior_decay,
        semantic_prior_decay_scale=args.semantic_prior_decay_scale,
        memory_path=memory_path,
        start_path_min_distance=args.start_path_min_distance,
        start_path_samples=args.start_path_samples,
        enable_oracle_metrics=not args.disable_oracle_metrics,
    )

    states: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    route_points = []
    route_plan = None
    if args.trajectory_mode == "coverage-loop":
        route_points, route_plan = _build_coverage_loop_route(
            session=session,
            sample_count=args.coverage_samples,
            waypoint_count=args.coverage_waypoints,
            route_steps=args.coverage_route_steps,
        )
        map_fit = _fit_session_bev_to_route(
            session=session,
            route_points=route_points,
            margin_m=args.coverage_map_margin_m,
        )
        route_plan["bev_fit"] = map_fit
        route_passes = max(1, int(args.coverage_route_passes))
        actions = ["route_step"] * max(0, len(route_points) - 1) * route_passes
        run_config["num_actions"] = len(actions)
        run_config["route_plan"] = "route_plan.json"
        run_config["coverage_map_margin_m"] = args.coverage_map_margin_m
        run_config["bev_fit"] = map_fit
        run_config["coverage_route_passes"] = route_passes
        _write_json(out_dir / "route_plan.json", route_plan)
    _write_json(out_dir / "run_config.json", run_config)

    full_state_file = (out_dir / "states_full.jsonl").open("w", encoding="utf-8") if args.save_full_state_jsonl else None
    record_action_index = 0
    route_step_index = 0
    yaw_scan_steps = int(args.yaw_scan_steps) if args.yaw_scan_steps > 0 else max(1, int(round(360.0 / max(1e-6, args.turn_amount))))
    try:
        state = session.reset()
        if route_points:
            _place_session_at_route_start(session, route_points)
        loaded_bev_state = None
        if load_bev_state is not None:
            loaded_bev_state = session.load_bev_state(
                load_bev_state,
                align=args.load_bev_align,
                keep_trajectory=not args.drop_loaded_trajectory,
                load_semantic_as_prior=args.load_semantic_as_prior,
            )
            run_config["loaded_bev_state"] = loaded_bev_state
            _write_json(out_dir / "run_config.json", run_config)
        loaded_object_memory = None
        if load_object_memory is not None:
            loaded_object_memory = session.load_object_memory(
                load_object_memory,
                source=args.load_object_memory_source,
                align_to_loaded_bev=not args.no_align_object_memory_to_loaded_bev,
                reset_evidence=args.reset_loaded_object_evidence,
            )
            run_config["loaded_object_memory"] = loaded_object_memory
            _write_json(out_dir / "run_config.json", run_config)
        if route_points or loaded_bev_state is not None:
            state = session.state()
        _record_state(
            state=state,
            states=states,
            checkpoints=checkpoints,
            images_dir=images_dir,
            reason="initial",
            action=None,
            action_index=0,
            image_interval=max(1, args.image_interval),
            summary_interval=max(1, args.summary_interval),
            full_state_file=full_state_file,
        )

        for action in actions:
            record_action_index += 1
            action_started = time.time()
            try:
                if route_points:
                    route_step_index += 1
                    route_point_index = ((route_step_index - 1) % max(1, len(route_points) - 1)) + 1
                    state = _route_step(session, route_points, route_point_index)
                else:
                    state = session.action(action)
                state["action_elapsed_sec"] = round(time.time() - action_started, 4)
                _record_state(
                    state=state,
                    states=states,
                    checkpoints=checkpoints,
                    images_dir=images_dir,
                    reason=f"action_{record_action_index}",
                    action=action,
                    action_index=record_action_index,
                    image_interval=max(1, args.image_interval),
                    summary_interval=max(1, args.summary_interval),
                    full_state_file=full_state_file,
                )
                if (
                    route_points
                    and args.look_sweep_every > 0
                    and route_step_index % args.look_sweep_every == 0
                ):
                    for sweep_action in _look_sweep_actions(args.look_sweep_levels):
                        record_action_index += 1
                        sweep_started = time.time()
                        state = session.action(sweep_action)
                        state["action_elapsed_sec"] = round(time.time() - sweep_started, 4)
                        _record_state(
                            state=state,
                            states=states,
                            checkpoints=checkpoints,
                            images_dir=images_dir,
                            reason=f"pitch_sweep_{record_action_index}",
                            action=sweep_action,
                            action_index=record_action_index,
                            image_interval=max(1, args.image_interval),
                            summary_interval=max(1, args.summary_interval),
                            full_state_file=full_state_file,
                        )
                if (
                    route_points
                    and args.yaw_scan_every > 0
                    and route_step_index % args.yaw_scan_every == 0
                ):
                    for scan_step in range(yaw_scan_steps):
                        record_action_index += 1
                        scan_started = time.time()
                        state = session.action("turn_left")
                        state["action_elapsed_sec"] = round(time.time() - scan_started, 4)
                        _record_state(
                            state=state,
                            states=states,
                            checkpoints=checkpoints,
                            images_dir=images_dir,
                            reason=f"yaw_scan_{record_action_index}",
                            action="turn_left",
                            action_index=record_action_index,
                            image_interval=max(1, args.image_interval),
                            summary_interval=max(1, args.summary_interval),
                            full_state_file=full_state_file,
                        )
            except Exception as exc:
                errors.append({"action_index": record_action_index, "action": action, "error": str(exc)})
                break

        try:
            final_state = session.save_memory()
            session.save_bev_state(save_bev_state)
            if states:
                states[-1]["memory_saved_path"] = final_state.get("memory_saved_path")
                states[-1]["bev_state_saved_path"] = str(save_bev_state)
            checkpoints.append(
                {
                    "step": int(final_state.get("step", states[-1].get("step", 0) if states else 0)),
                    "memory_step": int(final_state.get("memory_step", states[-1].get("memory_step", 0) if states else 0)),
                    "reason": "final_save",
                    "action": "save_memory",
                    "action_index": record_action_index + 1,
                    "image_files": {},
                }
            )
        except Exception as exc:
            errors.append({"action_index": record_action_index + 1, "action": "save_memory", "error": str(exc)})
    finally:
        if full_state_file is not None:
            full_state_file.close()
        session.close()

    object_history = _object_history(states)
    metrics = _compute_metrics(
        states=states,
        object_history=object_history,
        target_classes=target_classes,
        min_oracle_free_iou=args.min_oracle_free_iou,
        min_oracle_occupied_f1=args.min_oracle_occupied_f1,
        max_mean_step_drift_m=args.max_mean_step_drift_m,
        max_tail_drift_m=args.max_tail_drift_m,
        stability_window=args.stability_window,
        min_final_items=args.min_final_items,
        min_active_items=args.min_active_items,
        semantic_required=not args.allow_no_semantic,
    )
    summary = {
        "episode_name": args.episode_name,
        "out_dir": str(out_dir),
        "memory_path": str(memory_path),
        "bev_state_path": str(save_bev_state),
        "num_requested_actions": len(actions),
        "num_executed_actions": record_action_index,
        "num_recorded_states": len(states),
        "errors": errors,
        "checkpoints": checkpoints,
        "metrics": metrics,
    }

    _write_json(out_dir / "timeline_compact.json", states)
    _write_json(out_dir / "object_history.json", object_history)
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "summary_report.json", summary)
    _write_summary_html(out_dir, summary)
    print(json.dumps(summary, indent=2))


def _run_out_dir(out_dir: str | None, episode_name: str, started_at: str) -> Path:
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in episode_name).strip("_")
    if not safe_name:
        safe_name = "auto_episode"
    return (ROOT / "outputs" / "phase213_episode_runs" / f"{started_at}_{safe_name}").resolve()


def _parse_actions(actions: str | None, trajectory_mode: str, path_steps: int) -> list[str]:
    if actions is None:
        if trajectory_mode == "path":
            return ["path_step"] * max(1, int(path_steps))
        return list(DEFAULT_ACTIONS)
    return [item.strip() for item in actions.split(",") if item.strip()]


def _look_sweep_actions(levels: int) -> list[str]:
    levels = max(1, int(levels))
    return (
        ["look_up"] * levels
        + ["look_down"] * levels
        + ["look_down"] * levels
        + ["look_up"] * levels
    )


def _build_coverage_loop_route(
    session,
    sample_count: int,
    waypoint_count: int,
    route_steps: int,
) -> tuple[list[Any], dict[str, Any]]:
    import habitat_sim
    import numpy as np

    pathfinder = getattr(session.sim, "pathfinder", None)
    if pathfinder is None or not getattr(pathfinder, "is_loaded", False):
        raise RuntimeError("coverage-loop requires a loaded Habitat navmesh/pathfinder")

    samples = []
    for _ in range(max(8, int(sample_count))):
        point = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if point.shape == (3,) and np.isfinite(point).all():
            samples.append(point)
    if len(samples) < 3:
        raise RuntimeError("coverage-loop could not sample enough navigable points")

    sample_array = np.asarray(samples, dtype=np.float32)
    center = np.mean(sample_array, axis=0)
    waypoint_indices = _perimeter_waypoint_indices(sample_array, center, max(3, int(waypoint_count)))
    waypoints = [sample_array[idx] for idx in waypoint_indices]
    waypoints = _order_waypoints_by_angle(waypoints, center)
    waypoints.append(waypoints[0].copy())

    route_polyline = []
    segment_summaries = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        shortest_path = habitat_sim.ShortestPath()
        shortest_path.requested_start = start
        shortest_path.requested_end = end
        if not pathfinder.find_path(shortest_path):
            continue
        points = [np.asarray(point, dtype=np.float32) for point in shortest_path.points]
        if len(points) < 2:
            continue
        if route_polyline:
            points = points[1:]
        route_polyline.extend(points)
        segment_summaries.append(
            {
                "start": _point_list(start),
                "end": _point_list(end),
                "geodesic_distance_m": float(shortest_path.geodesic_distance),
                "num_points": len(points),
            }
        )

    if len(route_polyline) < 2:
        raise RuntimeError("coverage-loop could not connect sampled waypoints with shortest paths")

    route_points = _resample_polyline(route_polyline, max(2, int(route_steps) + 1))
    plan = {
        "strategy": "coverage-loop",
        "sample_count": len(samples),
        "requested_waypoints": int(waypoint_count),
        "selected_waypoints": [_point_list(point) for point in waypoints[:-1]],
        "center": _point_list(center),
        "segments": segment_summaries,
        "route_steps": max(0, len(route_points) - 1),
        "route_points": [_point_list(point) for point in route_points],
    }
    return route_points, plan


def _fit_session_bev_to_route(session, route_points: list[Any], margin_m: float) -> dict[str, Any]:
    import math
    import numpy as np

    if not route_points:
        return {}
    points = np.asarray(route_points, dtype=np.float32)
    min_x = float(points[:, 0].min())
    max_x = float(points[:, 0].max())
    min_z = float(points[:, 2].min())
    max_z = float(points[:, 2].max())
    margin = max(0.0, float(margin_m))
    resolution = float(session.bev_resolution)
    span_x = max_x - min_x
    span_z = max_z - min_z
    required_world_size = max(span_x, span_z) + 2.0 * margin
    required_cells = int(math.ceil(required_world_size / resolution)) + 1
    grid_size = max(int(session.grid_size), required_cells)
    world_size = grid_size * resolution
    center_x = 0.5 * (min_x + max_x)
    center_z = 0.5 * (min_z + max_z)
    origin = (center_x - 0.5 * world_size, center_z - 0.5 * world_size)
    session.grid_size = grid_size
    session.memory_origin_world_xz = origin
    return {
        "route_bounds_xz": {
            "min_x": min_x,
            "max_x": max_x,
            "min_z": min_z,
            "max_z": max_z,
            "span_x": span_x,
            "span_z": span_z,
        },
        "margin_m": margin,
        "resolution": resolution,
        "grid_size": [grid_size, grid_size],
        "world_size_m": world_size,
        "origin_world_xz": [float(origin[0]), float(origin[1])],
    }


def _perimeter_waypoint_indices(sample_array, center, waypoint_count: int) -> list[int]:
    import numpy as np

    rel = sample_array[:, [0, 2]] - center[[0, 2]]
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    distances = np.linalg.norm(rel, axis=1)
    selected: list[int] = []
    bins = np.linspace(-np.pi, np.pi, waypoint_count + 1)
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (angles >= low) & (angles < high)
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            continue
        selected.append(int(candidates[np.argmax(distances[candidates])]))
    if len(selected) >= 3:
        return selected
    return [int(idx) for idx in np.argsort(distances)[-max(3, waypoint_count) :]]


def _order_waypoints_by_angle(points: list[Any], center) -> list[Any]:
    import math

    return sorted(points, key=lambda point: math.atan2(float(point[2] - center[2]), float(point[0] - center[0])))


def _resample_polyline(points: list[Any], max_samples: int) -> list[Any]:
    import numpy as np

    if not points:
        return []
    if len(points) == 1 or max_samples <= 1:
        return [np.asarray(points[0], dtype=np.float32)]

    cumulative = [0.0]
    for prev, cur in zip(points[:-1], points[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(np.asarray(cur) - np.asarray(prev))))
    total = cumulative[-1]
    if total <= 0.0:
        return [np.asarray(points[0], dtype=np.float32)]

    targets = np.linspace(0.0, total, max_samples)
    out = []
    segment = 0
    for target in targets:
        while segment + 1 < len(cumulative) and cumulative[segment + 1] < target:
            segment += 1
        if segment + 1 >= len(points):
            out.append(np.asarray(points[-1], dtype=np.float32))
            continue
        span = cumulative[segment + 1] - cumulative[segment]
        alpha = 0.0 if span <= 0.0 else (target - cumulative[segment]) / span
        out.append((1.0 - alpha) * np.asarray(points[segment]) + alpha * np.asarray(points[segment + 1]))
    return [np.asarray(point, dtype=np.float32) for point in out]


def _place_session_at_route_start(session, route_points: list[Any]) -> None:
    if not route_points:
        return
    look_at = route_points[1] if len(route_points) > 1 else None
    session._set_agent_pose(route_points[0], look_at)
    session.step_count = 0
    session.memory_step_count = 0
    session._reset_memory()
    session.last_payload = None


def _route_step(session, route_points: list[Any], action_index: int) -> dict[str, Any]:
    idx = min(max(1, action_index), len(route_points) - 1)
    look_at = route_points[idx + 1] if idx + 1 < len(route_points) else route_points[idx - 1]
    with session.lock:
        session._set_agent_pose(route_points[idx], look_at)
        session.step_count += 1
        session.memory_step_count += 1
        session.last_payload = session._state_payload()
        return session.last_payload


def _point_list(point) -> list[float]:
    return [float(point[0]), float(point[1]), float(point[2])]


def _record_state(
    state: dict[str, Any],
    states: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    images_dir: Path,
    reason: str,
    action: str | None,
    action_index: int,
    image_interval: int,
    summary_interval: int,
    full_state_file,
    force_image: bool = False,
    force_summary: bool = False,
) -> None:
    compact = _compact_state(state)
    compact["action"] = action
    compact["action_index"] = action_index
    states.append(compact)

    if full_state_file is not None:
        full_state_file.write(json.dumps(state, separators=(",", ":")) + "\n")
        full_state_file.flush()

    save_summary = force_summary or action_index == 0 or action_index % summary_interval == 0
    save_image = force_image or save_summary or action_index % image_interval == 0
    image_files: dict[str, str] = {}
    if save_image:
        suffix = "_final" if reason == "final_save" else ""
        image_files = _save_checkpoint_images(state, images_dir, prefix=f"step_{int(state['step']):04d}{suffix}")

    if save_summary:
        checkpoints.append(
            {
                "step": int(state["step"]),
                "memory_step": int(state.get("memory_step", state["step"])),
                "reason": reason,
                "action": action,
                "action_index": action_index,
                "image_files": image_files,
            }
        )


def _compact_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key not in IMAGE_STATE_KEYS}


def _save_checkpoint_images(state: dict[str, Any], images_dir: Path, prefix: str) -> dict[str, str]:
    saved: dict[str, str] = {}
    for key, (label, suffix) in IMAGE_KEYS.items():
        value = state.get(key)
        if not value:
            continue
        path = images_dir / f"{prefix}_{label}{suffix}"
        path.write_bytes(base64.b64decode(value))
        saved[label] = str(Path("images") / path.name)
    return saved


def _object_history(states: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    history: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        step = int(state.get("step", 0))
        for item in state.get("memory_items", []):
            item_id = str(item["id"])
            entry = {
                "step": step,
                "category": item.get("category"),
                "semantic_id": item.get("semantic_id"),
                "centroid_xz": item.get("centroid_xz"),
                "confidence": item.get("confidence"),
                "freshness": item.get("freshness"),
                "status": item.get("status"),
                "last_seen_step": item.get("last_seen_step"),
                "missed_observation_count": item.get("missed_observation_count", 0),
                "negative_evidence_count": item.get("negative_evidence_count", 0),
            }
            history.setdefault(item_id, []).append(entry)
    return history


def _compute_metrics(
    states: list[dict[str, Any]],
    object_history: dict[str, list[dict[str, Any]]],
    target_classes: list[str],
    min_oracle_free_iou: float,
    min_oracle_occupied_f1: float,
    max_mean_step_drift_m: float,
    max_tail_drift_m: float,
    stability_window: int,
    min_final_items: int,
    min_active_items: int,
    semantic_required: bool = True,
) -> dict[str, Any]:
    if not states:
        return {"passed": False, "error": "no states recorded"}
    final = states[-1]
    final_memory = final.get("memory", {})
    final_semantic = final.get("semantic", {})
    final_oracle = final.get("geometry_oracle", {})
    final_per_class = final_memory.get("per_class", {})
    class_coverage = {category: int(final_per_class.get(category, 0)) for category in target_classes}

    stability_rows = []
    step_drifts = []
    total_drifts = []
    tail_drifts = []
    confidence_values = []
    freshness_values = []
    max_missed = 0
    max_negative = 0
    for item_id, observations in sorted(object_history.items()):
        centroids = [
            tuple(float(value) for value in obs["centroid_xz"])
            for obs in observations
            if obs.get("centroid_xz") and len(obs["centroid_xz"]) >= 2
        ]
        confidence_values.extend(float(obs["confidence"]) for obs in observations if obs.get("confidence") is not None)
        freshness_values.extend(float(obs["freshness"]) for obs in observations if obs.get("freshness") is not None)
        item_max_missed = max((int(obs.get("missed_observation_count", 0)) for obs in observations), default=0)
        item_max_negative = max((int(obs.get("negative_evidence_count", 0)) for obs in observations), default=0)
        max_missed = max(max_missed, item_max_missed)
        max_negative = max(max_negative, item_max_negative)
        if len(centroids) < 2:
            continue
        item_step_drifts = [_distance(prev, cur) for prev, cur in zip(centroids[:-1], centroids[1:])]
        total_drift = max(_distance(centroids[0], centroid) for centroid in centroids[1:])
        tail_centroids = centroids[-max(2, int(stability_window)) :]
        tail_drift = max(_distance(tail_centroids[0], centroid) for centroid in tail_centroids[1:])
        step_drifts.extend(item_step_drifts)
        total_drifts.append(total_drift)
        tail_drifts.append(tail_drift)
        stability_rows.append(
            {
                "id": item_id,
                "category": observations[-1].get("category"),
                "semantic_id": observations[-1].get("semantic_id"),
                "num_observations": len(observations),
                "mean_step_drift_m": _mean(item_step_drifts),
                "max_step_drift_m": max(item_step_drifts) if item_step_drifts else 0.0,
                "total_drift_m": total_drift,
                "tail_drift_m": tail_drift,
                "final_status": observations[-1].get("status"),
                "max_missed_observation_count": item_max_missed,
                "max_negative_evidence_count": item_max_negative,
            }
        )

    mean_step_drift = _mean(step_drifts)
    max_total_drift = max(total_drifts) if total_drifts else 0.0
    max_tail_drift = max(tail_drifts) if tail_drifts else 0.0
    final_items = int(final_memory.get("num_items", 0))
    active_items = int(final_memory.get("active_items", 0))
    covered_all_classes = (not semantic_required) or all(count > 0 for count in class_coverage.values())
    bev_nonempty = int(final.get("bev", {}).get("num_explored_cells", 0)) > 0 and int(final.get("bev", {}).get("num_occupied_cells", 0)) > 0
    oracle_enabled = bool(final_oracle.get("enabled"))
    oracle_free_iou = float(final_oracle.get("free_iou_observed", 0.0))
    oracle_occupied_f1 = float(final_oracle.get("occupied_f1_observed", 0.0))
    geometry_ok = (
        bev_nonempty
        and (not oracle_enabled or oracle_free_iou >= min_oracle_free_iou)
        and (not oracle_enabled or oracle_occupied_f1 >= min_oracle_occupied_f1)
    )
    semantic_ok = (not semantic_required) or (
        int(final_semantic.get("observed_target_instances", 0)) > 0
        and int(final_semantic.get("semantic_cells", 0)) > 0
    )
    stability_ok = (not semantic_required) or (
        bool(stability_rows)
        and mean_step_drift <= max_mean_step_drift_m
        and max_tail_drift <= max_tail_drift_m
    )
    memory_ok = (not semantic_required) or (final_items >= min_final_items and active_items >= min_active_items)
    passed = geometry_ok and semantic_ok and covered_all_classes and memory_ok and stability_ok

    observability = _observability_metrics(states)
    return {
        "passed": passed,
        "criteria": {
            "geometry_ok": geometry_ok,
            "bev_nonempty": bev_nonempty,
            "oracle_enabled": oracle_enabled,
            "min_oracle_free_iou": min_oracle_free_iou,
            "min_oracle_occupied_f1": min_oracle_occupied_f1,
            "semantic_ok": semantic_ok,
            "semantic_required": semantic_required,
            "covered_all_classes": covered_all_classes,
            "memory_ok": memory_ok,
            "stability_ok": stability_ok,
            "max_mean_step_drift_m": max_mean_step_drift_m,
            "max_tail_drift_m": max_tail_drift_m,
            "stability_window": int(stability_window),
            "min_final_items": min_final_items,
            "min_active_items": min_active_items,
        },
        "final_step": int(final.get("step", 0)),
        "class_coverage": class_coverage,
        "final_memory": final_memory,
        "final_semantic": final_semantic,
        "final_geometry_oracle": final_oracle,
        "final_bev": final.get("bev", {}),
        "observability": observability,
        "object_stability": {
            "tracked_items": len(stability_rows),
            "mean_step_drift_m": mean_step_drift,
            "max_step_drift_m": max(step_drifts) if step_drifts else 0.0,
            "max_total_drift_m": max_total_drift,
            "max_tail_drift_m": max_tail_drift,
            "mean_confidence": _mean(confidence_values),
            "mean_freshness": _mean(freshness_values),
            "max_missed_observation_count": max_missed,
            "max_negative_evidence_count": max_negative,
            "items": stability_rows,
        },
    }


def _observability_metrics(states: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    total_positive = 0
    total_expected_visible = 0
    total_misses = 0
    total_not_observable = 0
    for state in states:
        report = state.get("observability") or {}
        positive = len(report.get("positive_ids", []))
        expected_visible = len(report.get("expected_visible_ids", []))
        misses = len(report.get("expected_visible_miss_ids", []))
        not_observable = len(report.get("not_observable_ids", []))
        total_positive += positive
        total_expected_visible += expected_visible
        total_misses += misses
        total_not_observable += not_observable
        rows.append(
            {
                "step": int(state.get("step", 0)),
                "positive": positive,
                "expected_visible": expected_visible,
                "expected_visible_miss": misses,
                "not_observable": not_observable,
            }
        )
    return {
        "total_positive_observations": total_positive,
        "total_expected_visible": total_expected_visible,
        "total_expected_visible_misses": total_misses,
        "total_not_observable": total_not_observable,
        "timeline": rows,
    }


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summary_html(out_dir: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    rows = []
    for checkpoint in summary["checkpoints"]:
        image_files = checkpoint.get("image_files", {})
        rows.append(
            "<tr>"
            f"<td>{checkpoint['step']}</td>"
            f"<td>{checkpoint['reason']}</td>"
            f"<td>{checkpoint.get('action') or ''}</td>"
            f"<td>{_img_tag(image_files.get('rgb'))}</td>"
            f"<td>{_img_tag(image_files.get('bev'))}</td>"
            f"<td>{_img_tag(image_files.get('semantic_bev'))}</td>"
            f"<td>{_img_tag(image_files.get('oracle_diff'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Phase 2.13 Auto Episode Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;line-height:1.42;color:#182028}}
code,pre{{background:#f5f6f7;padding:2px 4px;border-radius:4px}}
pre{{padding:12px;overflow:auto}}
img{{width:220px;border:1px solid #ccd1d5;background:#f9fafb}}
td,th{{vertical-align:top;padding:8px;border-bottom:1px solid #dde2e6}}
table{{border-collapse:collapse;width:100%}}
.ok{{color:#16784f}}.bad{{color:#b13b3b}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:16px 0}}
.metric{{border:1px solid #dde2e6;border-radius:6px;padding:10px}}
.metric span{{display:block;color:#64717c;font-size:12px;margin-bottom:4px}}
.metric strong{{font-size:18px}}
</style></head>
<body>
<h1>Phase 2.13 Auto Episode Report</h1>
<p>Episode: <code>{summary['episode_name']}</code></p>
<p>Status: <strong class="{'ok' if metrics.get('passed') else 'bad'}">{metrics.get('passed')}</strong></p>
<div class="grid">
<div class="metric"><span>Final step</span><strong>{metrics.get('final_step', 0)}</strong></div>
<div class="metric"><span>Memory items</span><strong>{metrics.get('final_memory', {}).get('num_items', 0)}</strong></div>
<div class="metric"><span>Active items</span><strong>{metrics.get('final_memory', {}).get('active_items', 0)}</strong></div>
<div class="metric"><span>Expected-visible misses</span><strong>{metrics.get('observability', {}).get('total_expected_visible_misses', 0)}</strong></div>
<div class="metric"><span>Not observable</span><strong>{metrics.get('observability', {}).get('total_not_observable', 0)}</strong></div>
<div class="metric"><span>Mean drift</span><strong>{metrics.get('object_stability', {}).get('mean_step_drift_m', 0):.3f} m</strong></div>
</div>
<h2>Artifacts</h2>
<ul>
<li><code>run_config.json</code></li>
<li><code>timeline_compact.json</code></li>
<li><code>object_history.json</code></li>
<li><code>metrics.json</code></li>
<li><code>summary_report.json</code></li>
<li><code>live_object_memory.json</code></li>
</ul>
<h2>Metrics</h2>
<pre>{json.dumps(metrics, indent=2)}</pre>
<h2>Visual Checkpoints</h2>
<table>
<tr><th>Step</th><th>Reason</th><th>Action</th><th>RGB</th><th>BEV</th><th>Semantic BEV</th><th>Oracle Diff</th></tr>
{''.join(rows)}
</table>
</body></html>
"""
    (out_dir / "summary.html").write_text(html, encoding="utf-8")


def _img_tag(path: str | None) -> str:
    return f'<img src="{path}">' if path else ""


if __name__ == "__main__":
    main()
