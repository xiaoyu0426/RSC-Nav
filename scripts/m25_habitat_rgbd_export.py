from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from phase23_habitat_control_server import (  # noqa: E402
    HabitatControlSession,
    _depth_image,
    _rgb_array,
    _valid_depth,
    ensure_conda_nvidia_egl_vendor,
)
from phase213_auto_episode_runner import (  # noqa: E402
    _build_coverage_loop_route,
    _fit_session_bev_to_route,
    _place_session_at_route_start,
    _route_step,
)
from semantic_bev_memory import semantic_array  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw Habitat RGB-D/pose frames for M2.5 grounding.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dataset-config")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--move-amount", type=float, default=0.25)
    parser.add_argument("--turn-amount", type=float, default=45.0)
    parser.add_argument("--bev-resolution", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--semantic-categories", default="wall,door,table,chair,bed,sofa")
    parser.add_argument("--trajectory-mode", choices=["actions", "coverage-loop"], default="actions")
    parser.add_argument("--actions", default=None, help="Comma-separated actions. Defaults to a compact move/yaw scan.")
    parser.add_argument("--max-frames", type=int, default=18)
    parser.add_argument("--coverage-samples", type=int, default=800)
    parser.add_argument("--coverage-waypoints", type=int, default=10)
    parser.add_argument("--coverage-route-steps", type=int, default=72)
    parser.add_argument("--coverage-route-passes", type=int, default=1)
    parser.add_argument("--coverage-map-margin-m", type=float, default=7.0)
    parser.add_argument("--yaw-scan-every", type=int, default=0)
    parser.add_argument("--yaw-scan-steps", type=int, default=0)
    parser.add_argument(
        "--lightweight-capture",
        action="store_true",
        help="Record RGB-D/pose without rendering online BEV or updating oracle memory every frame.",
    )
    parser.add_argument(
        "--pitch-scan-every",
        type=int,
        default=0,
        help="During coverage-loop traversal, add a downward tabletop scan every N route steps.",
    )
    parser.add_argument(
        "--pitch-scan-yaw-steps",
        type=int,
        default=0,
        help="Number of 45-degree-style yaw observations while looking down; defaults to a full turn.",
    )
    args = parser.parse_args()

    ensure_conda_nvidia_egl_vendor()
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    memory_path = out_dir / "habitat_oracle_object_memory.json"

    semantic_categories = [item.strip().lower() for item in args.semantic_categories.split(",") if item.strip()]
    session = HabitatControlSession(
        scene=Path(args.scene).expanduser().resolve(),
        resolution=args.resolution,
        move_amount=args.move_amount,
        turn_amount=args.turn_amount,
        scene_dataset_config=Path(args.scene_dataset_config).expanduser().resolve() if args.scene_dataset_config else None,
        bev_resolution=args.bev_resolution,
        grid_size=args.grid_size,
        sample_stride=args.sample_stride,
        semantic_categories=semantic_categories,
        memory_path=memory_path,
    )
    route_points = []
    route_plan = None
    if args.trajectory_mode == "coverage-loop":
        route_points, route_plan = _build_coverage_loop_route(
            session,
            sample_count=args.coverage_samples,
            waypoint_count=args.coverage_waypoints,
            route_steps=args.coverage_route_steps,
        )
        route_plan["bev_fit"] = _fit_session_bev_to_route(
            session=session,
            route_points=route_points,
            margin_m=args.coverage_map_margin_m,
        )

    frames = []
    try:
        if route_points:
            session.reset()
            _place_session_at_route_start(session, route_points)
        state = _capture_state(session, frame_index=0, lightweight=args.lightweight_capture)
        frames.append(_save_frame(session, frames_dir, frame_index=0, action="state", state=state))
        if route_points:
            actions = ["route_step"] * max(0, len(route_points) - 1) * max(1, int(args.coverage_route_passes))
            frame_index = 1
            route_step_index = 0
            yaw_scan_steps = int(args.yaw_scan_steps) if args.yaw_scan_steps > 0 else max(1, int(round(360.0 / max(1e-6, args.turn_amount))))
            pitch_scan_yaw_steps = (
                int(args.pitch_scan_yaw_steps)
                if args.pitch_scan_yaw_steps > 0
                else max(1, int(round(360.0 / max(1e-6, args.turn_amount))))
            )
            for _action in actions:
                if len(frames) >= int(args.max_frames):
                    break
                route_step_index += 1
                route_point_index = ((route_step_index - 1) % max(1, len(route_points) - 1)) + 1
                state = _capture_route_step(
                    session,
                    route_points,
                    route_point_index,
                    frame_index=frame_index,
                    lightweight=args.lightweight_capture,
                )
                frames.append(_save_frame(session, frames_dir, frame_index=frame_index, action="route_step", state=state))
                frame_index += 1
                if args.yaw_scan_every > 0 and route_step_index % args.yaw_scan_every == 0:
                    for _ in range(yaw_scan_steps):
                        if len(frames) >= int(args.max_frames):
                            break
                        state = _capture_action(session, "turn_left", frame_index, args.lightweight_capture)
                        frames.append(_save_frame(session, frames_dir, frame_index=frame_index, action="yaw_scan_turn_left", state=state))
                        frame_index += 1
                if args.pitch_scan_every > 0 and route_step_index % args.pitch_scan_every == 0:
                    if len(frames) < int(args.max_frames):
                        state = _capture_action(session, "look_down", frame_index, args.lightweight_capture)
                        frames.append(_save_frame(session, frames_dir, frame_index=frame_index, action="pitch_scan_look_down", state=state))
                        frame_index += 1
                    for _ in range(pitch_scan_yaw_steps):
                        if len(frames) >= int(args.max_frames):
                            break
                        state = _capture_action(session, "turn_left", frame_index, args.lightweight_capture)
                        frames.append(
                            _save_frame(
                                session,
                                frames_dir,
                                frame_index=frame_index,
                                action="pitch_scan_turn_left",
                                state=state,
                            )
                        )
                        frame_index += 1
                    if len(frames) < int(args.max_frames):
                        state = _capture_action(session, "look_up", frame_index, args.lightweight_capture)
                        frames.append(_save_frame(session, frames_dir, frame_index=frame_index, action="pitch_scan_look_up", state=state))
                        frame_index += 1
        else:
            actions = _actions(args.actions, max_frames=args.max_frames)
            for frame_index, action in enumerate(actions, start=1):
                state = _capture_action(session, action, frame_index, args.lightweight_capture)
                frames.append(_save_frame(session, frames_dir, frame_index=frame_index, action=action, state=state))
        if args.lightweight_capture:
            final_state = {"memory": {}}
        else:
            final_state = session.save_memory()
            session.save_bev_state(out_dir / "bev_state.npz")
    finally:
        session.close()

    metadata = {
        "phase": "m25_habitat_rgbd_export",
        "scene": str(Path(args.scene).expanduser().resolve()),
        "scene_dataset_config": str(Path(args.scene_dataset_config).expanduser().resolve()) if args.scene_dataset_config else None,
        "resolution": args.resolution,
        "hfov_deg": 90.0,
        "semantic_categories": semantic_categories,
        "trajectory_mode": args.trajectory_mode,
        "lightweight_capture": bool(args.lightweight_capture),
        "route_plan": route_plan,
        "num_frames": len(frames),
        "memory_path": str(memory_path),
        "bev_state": str(out_dir / "bev_state.npz"),
        "final_memory": final_state.get("memory", {}),
        "frames": frames,
    }
    (out_dir / "frames_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def _capture_state(session: HabitatControlSession, frame_index: int, lightweight: bool) -> dict[str, Any]:
    return {"memory_step": int(frame_index)} if lightweight else session.state()


def _capture_action(
    session: HabitatControlSession,
    action: str,
    frame_index: int,
    lightweight: bool,
) -> dict[str, Any]:
    if not lightweight:
        return session.action(action)
    session.sim.step(action)
    session.step_count += 1
    session.memory_step_count += 1
    session.last_payload = None
    return {"memory_step": int(frame_index)}


def _capture_route_step(
    session: HabitatControlSession,
    route_points: list[Any],
    route_point_index: int,
    frame_index: int,
    lightweight: bool,
) -> dict[str, Any]:
    if not lightweight:
        return _route_step(session, route_points, route_point_index)
    idx = min(max(1, route_point_index), len(route_points) - 1)
    look_at = route_points[idx + 1] if idx + 1 < len(route_points) else route_points[idx - 1]
    session._set_agent_pose(route_points[idx], look_at)
    session.step_count += 1
    session.memory_step_count += 1
    session.last_payload = None
    return {"memory_step": int(frame_index)}


def _save_frame(session: HabitatControlSession, frames_dir: Path, frame_index: int, action: str, state: dict[str, Any]) -> dict[str, Any]:
    observations = session.sim.get_sensor_observations()
    rgb = _rgb_array(observations.get("rgb"))
    depth = _valid_depth(observations.get("depth"))
    semantic = semantic_array(observations.get("semantic")) if "semantic" in observations else None
    agent_state = session.sim.get_agent(0).get_state()
    sensor_state = agent_state.sensor_states.get("depth") or next(iter(agent_state.sensor_states.values()))

    stem = f"frame_{frame_index:04d}"
    rgb_path = frames_dir / f"{stem}_rgb.jpg"
    depth_npy = frames_dir / f"{stem}_depth.npy"
    depth_png = frames_dir / f"{stem}_depth.png"
    semantic_npy = frames_dir / f"{stem}_semantic.npy"
    bev_png = frames_dir / f"{stem}_bev.png"
    semantic_bev_png = frames_dir / f"{stem}_semantic_bev.png"
    Image.fromarray(rgb).save(rgb_path, quality=95)
    np.save(depth_npy, depth.astype(np.float32))
    _depth_image(depth).save(depth_png)
    semantic_path = None
    if semantic is not None:
        np.save(semantic_npy, semantic.astype(np.int32))
        semantic_path = str(semantic_npy)
    bev_path = _save_base64_image(state.get("bev_png"), bev_png)
    semantic_bev_path = _save_base64_image(state.get("semantic_png"), semantic_bev_png)

    return {
        "frame_index": frame_index,
        "action": action,
        "rgb_path": str(rgb_path),
        "depth_npy": str(depth_npy),
        "depth_png": str(depth_png),
        "semantic_npy": semantic_path,
        "bev_png": bev_path,
        "semantic_bev_png": semantic_bev_path,
        "memory_step": int(state.get("memory_step", frame_index)),
        "pose": state.get("pose", {}),
        "sensor_position_xyz": _list3(sensor_state.position),
        "sensor_rotation_matrix": _rotation_matrix(sensor_state.rotation),
        "agent_position_xyz": _list3(agent_state.position),
        "agent_rotation_matrix": _rotation_matrix(agent_state.rotation),
        "semantic_report": state.get("semantic", {}),
        "memory_summary": state.get("memory", {}),
    }


def _actions(raw: str | None, max_frames: int) -> list[str]:
    if raw:
        actions = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        actions = []
        for _ in range(2):
            actions.extend(["turn_left", "turn_left", "move_forward", "turn_right", "move_forward", "turn_right"])
        actions.extend(["turn_left", "turn_left", "turn_left", "turn_left"])
    return actions[: max(0, int(max_frames) - 1)]


def _list3(value) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _rotation_matrix(rotation) -> list[list[float]]:
    if hasattr(rotation, "transform_vector"):
        axes = [
            np.asarray(rotation.transform_vector([1.0, 0.0, 0.0]), dtype=np.float32),
            np.asarray(rotation.transform_vector([0.0, 1.0, 0.0]), dtype=np.float32),
            np.asarray(rotation.transform_vector([0.0, 0.0, 1.0]), dtype=np.float32),
        ]
        matrix = np.stack(axes, axis=1)
    else:
        import quaternion as np_quaternion

        matrix = np.asarray(np_quaternion.as_rotation_matrix(rotation), dtype=np.float32)
    return [[float(value) for value in row] for row in matrix]


def _save_base64_image(encoded: Any, path: Path) -> str | None:
    if not isinstance(encoded, str) or not encoded:
        return None
    path.write_bytes(base64.b64decode(encoded))
    return str(path)


if __name__ == "__main__":
    main()
