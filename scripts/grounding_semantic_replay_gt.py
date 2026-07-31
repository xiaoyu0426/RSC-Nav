from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved Habitat camera poses with the semantic sensor enabled "
            "to build detector-only audit ground truth."
        )
    )
    parser.add_argument("--frames-metadata", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--scene")
    parser.add_argument("--scene-dataset-config")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--min-instance-area-px", type=int, default=24)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument(
        "--verify-rgb-every",
        type=int,
        default=25,
        help="Compare replayed RGB with the saved JPEG every N selected frames.",
    )
    parser.add_argument("--max-rgb-mae", type=float, default=3.0)
    parser.add_argument("--max-rgb-p95-error", type=float, default=12.0)
    parser.add_argument("--save-semantic-png-dir")
    args = parser.parse_args()

    _progress("reading_metadata")
    metadata_path = Path(args.frames_metadata).expanduser().resolve()
    metadata = _read_json(metadata_path)
    scene = Path(args.scene or metadata["scene"]).expanduser().resolve()
    dataset_config_value = (
        args.scene_dataset_config or metadata.get("scene_dataset_config")
    )
    dataset_config = (
        Path(dataset_config_value).expanduser().resolve()
        if dataset_config_value
        else None
    )
    resolution = int(metadata["resolution"])
    selected_frames = _select_frames(
        metadata.get("frames", []),
        start=max(0, int(args.frame_start)),
        end=args.frame_end,
        stride=max(1, int(args.frame_stride)),
        max_frames=args.max_frames,
    )
    if not selected_frames:
        raise ValueError("No frames selected for semantic replay")

    png_dir = (
        Path(args.save_semantic_png_dir).expanduser().resolve()
        if args.save_semantic_png_dir
        else None
    )
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)

    _progress("creating_habitat_session")
    session = _create_session(
        scene=scene,
        scene_dataset_config=dataset_config,
        resolution=resolution,
    )
    _progress("habitat_session_ready")
    frames: list[dict[str, Any]] = []
    rgb_checks: list[dict[str, Any]] = []
    observed_unknown_ids: set[int] = set()
    try:
        object_index = _semantic_object_index(session.sim.semantic_scene)
        _progress(
            "semantic_index_ready",
            object_count=len(object_index),
        )
        for selected_index, frame in enumerate(selected_frames):
            if selected_index == 0 or selected_index % 100 == 0:
                _progress(
                    "replaying_frame",
                    frame_index=int(frame["frame_index"]),
                    selected_index=selected_index,
                    selected_total=len(selected_frames),
                )
            _set_replay_pose(session.sim.get_agent(0), frame)
            observations = session.sim.get_sensor_observations()
            semantic = _semantic_array(observations.get("semantic"))
            saved_depth = _saved_depth(
                frame.get("depth_npy"),
                expected_shape=semantic.shape,
            )
            visible_ids = {
                int(value)
                for value in np.unique(semantic)
                if int(value) >= 0
            }
            observed_unknown_ids.update(visible_ids.difference(object_index))
            instances = _visible_instances(
                semantic,
                object_index,
                min_area_px=max(1, int(args.min_instance_area_px)),
                depth=saved_depth,
                sensor_position_xyz=frame.get("sensor_position_xyz"),
                sensor_rotation_matrix=frame.get("sensor_rotation_matrix"),
                min_depth_m=float(args.min_depth_m),
                max_depth_m=float(args.max_depth_m),
            )
            frame_record: dict[str, Any] = {
                "frame_index": int(frame["frame_index"]),
                "instances": instances,
                "num_instances": len(instances),
                "num_target_instances": sum(
                    item.get("canonical_label") in {"door", "window"}
                    for item in instances
                ),
            }
            if png_dir is not None:
                png_path = (
                    png_dir
                    / f"frame_{int(frame['frame_index']):04d}_semantic.png"
                )
                _semantic_preview(semantic).save(png_path)
                frame_record["semantic_png"] = str(png_path)
            frames.append(frame_record)

            verify_every = max(0, int(args.verify_rgb_every))
            should_verify = (
                verify_every > 0
                and (
                    selected_index == 0
                    or selected_index == len(selected_frames) - 1
                    or selected_index % verify_every == 0
                )
            )
            if should_verify:
                rgb_checks.append(
                    _rgb_replay_check(
                        observations.get("rgb"),
                        frame.get("rgb_path"),
                        frame_index=int(frame["frame_index"]),
                    )
                )
    finally:
        session.close()

    rgb_integrity = _rgb_integrity_report(
        rgb_checks,
        enabled=max(0, int(args.verify_rgb_every)) > 0,
        max_mae=float(args.max_rgb_mae),
        max_p95_abs_error=float(args.max_rgb_p95_error),
    )
    output = {
        "schema_version": 1,
        "source": {
            "frames_metadata": str(metadata_path),
            "frames_metadata_sha256": _sha256(metadata_path),
            "scene": str(scene),
            "scene_dataset_config": (
                str(dataset_config) if dataset_config is not None else None
            ),
            "resolution": resolution,
            "source_num_frames": int(metadata.get("num_frames", 0)),
        },
        "selection": {
            "frame_start": int(args.frame_start),
            "frame_end": (
                int(args.frame_end) if args.frame_end is not None else None
            ),
            "frame_stride": max(1, int(args.frame_stride)),
            "max_frames": (
                int(args.max_frames) if args.max_frames is not None else None
            ),
            "selected_num_frames": len(frames),
            "min_instance_area_px": max(
                1,
                int(args.min_instance_area_px),
            ),
        },
        "ground_truth_contract": {
            "policy_access": False,
            "source": "Habitat semantic sensor replayed after the live run",
            "unit": "visible semantic instance bounding box per saved camera pose",
            "visible_3d_center": (
                "Median world coordinate of valid saved online depth pixels "
                "inside the posthoc semantic instance mask."
            ),
            "positive_labels": ["door", "window"],
            "doorway_limit": (
                "A passable opening without door-labeled semantic pixels is not "
                "a positive box under this detector-only audit."
            ),
        },
        "object_index_size": len(object_index),
        "observed_unknown_semantic_ids": sorted(observed_unknown_ids),
        "rgb_replay_checks": rgb_checks,
        "rgb_replay_integrity": rgb_integrity,
        "frames": frames,
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_json": str(out_path),
                "selected_frames": len(frames),
                "visible_instances": sum(
                    frame["num_instances"] for frame in frames
                ),
                "target_instances": sum(
                    frame["num_target_instances"] for frame in frames
                ),
                "rgb_checks": len(rgb_checks),
                "unknown_semantic_ids": len(observed_unknown_ids),
            },
            indent=2,
        )
    )
    if rgb_integrity["enabled"] and not rgb_integrity["passed"]:
        raise RuntimeError(
            "RGB replay integrity gate failed; inspect rgb_replay_integrity"
        )


def _progress(stage: str, **details: Any) -> None:
    print(
        json.dumps(
            {"stage": stage, **details},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _create_session(
    *,
    scene: Path,
    scene_dataset_config: Path | None,
    resolution: int,
):
    import habitat_sim
    from habitat_sim import SensorSubType, SensorType

    sensor_specs = []
    for uuid, sensor_type in (
        ("rgb", SensorType.COLOR),
        ("semantic", SensorType.SEMANTIC),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = SensorSubType.PINHOLE
        spec.resolution = [resolution, resolution]
        spec.position = [0.0, 1.5, 0.0]
        sensor_specs.append(spec)

    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = str(scene)
    sim_config.enable_physics = False
    if scene_dataset_config is not None:
        sim_config.scene_dataset_config_file = str(scene_dataset_config)

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = sensor_specs
    simulator = habitat_sim.Simulator(
        habitat_sim.Configuration(sim_config, [agent_config])
    )
    return _ReplaySession(simulator)


class _ReplaySession:
    def __init__(self, simulator: Any) -> None:
        self.sim = simulator

    def close(self) -> None:
        self.sim.close()


def _set_replay_pose(agent, frame: dict[str, Any]) -> None:
    state = agent.get_state()
    state.position = np.asarray(
        frame["agent_position_xyz"],
        dtype=np.float32,
    )
    quaternion_type = type(state.rotation)
    state.rotation = _quaternion_from_rotation_matrix(
        frame["agent_rotation_matrix"],
        quaternion_type,
    )
    sensor_position = np.asarray(
        frame["sensor_position_xyz"],
        dtype=np.float32,
    )
    sensor_rotation = _quaternion_from_rotation_matrix(
        frame["sensor_rotation_matrix"],
        quaternion_type,
    )
    for sensor_state in state.sensor_states.values():
        sensor_state.position = sensor_position
        sensor_state.rotation = sensor_rotation
    agent.set_state(
        state,
        reset_sensors=False,
        infer_sensor_states=False,
    )


def _quaternion_from_rotation_matrix(
    value: Any,
    quaternion_type: Any,
) -> Any:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {matrix.shape}")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(
            1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
        ) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(
            1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
        ) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(
            1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
        ) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("Rotation matrix produced a zero-norm quaternion")
    return quaternion_type(w / norm, x / norm, y / norm, z / norm)


def _semantic_array(value: Any) -> np.ndarray:
    if value is None:
        raise RuntimeError("Habitat semantic sensor returned no observation")
    semantic = np.asarray(value)
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    if semantic.ndim != 2:
        raise ValueError(f"Unexpected semantic observation shape {semantic.shape}")
    return semantic.astype(np.int32, copy=False)


def _semantic_object_index(semantic_scene: Any) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for obj in getattr(semantic_scene, "objects", []):
        if obj is None or getattr(obj, "category", None) is None:
            continue
        semantic_id = int(obj.semantic_id)
        raw_category = str(obj.category.name()).strip().lower()
        center, size = _object_aabb(obj)
        result[semantic_id] = {
            "semantic_id": semantic_id,
            "object_id": str(obj.id),
            "raw_category": raw_category,
            "canonical_label": _canonical_gt_label(raw_category),
            "world_center_xyz": center,
            "world_size_xyz": size,
        }
    return result


def _object_aabb(obj: Any) -> tuple[list[float], list[float]]:
    aabb = getattr(obj, "aabb", None)
    if aabb is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    center_value = _attribute_or_call(aabb, "center")
    size_value = _attribute_or_call(aabb, "sizes")
    if size_value is None:
        size_value = _attribute_or_call(aabb, "size")
    if center_value is None or size_value is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    center = np.asarray(center_value, dtype=np.float32).reshape(-1)
    sizes = np.asarray(size_value, dtype=np.float32).reshape(-1)
    if center.size < 3 or sizes.size < 3:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return (
        [float(center[index]) for index in range(3)],
        [float(sizes[index]) for index in range(3)],
    )


def _attribute_or_call(value: Any, name: str) -> Any:
    attribute = getattr(value, name, None)
    return attribute() if callable(attribute) else attribute


def _canonical_gt_label(raw_category: str) -> str | None:
    normalized = " ".join(str(raw_category).strip().lower().split())
    if normalized == "door":
        return "door"
    if normalized == "window":
        return "window"
    return None


def _visible_instances(
    semantic: np.ndarray,
    object_index: dict[int, dict[str, Any]],
    *,
    min_area_px: int,
    depth: np.ndarray | None = None,
    sensor_position_xyz: Any = None,
    sensor_rotation_matrix: Any = None,
    min_depth_m: float = 0.05,
    max_depth_m: float = 6.0,
) -> list[dict[str, Any]]:
    ids, counts = np.unique(semantic, return_counts=True)
    rows: list[dict[str, Any]] = []
    for semantic_id_value, count_value in zip(ids, counts):
        semantic_id = int(semantic_id_value)
        area_px = int(count_value)
        obj = object_index.get(semantic_id)
        if obj is None or area_px < int(min_area_px):
            continue
        ys, xs = np.nonzero(semantic == semantic_id)
        if xs.size == 0:
            continue
        row = dict(obj)
        visible_geometry = None
        if row.get("canonical_label") in {"door", "window"}:
            visible_geometry = _visible_world_geometry(
                ys,
                xs,
                depth=depth,
                sensor_position_xyz=sensor_position_xyz,
                sensor_rotation_matrix=sensor_rotation_matrix,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
        row.update(
            {
                "area_px": area_px,
                "box": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
                "world_visible_center_xyz": (
                    visible_geometry["world_visible_center_xyz"]
                    if visible_geometry is not None
                    else None
                ),
                "visible_depth_median": (
                    visible_geometry["visible_depth_median"]
                    if visible_geometry is not None
                    else None
                ),
                "visible_depth_valid_ratio": (
                    visible_geometry["visible_depth_valid_ratio"]
                    if visible_geometry is not None
                    else None
                ),
                "visible_projected_points": (
                    visible_geometry["visible_projected_points"]
                    if visible_geometry is not None
                    else 0
                ),
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda item: (
            str(item["raw_category"]),
            int(item["semantic_id"]),
        )
    )
    return rows


def _saved_depth(
    path_value: Any,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    if path_value is None:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    depth = np.asarray(np.load(path), dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.shape != expected_shape:
        return None
    return depth


def _visible_world_geometry(
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    depth: np.ndarray | None,
    sensor_position_xyz: Any,
    sensor_rotation_matrix: Any,
    min_depth_m: float,
    max_depth_m: float,
) -> dict[str, Any] | None:
    if depth is None or sensor_position_xyz is None or sensor_rotation_matrix is None:
        return None
    z_all = depth[ys, xs].astype(np.float32)
    valid = (
        np.isfinite(z_all)
        & (z_all > float(min_depth_m))
        & (z_all < float(max_depth_m))
    )
    if int(valid.sum()) < 8:
        return None
    rows = ys[valid]
    cols = xs[valid]
    z = z_all[valid]
    max_points = 5000
    stride = max(1, int(math.ceil(z.size / max_points)))
    rows = rows[::stride]
    cols = cols[::stride]
    z = z[::stride]

    height, width = depth.shape
    fx = width / (2.0 * math.tan(math.radians(90.0) / 2.0))
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    x_camera = (cols.astype(np.float32) - cx) / fx * z
    y_camera = -(rows.astype(np.float32) - cy) / fy * z
    z_camera = -z
    camera_points = np.stack([x_camera, y_camera, z_camera], axis=1)
    rotation = np.asarray(sensor_rotation_matrix, dtype=np.float32)
    position = np.asarray(sensor_position_xyz, dtype=np.float32).reshape(1, 3)
    if rotation.shape != (3, 3) or position.shape != (1, 3):
        return None
    world = position + camera_points @ rotation.T
    center = np.median(world, axis=0)
    if not np.isfinite(center).all():
        return None
    return {
        "world_visible_center_xyz": [
            float(center[0]),
            float(center[1]),
            float(center[2]),
        ],
        "visible_depth_median": float(np.median(z)),
        "visible_depth_valid_ratio": float(valid.mean()),
        "visible_projected_points": int(world.shape[0]),
    }


def _select_frames(
    frames: list[dict[str, Any]],
    *,
    start: int,
    end: int | None,
    stride: int,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    selected = [
        frame
        for frame in frames
        if int(frame["frame_index"]) >= int(start)
        and (end is None or int(frame["frame_index"]) <= int(end))
        and (int(frame["frame_index"]) - int(start)) % int(stride) == 0
    ]
    if max_frames is not None:
        selected = selected[: max(0, int(max_frames))]
    return selected


def _rgb_replay_check(
    replay_value: Any,
    source_path_value: Any,
    *,
    frame_index: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frame_index": int(frame_index),
        "source_rgb_path": (
            str(source_path_value) if source_path_value is not None else None
        ),
        "available": False,
    }
    if replay_value is None or source_path_value is None:
        return result
    source_path = Path(str(source_path_value))
    if not source_path.exists():
        return result
    replay = np.asarray(replay_value)
    if replay.ndim == 3 and replay.shape[-1] == 4:
        replay = replay[..., :3]
    source = np.asarray(Image.open(source_path).convert("RGB"))
    if replay.shape != source.shape:
        result["shape_mismatch"] = {
            "replay": list(replay.shape),
            "source": list(source.shape),
        }
        return result
    delta = np.abs(
        replay.astype(np.float32) - source.astype(np.float32)
    )
    result.update(
        {
            "available": True,
            "mae": float(delta.mean()),
            "p95_abs_error": float(np.percentile(delta, 95)),
            "max_abs_error": float(delta.max()),
        }
    )
    return result


def _rgb_integrity_report(
    checks: list[dict[str, Any]],
    *,
    enabled: bool,
    max_mae: float,
    max_p95_abs_error: float,
) -> dict[str, Any]:
    available = [item for item in checks if item.get("available")]
    observed_max_mae = (
        max(float(item["mae"]) for item in available) if available else None
    )
    observed_max_p95 = (
        max(float(item["p95_abs_error"]) for item in available)
        if available
        else None
    )
    passed = (
        not enabled
        or (
            bool(checks)
            and len(available) == len(checks)
            and observed_max_mae is not None
            and observed_max_mae <= max_mae
            and observed_max_p95 is not None
            and observed_max_p95 <= max_p95_abs_error
        )
    )
    return {
        "enabled": enabled,
        "required_checks": len(checks),
        "available_checks": len(available),
        "max_mae_allowed": max_mae,
        "max_p95_abs_error_allowed": max_p95_abs_error,
        "observed_max_mae": observed_max_mae,
        "observed_max_p95_abs_error": observed_max_p95,
        "passed": passed,
    }


def _semantic_preview(semantic: np.ndarray) -> Image.Image:
    ids = semantic.astype(np.uint32)
    rgb = np.stack(
        [
            (ids * 37 + 17) % 255,
            (ids * 67 + 53) % 255,
            (ids * 97 + 91) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
