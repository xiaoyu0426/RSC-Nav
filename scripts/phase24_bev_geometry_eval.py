from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_bev_mapper import (
    DenseBEVConfig,
    DenseBEVMapper,
    depth_to_world_samples,
    mapping_metrics,
    oracle_navmesh_mask,
)


DEFAULT_ACTIONS = (
    ["move_forward"] * 8
    + ["turn_left"] * 2
    + ["move_forward"] * 8
    + ["turn_right"] * 2
    + ["move_forward"] * 8
    + ["turn_right"] * 2
    + ["move_forward"] * 8
)

SEMANTIC_COLORS = {
    "wall": "#2f2f2f",
    "door": "#1f77b4",
    "table": "#f28e2b",
    "chair": "#59a14f",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.4 dense BEV geometry evaluation.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "phase24_bev_eval"))
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--bev-resolution", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--obstacle-dilation-cells", type=int, default=1)
    parser.add_argument("--move-amount", type=float, default=0.25)
    parser.add_argument("--turn-amount", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=len(DEFAULT_ACTIONS))
    parser.add_argument("--trajectory-mode", choices=("path", "actions"), default="path")
    parser.add_argument("--path-min-distance", type=float, default=3.0)
    parser.add_argument("--scene-dataset-config")
    parser.add_argument("--semantic-categories", default="wall,door,table,chair")
    parser.add_argument("--semantic-confidence-saturation", type=float, default=80.0)
    parser.add_argument("--freshness-tau-steps", type=float, default=20.0)
    args = parser.parse_args()

    scene = Path(args.scene).expanduser().resolve()
    if not scene.exists():
        raise FileNotFoundError(scene)
    scene_dataset_config = Path(args.scene_dataset_config).expanduser().resolve() if args.scene_dataset_config else None
    if scene_dataset_config is not None and not scene_dataset_config.exists():
        raise FileNotFoundError(scene_dataset_config)
    semantic_categories = [item.strip().lower() for item in args.semantic_categories.split(",") if item.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import habitat_sim
    from habitat_sim import SensorSubType, SensorType

    sim = _make_simulator(
        habitat_sim=habitat_sim,
        SensorSubType=SensorSubType,
        SensorType=SensorType,
        scene_path=str(scene),
        scene_dataset_config=str(scene_dataset_config) if scene_dataset_config else None,
        resolution=args.resolution,
        move_amount=args.move_amount,
        turn_amount=args.turn_amount,
        enable_semantic=scene_dataset_config is not None,
    )
    try:
        agent_state = _reset_agent(sim)
        path_positions = []
        if args.trajectory_mode == "path":
            path_positions = _sample_navigable_path(
                sim,
                min_distance_m=args.path_min_distance,
                max_samples=args.max_steps + 1,
            )
            if path_positions:
                agent_state = _set_agent_pose(sim, path_positions[0], _next_point(path_positions, 0))
        pose = _pose_xz(agent_state)
        config = DenseBEVConfig(
            grid_size=(args.grid_size, args.grid_size),
            resolution=args.bev_resolution,
            sample_stride=args.sample_stride,
            max_depth_m=6.0,
            obstacle_dilation_radius_cells=args.obstacle_dilation_cells,
        )
        origin = (
            pose[0] - (config.grid_size[0] // 2) * config.resolution,
            pose[1] - (config.grid_size[1] // 2) * config.resolution,
        )
        mapper = DenseBEVMapper(origin_world_xz=origin, config=config)
        semantic_mapper = None
        if scene_dataset_config is not None:
            semantic_mapper = SemanticBEVAccumulator(
                mapper=mapper,
                semantic_scene=sim.semantic_scene,
                categories=semantic_categories,
                confidence_saturation=args.semantic_confidence_saturation,
                freshness_tau_steps=args.freshness_tau_steps,
            )

        frames = []
        actions = DEFAULT_ACTIONS[: args.max_steps]
        trajectory_steps = _trajectory_steps(args.trajectory_mode, actions, path_positions)
        last_step = max(0, len(trajectory_steps) - 1)
        mid_step = last_step // 2
        for step, item in enumerate(trajectory_steps):
            action = item["label"]
            if args.trajectory_mode == "actions" and action != "__initial__":
                sim.step(action)
            elif args.trajectory_mode == "path":
                agent_state = _set_agent_pose(sim, item["position"], item.get("next_position"))
            agent_state = sim.get_agent(0).get_state()
            observations = sim.get_sensor_observations()
            depth = _valid_depth(observations.get("depth"))
            sensor_state = agent_state.sensor_states.get("depth") or next(iter(agent_state.sensor_states.values()))
            snapshot = mapper.update_from_depth(
                depth=depth,
                agent_position_xyz=agent_state.position,
                sensor_position_xyz=sensor_state.position,
                sensor_rotation=sensor_state.rotation,
                hfov_deg=90.0,
            )
            if semantic_mapper is not None and observations.get("semantic") is not None:
                semantic_mapper.update_from_observation(
                    depth=depth,
                    semantic=_semantic_array(observations.get("semantic")),
                    sensor_position_xyz=np.asarray(sensor_state.position, dtype=np.float32),
                    sensor_rotation=sensor_state.rotation,
                    floor_y=float(agent_state.position[1]),
                    hfov_deg=90.0,
                    step=step,
                )
            if step in {0, mid_step, last_step}:
                frame_prefix = f"frame_{step:03d}"
                rgb = _rgb_array(observations.get("rgb"))
                _save_rgb(rgb, out_dir / f"{frame_prefix}_rgb.png")
                _save_depth(depth, out_dir / f"{frame_prefix}_depth.png")
                semantic_frame = None
                if semantic_mapper is not None and observations.get("semantic") is not None:
                    semantic_frame = f"{frame_prefix}_semantic.png"
                    _save_semantic_frame(
                        _semantic_array(observations.get("semantic")),
                        semantic_mapper.semantic_id_to_class,
                        semantic_mapper.categories,
                        out_dir / semantic_frame,
                    )
                frames.append(
                    {
                        "step": step,
                        "action": action,
                        "rgb": f"{frame_prefix}_rgb.png",
                        "depth": f"{frame_prefix}_depth.png",
                        "semantic": semantic_frame,
                        "snapshot": snapshot,
                    }
                )

        height = float(sim.get_agent(0).get_state().position[1])
        oracle_free = oracle_navmesh_mask(
            sim.pathfinder,
            mapper.origin_world_xz,
            mapper.config.grid_size,
            mapper.config.resolution,
            height,
        )
        pred_free = mapper.free_mask()
        pred_occupied = mapper.occupied_mask()
        metrics = mapping_metrics(
            pred_free=pred_free,
            pred_occupied=pred_occupied,
            explored=mapper.explored,
            oracle_free=oracle_free,
            resolution=mapper.config.resolution,
        )

        _plot_bev_layers(mapper, oracle_free, out_dir)
        semantic_report = None
        if semantic_mapper is not None:
            semantic_report = semantic_mapper.report()
            _plot_semantic_layers(semantic_mapper, mapper.trajectory, out_dir)
            (out_dir / "semantic_tracks.json").write_text(
                json.dumps(semantic_report, indent=2),
                encoding="utf-8",
            )
        _write_summary_html(out_dir, scene, mapper.snapshot(), metrics, frames, semantic_report)
        report = {
            "phase": "phase24_bev_geometry_eval",
            "scene": str(scene),
            "scene_dataset_config": str(scene_dataset_config) if scene_dataset_config else None,
            "actions": actions,
            "config": {
                "rgbd_resolution": args.resolution,
                "bev_resolution": args.bev_resolution,
                "grid_size": args.grid_size,
                "sample_stride": args.sample_stride,
                "obstacle_dilation_cells": args.obstacle_dilation_cells,
                "move_amount": args.move_amount,
                "turn_amount": args.turn_amount,
                "trajectory_mode": args.trajectory_mode,
                "path_min_distance_m": args.path_min_distance,
                "semantic_categories": semantic_categories,
                "freshness_tau_steps": args.freshness_tau_steps,
            },
            "mapper_snapshot": mapper.snapshot(),
            "metrics": metrics,
            "semantic_report": semantic_report,
            "frames": frames,
            "outputs": {
                "summary_html": str(out_dir / "summary.html"),
                "ours_bev": str(out_dir / "ours_bev.png"),
                "oracle_bev": str(out_dir / "oracle_bev.png"),
                "diff_bev": str(out_dir / "diff_bev.png"),
                "confidence": str(out_dir / "confidence.png"),
                "semantic_bev": str(out_dir / "semantic_bev.png") if semantic_mapper is not None else None,
                "semantic_confidence": str(out_dir / "semantic_confidence.png") if semantic_mapper is not None else None,
            },
        }
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["metrics"], indent=2))
        print(f"Wrote {out_dir / 'summary.html'}")
    finally:
        sim.close()


def _make_simulator(
    habitat_sim,
    SensorSubType,
    SensorType,
    scene_path: str,
    scene_dataset_config: str | None,
    resolution: int,
    move_amount: float,
    turn_amount: float,
    enable_semantic: bool,
):
    sensor_specs = [
        _camera_spec(habitat_sim, SensorSubType, SensorType.COLOR, "rgb", resolution),
        _camera_spec(habitat_sim, SensorSubType, SensorType.DEPTH, "depth", resolution),
    ]
    if enable_semantic:
        sensor_specs.append(_camera_spec(habitat_sim, SensorSubType, SensorType.SEMANTIC, "semantic", resolution))
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    if scene_dataset_config:
        sim_cfg.scene_dataset_config_file = scene_dataset_config
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=move_amount)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=turn_amount)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=turn_amount)
        ),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _camera_spec(habitat_sim, SensorSubType, sensor_type, uuid: str, resolution: int):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = SensorSubType.PINHOLE
    spec.resolution = [resolution, resolution]
    spec.position = [0.0, 1.5, 0.0]
    return spec


class SemanticBEVAccumulator:
    def __init__(
        self,
        mapper: DenseBEVMapper,
        semantic_scene,
        categories: list[str],
        confidence_saturation: float,
        freshness_tau_steps: float,
    ) -> None:
        self.mapper = mapper
        self.categories = categories
        self.category_to_index = {category: idx for idx, category in enumerate(categories)}
        self.semantic_id_to_class: dict[int, str] = {}
        self.semantic_id_to_gt: dict[int, dict] = {}
        self.evidence = np.zeros((len(categories), *mapper.config.grid_size), dtype=np.float32)
        self.instance_stats: dict[int, dict] = {}
        self.confidence_saturation = max(1.0, float(confidence_saturation))
        self.freshness_tau_steps = max(1.0, float(freshness_tau_steps))
        self.frame_seen: dict[int, set[int]] = {}
        self.latest_step = 0
        self._index_scene_objects(semantic_scene)

    def _index_scene_objects(self, semantic_scene) -> None:
        for obj in getattr(semantic_scene, "objects", []):
            if obj is None or obj.category is None:
                continue
            category = _match_category(obj.category.name(), self.categories)
            if category is None:
                continue
            semantic_id = int(obj.semantic_id)
            self.semantic_id_to_class[semantic_id] = category
            center = _object_center(obj)
            self.semantic_id_to_gt[semantic_id] = {
                "object_id": str(obj.id),
                "semantic_id": semantic_id,
                "category": category,
                "raw_category": obj.category.name(),
                "gt_center_xz": center,
            }

    def update_from_observation(
        self,
        depth: np.ndarray,
        semantic: np.ndarray,
        sensor_position_xyz: np.ndarray,
        sensor_rotation,
        floor_y: float,
        hfov_deg: float,
        step: int,
    ) -> None:
        samples = depth_to_world_samples(
            depth=depth,
            sensor_position_xyz=sensor_position_xyz,
            sensor_rotation=sensor_rotation,
            hfov_deg=hfov_deg,
            stride=self.mapper.config.sample_stride,
            min_depth_m=self.mapper.config.min_depth_m,
            max_depth_m=self.mapper.config.max_depth_m,
        )
        points_world = samples["points_world"]
        if points_world.size == 0:
            return
        rows = np.clip(samples["rows"], 0, semantic.shape[0] - 1)
        cols = np.clip(samples["cols"], 0, semantic.shape[1] - 1)
        semantic_ids = semantic[rows, cols].astype(np.int64)
        rel_y = points_world[:, 1] - float(floor_y)
        height_mask = np.logical_and(rel_y >= -0.3, rel_y <= 2.5)
        seen_ids: set[int] = set()
        self.latest_step = max(self.latest_step, int(step))

        for semantic_id, point, keep in zip(semantic_ids, points_world, height_mask):
            if not keep:
                continue
            semantic_id = int(semantic_id)
            category = self.semantic_id_to_class.get(semantic_id)
            if category is None:
                continue
            cell = self.mapper.world_to_grid((float(point[0]), float(point[2])))
            if cell is None:
                continue
            seen_ids.add(semantic_id)
            class_index = self.category_to_index[category]
            self.evidence[class_index, cell[0], cell[1]] += 1.0
            stats = self.instance_stats.setdefault(
                semantic_id,
                {
                    "semantic_id": semantic_id,
                    "category": category,
                    "count": 0,
                    "sum_x": 0.0,
                    "sum_z": 0.0,
                    "first_seen_step": int(step),
                    "last_seen_step": int(step),
                    "cells": set(),
                },
            )
            stats["count"] += 1
            stats["sum_x"] += float(point[0])
            stats["sum_z"] += float(point[2])
            stats["last_seen_step"] = int(step)
            stats["cells"].add(cell)
        if seen_ids:
            self.frame_seen.setdefault(int(step), set()).update(seen_ids)

    def semantic_state(self) -> np.ndarray:
        state = np.full(self.mapper.config.grid_size, -1, dtype=np.int16)
        if not self.categories:
            return state
        max_evidence = self.evidence.max(axis=0)
        state[max_evidence > 0] = np.argmax(self.evidence, axis=0)[max_evidence > 0]
        return state

    def confidence(self) -> np.ndarray:
        if not self.categories:
            return np.zeros(self.mapper.config.grid_size, dtype=np.float32)
        return np.clip(self.evidence.max(axis=0) / self.confidence_saturation, 0.0, 1.0)

    def report(self) -> dict:
        tracks = []
        centroid_errors = []
        per_class_errors: dict[str, list[float]] = {category: [] for category in self.categories}
        for semantic_id, stats in sorted(self.instance_stats.items()):
            count = int(stats["count"])
            centroid = [stats["sum_x"] / count, stats["sum_z"] / count] if count else [None, None]
            gt = self.semantic_id_to_gt.get(semantic_id, {})
            gt_center = gt.get("gt_center_xz")
            error = None
            if gt_center is not None and centroid[0] is not None:
                error = float(np.linalg.norm(np.asarray(centroid) - np.asarray(gt_center)))
                centroid_errors.append(error)
                per_class_errors[stats["category"]].append(error)
            visible_steps = self._visible_steps(semantic_id)
            fragmentation_count = max(0, _count_segments(visible_steps) - 1)
            age_steps = max(0, int(self.latest_step) - int(stats["last_seen_step"]))
            freshness = float(np.exp(-age_steps / self.freshness_tau_steps))
            tracks.append(
                {
                    "semantic_id": semantic_id,
                    "object_id": gt.get("object_id"),
                    "category": stats["category"],
                    "count": count,
                    "centroid_xz": centroid,
                    "gt_center_xz": gt_center,
                    "centroid_error_m": error,
                    "footprint_cells": len(stats["cells"]),
                    "confidence": min(1.0, count / self.confidence_saturation),
                    "freshness": freshness,
                    "age_steps": age_steps,
                    "visible_steps": visible_steps,
                    "visibility_segments": _count_segments(visible_steps),
                    "fragmentation_count": fragmentation_count,
                    "first_seen_step": int(stats["first_seen_step"]),
                    "last_seen_step": int(stats["last_seen_step"]),
                }
            )

        state = self.semantic_state()
        per_class_cells = {
            category: int((state == idx).sum())
            for category, idx in self.category_to_index.items()
        }
        per_class_mean_error = {
            category: (float(np.mean(values)) if values else None)
            for category, values in per_class_errors.items()
        }
        return {
            "categories": self.categories,
            "indexed_target_instances": len(self.semantic_id_to_class),
            "observed_target_instances": len(tracks),
            "semantic_cells": int((state >= 0).sum()),
            "per_class_cells": per_class_cells,
            "mean_centroid_error_m": float(np.mean(centroid_errors)) if centroid_errors else None,
            "per_class_mean_centroid_error_m": per_class_mean_error,
            "mean_fragmentation_count": float(np.mean([track["fragmentation_count"] for track in tracks])) if tracks else 0.0,
            "id_switches_upper_bound": 0,
            "mean_freshness": float(np.mean([track["freshness"] for track in tracks])) if tracks else 0.0,
            "tracks": tracks,
        }

    def _visible_steps(self, semantic_id: int) -> list[int]:
        return sorted(
            step
            for step, ids in self.frame_seen.items()
            if semantic_id in ids
        )


def _match_category(raw_name: str, categories: list[str]) -> str | None:
    name = (raw_name or "").lower()
    for category in categories:
        if category in name:
            return category
    return None


def _count_segments(steps: list[int]) -> int:
    if not steps:
        return 0
    segments = 1
    prev = steps[0]
    for step in steps[1:]:
        if step != prev + 1:
            segments += 1
        prev = step
    return segments


def _object_center(obj):
    aabb = getattr(obj, "aabb", None)
    center = getattr(aabb, "center", None)
    if callable(center):
        center = center()
    if center is None:
        return None
    arr = np.asarray(center, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[2])]


def _reset_agent(sim):
    agent = sim.get_agent(0)
    state = agent.get_state()
    pathfinder = getattr(sim, "pathfinder", None)
    if pathfinder is not None and getattr(pathfinder, "is_loaded", False):
        point = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if point.shape == (3,) and np.isfinite(point).all():
            state.position = point
    agent.set_state(state)
    return agent.get_state()


def _sample_navigable_path(sim, min_distance_m: float, max_samples: int, attempts: int = 80) -> list[np.ndarray]:
    pathfinder = getattr(sim, "pathfinder", None)
    if pathfinder is None or not getattr(pathfinder, "is_loaded", False):
        return []

    import habitat_sim

    best_points = []
    best_distance = -1.0
    for _ in range(attempts):
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        path.requested_end = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if not pathfinder.find_path(path):
            continue
        distance = float(path.geodesic_distance)
        points = [np.asarray(point, dtype=np.float32) for point in path.points]
        if distance > best_distance and len(points) >= 2:
            best_distance = distance
            best_points = points
        if distance >= min_distance_m and len(points) >= 2:
            return _resample_polyline(points, max_samples)
    return _resample_polyline(best_points, max_samples) if best_points else []


def _resample_polyline(points: list[np.ndarray], max_samples: int) -> list[np.ndarray]:
    if not points:
        return []
    if len(points) == 1 or max_samples <= 1:
        return points[:1]

    cumulative = [0.0]
    for prev, cur in zip(points[:-1], points[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(cur - prev)))
    total = cumulative[-1]
    if total <= 0.0:
        return points[:1]

    targets = np.linspace(0.0, total, max_samples)
    out = []
    segment = 0
    for target in targets:
        while segment + 1 < len(cumulative) and cumulative[segment + 1] < target:
            segment += 1
        if segment + 1 >= len(points):
            out.append(points[-1].copy())
            continue
        span = cumulative[segment + 1] - cumulative[segment]
        alpha = 0.0 if span <= 0.0 else (target - cumulative[segment]) / span
        out.append((1.0 - alpha) * points[segment] + alpha * points[segment + 1])
    return [np.asarray(point, dtype=np.float32) for point in out]


def _trajectory_steps(mode: str, actions: list[str], path_positions: list[np.ndarray]) -> list[dict]:
    if mode == "path" and path_positions:
        return [
            {
                "label": "path_waypoint",
                "position": position,
                "next_position": _next_point(path_positions, idx),
            }
            for idx, position in enumerate(path_positions)
        ]
    return [{"label": "__initial__"}] + [{"label": action} for action in actions]


def _next_point(points: list[np.ndarray], idx: int):
    if not points:
        return None
    if idx + 1 < len(points):
        return points[idx + 1]
    if idx > 0:
        return points[idx - 1]
    return None


def _set_agent_pose(sim, position: np.ndarray, look_at):
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    if look_at is not None:
        rotation = _rotation_toward(state.position, np.asarray(look_at, dtype=np.float32))
        if rotation is not None:
            state.rotation = rotation
    try:
        agent.set_state(state, infer_sensor_states=True)
    except TypeError:
        agent.set_state(state)
    return agent.get_state()


def _rotation_toward(position: np.ndarray, look_at: np.ndarray):
    direction = np.asarray(look_at - position, dtype=np.float32)
    norm = float(np.linalg.norm(direction[[0, 2]]))
    if norm <= 1e-6:
        return None
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    yaw = float(np.arctan2(-direction[0], -direction[2]))
    import quaternion

    return quaternion.from_rotation_vector([0.0, yaw, 0.0])


def _pose_xz(agent_state) -> tuple[float, float]:
    position = np.asarray(agent_state.position, dtype=np.float32)
    return float(position[0]), float(position[2])


def _plot_bev_layers(mapper: DenseBEVMapper, oracle_free: np.ndarray, out_dir: Path) -> None:
    state = mapper.occupancy_state()
    _save_mask_plot(
        state.T,
        out_dir / "ours_bev.png",
        title="Dense RGB-D BEV",
        cmap=mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#333333"]),
        vmin=0,
        vmax=2,
        trajectory=mapper.trajectory,
    )
    _save_mask_plot(
        oracle_free.T.astype(np.int8),
        out_dir / "oracle_bev.png",
        title="Habitat Navmesh Oracle",
        cmap=mcolors.ListedColormap(["#333333", "#ffffff"]),
        vmin=0,
        vmax=1,
        trajectory=mapper.trajectory,
    )
    diff = np.zeros(mapper.config.grid_size, dtype=np.int8)
    diff[np.logical_and(mapper.free_mask(), oracle_free)] = 1
    diff[np.logical_and(mapper.free_mask(), ~oracle_free)] = 2
    diff[np.logical_and(mapper.occupied_mask(), oracle_free)] = 3
    diff[np.logical_and(mapper.occupied_mask(), ~oracle_free)] = 4
    _save_mask_plot(
        diff.T,
        out_dir / "diff_bev.png",
        title="BEV vs Oracle Diff",
        cmap=mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#f4a261", "#4ea8de", "#333333"]),
        vmin=0,
        vmax=4,
        trajectory=mapper.trajectory,
    )
    _save_mask_plot(
        mapper.confidence().T,
        out_dir / "confidence.png",
        title="BEV Occupancy Confidence",
        cmap="viridis",
        vmin=0,
        vmax=1,
        trajectory=mapper.trajectory,
    )


def _plot_semantic_layers(semantic_mapper: SemanticBEVAccumulator, trajectory, out_dir: Path) -> None:
    state = semantic_mapper.semantic_state()
    colors = ["#d9d9d9"] + [
        SEMANTIC_COLORS.get(category, "#9467bd")
        for category in semantic_mapper.categories
    ]
    _save_mask_plot(
        (state + 1).T,
        out_dir / "semantic_bev.png",
        title="Semantic GT BEV",
        cmap=mcolors.ListedColormap(colors),
        vmin=0,
        vmax=len(colors) - 1,
        trajectory=trajectory,
    )
    _save_mask_plot(
        semantic_mapper.confidence().T,
        out_dir / "semantic_confidence.png",
        title="Semantic BEV Confidence",
        cmap="magma",
        vmin=0,
        vmax=1,
        trajectory=trajectory,
    )


def _save_mask_plot(array: np.ndarray, path: Path, title: str, cmap, vmin: float, vmax: float, trajectory) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(array, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    if trajectory:
        xs, ys = zip(*trajectory)
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.2)
        ax.plot(xs[-1], ys[-1], color="#1f77b4", marker="*", markersize=10)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_summary_html(
    out_dir: Path,
    scene: Path,
    snapshot: dict,
    metrics: dict,
    frames: list[dict],
    semantic_report: dict | None = None,
) -> None:
    rows = "\n".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>"
        for key, value in metrics.items()
    )
    semantic_rows = ""
    semantic_maps = ""
    if semantic_report is not None:
        semantic_summary = {
            key: value
            for key, value in semantic_report.items()
            if key != "tracks"
        }
        semantic_rows = "\n".join(
            f"<tr><td>{key}</td><td>{json.dumps(value, ensure_ascii=False)}</td></tr>"
            for key, value in semantic_summary.items()
        )
        semantic_maps = """
<section><h3>Semantic</h3><img src="semantic_bev.png"></section>
<section><h3>Semantic Confidence</h3><img src="semantic_confidence.png"></section>
"""
    frame_html = "\n".join(
        _frame_html(frame)
        for frame in frames
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase 2.4 BEV Eval</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
img {{ max-width: 360px; margin: 6px; border: 1px solid #ccc; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }}
table {{ border-collapse: collapse; }}
td {{ border: 1px solid #ddd; padding: 6px 10px; }}
</style></head><body>
<h1>Phase 2.4 Dense BEV Geometry Evaluation</h1>
<p><strong>Scene:</strong> {scene}</p>
<p><strong>Explored:</strong> {snapshot['num_explored_cells']} cells,
<strong>Occupied:</strong> {snapshot['num_occupied_cells']} cells,
<strong>Mean confidence:</strong> {snapshot['mean_confidence']:.3f}</p>
<h2>Metrics</h2><table>{rows}</table>
<h2>Maps</h2>
<div class="grid">
<section><h3>Ours</h3><img src="ours_bev.png"></section>
<section><h3>Oracle</h3><img src="oracle_bev.png"></section>
<section><h3>Diff</h3><img src="diff_bev.png"></section>
<section><h3>Confidence</h3><img src="confidence.png"></section>
{semantic_maps}
</div>
<h2>Semantic Metrics</h2><table>{semantic_rows}</table>
<h2>Sample Frames</h2>
<div class="grid">{frame_html}</div>
</body></html>"""
    (out_dir / "summary.html").write_text(html, encoding="utf-8")


def _frame_html(frame: dict) -> str:
    semantic = f"<img src='{frame['semantic']}'>" if frame.get("semantic") else ""
    return f"<section><h3>Frame {frame['step']}</h3><img src='{frame['rgb']}'><img src='{frame['depth']}'>{semantic}</section>"


def _rgb_array(raw_rgb) -> np.ndarray:
    if raw_rgb is None:
        raise RuntimeError("missing RGB observation")
    rgb = np.asarray(raw_rgb)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    return rgb.astype(np.uint8)


def _valid_depth(raw_depth) -> np.ndarray:
    if raw_depth is None:
        raise RuntimeError("missing depth observation")
    depth = np.asarray(raw_depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    return depth


def _semantic_array(raw_semantic) -> np.ndarray:
    if raw_semantic is None:
        raise RuntimeError("missing semantic observation")
    semantic = np.asarray(raw_semantic)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[:, :, 0]
    return semantic.astype(np.int64)


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(rgb).save(path)


def _save_depth(depth: np.ndarray, path: Path) -> None:
    valid = np.isfinite(depth)
    norm = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        low = float(np.nanmin(depth[valid]))
        high = float(np.nanmax(depth[valid]))
        if high > low:
            norm = (depth - low) / (high - low)
    Image.fromarray(np.uint8(np.clip(norm, 0, 1) * 255)).save(path)


def _save_semantic_frame(
    semantic: np.ndarray,
    semantic_id_to_class: dict[int, str],
    categories: list[str],
    path: Path,
) -> None:
    class_to_index = {category: idx for idx, category in enumerate(categories)}
    target = np.full(semantic.shape, -1, dtype=np.int16)
    for semantic_id in np.unique(semantic):
        category = semantic_id_to_class.get(int(semantic_id))
        if category is None:
            continue
        target[semantic == semantic_id] = class_to_index[category]
    colors = ["#d9d9d9"] + [SEMANTIC_COLORS.get(category, "#9467bd") for category in categories]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(target + 1, cmap=mcolors.ListedColormap(colors), vmin=0, vmax=len(colors) - 1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
