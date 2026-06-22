from __future__ import annotations

import argparse
import json
import os
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

from bev_memory import BEVMemory
from observation_adapter import HabitatObservationAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.2 live Habitat-Sim scene smoke.")
    parser.add_argument("--scene", required=True, help="Path to a Habitat-Sim loadable scene, e.g. .glb")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "phase22_live"))
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max-rays", type=int, default=13)
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=_env_int("RSCNAV_HABITAT_GPU_DEVICE_ID"),
        help="Habitat-Sim SimulatorConfiguration.gpu_device_id.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = Path(args.scene).expanduser().resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    import habitat_sim
    from habitat_sim import SensorSubType, SensorType

    sim, sensor_uuids = _make_simulator(
        habitat_sim=habitat_sim,
        SensorSubType=SensorSubType,
        SensorType=SensorType,
        scene_path=str(scene_path),
        resolution=args.resolution,
        gpu_device_id=args.gpu_device_id,
    )
    try:
        agent_state = _place_agent_on_navigable_point(sim)
        observations = sim.get_sensor_observations()
    finally:
        sim.close()

    rgb = _rgb_array(observations.get("rgb"))
    depth = _valid_depth(observations.get("depth"))
    render_quality = _validate_live_render(rgb, depth)
    semantic = observations.get("semantic") if "semantic" in sensor_uuids else None

    pose = _agent_state_to_bev_pose(agent_state)
    raw_observation = {
        "frame_id": "phase22_habitat_live_scene_001",
        "time": 1,
        "scene_id": str(scene_path),
        "episode_id": "phase22_live_scene_smoke",
        "pose": pose,
        "rgb": rgb,
        "depth": depth,
        "semantic": semantic,
    }
    adapter = HabitatObservationAdapter(
        scene_id=str(scene_path),
        episode_id="phase22_live_scene_smoke",
        hfov_deg=args.hfov_deg,
        max_rays=args.max_rays,
        max_depth=10.0,
    )
    frame = adapter.to_frame(raw_observation)
    bev_grid_size = (96, 96)
    bev_resolution = 0.1
    bev_origin = (
        pose["x"] - (bev_grid_size[0] // 2) * bev_resolution,
        pose["y"] - (bev_grid_size[1] // 2) * bev_resolution,
    )
    bev = BEVMemory(
        grid_size=bev_grid_size,
        resolution=bev_resolution,
        origin_world_xy=bev_origin,
    )
    projected = bev.update_from_frame(frame)

    assert frame.rgb_shape is not None
    assert frame.depth_shape == tuple(depth.shape)
    assert len(frame.rays) > 0
    assert int(bev.explored.sum()) > 1

    _save_rgb(rgb, out_dir / "rgb.png")
    _save_depth(depth, out_dir / "depth.png")
    if semantic is not None:
        _save_semantic(np.asarray(semantic), out_dir / "semantic.png")
    _plot_bev(bev, out_dir / "bev_overlay.png")

    log = {
        "phase": "phase22_habitat_live_scene_smoke",
        "goal": "render one live Habitat-Sim scene frame, convert to ObservationFrame, update BEV",
        "scene": str(scene_path),
        "gpu_device_id": args.gpu_device_id,
        "sensor_uuids": sensor_uuids,
        "agent_state": _agent_state_log(agent_state),
        "render_quality": render_quality,
        "observation_shapes": {
            key: list(value.shape) for key, value in observations.items() if hasattr(value, "shape")
        },
        "frame": frame.to_dict(),
        "projected_semantic_evidence": [evidence.to_dict() for evidence in projected],
        "bev_snapshot": bev.snapshot(),
        "depth_stats": {
            "min": float(np.nanmin(depth)),
            "max": float(np.nanmax(depth)),
            "mean": float(np.nanmean(depth)),
        },
    }
    (out_dir / "phase22_habitat_live_scene_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'phase22_habitat_live_scene_log.json'}")
    print(f"Wrote {out_dir / 'rgb.png'}")
    print(f"Wrote {out_dir / 'depth.png'}")
    print(f"Wrote {out_dir / 'bev_overlay.png'}")
    print("Sensors:", sensor_uuids)
    print("Rays:", len(frame.rays))
    print("Explored cells:", int(bev.explored.sum()))


def _make_simulator(
    habitat_sim,
    SensorSubType,
    SensorType,
    scene_path: str,
    resolution: int,
    gpu_device_id: int | None,
):
    sensor_specs = [
        _camera_spec(habitat_sim, SensorSubType, SensorType.COLOR, "rgb", resolution),
        _camera_spec(habitat_sim, SensorSubType, SensorType.DEPTH, "depth", resolution),
    ]
    try:
        sensor_specs.append(
            _camera_spec(habitat_sim, SensorSubType, SensorType.SEMANTIC, "semantic", resolution)
        )
        return _build_simulator(habitat_sim, scene_path, sensor_specs, gpu_device_id), [spec.uuid for spec in sensor_specs]
    except Exception:
        sensor_specs = sensor_specs[:2]
        return _build_simulator(habitat_sim, scene_path, sensor_specs, gpu_device_id), [spec.uuid for spec in sensor_specs]


def _camera_spec(habitat_sim, SensorSubType, sensor_type, uuid: str, resolution: int):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = SensorSubType.PINHOLE
    spec.resolution = [resolution, resolution]
    spec.position = [0.0, 1.5, 0.0]
    return spec


def _build_simulator(habitat_sim, scene_path: str, sensor_specs, gpu_device_id: int | None):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    if gpu_device_id is not None:
        sim_cfg.gpu_device_id = gpu_device_id
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _place_agent_on_navigable_point(sim):
    agent = sim.get_agent(0)
    state = agent.get_state()
    pathfinder = getattr(sim, "pathfinder", None)
    if pathfinder is None or not getattr(pathfinder, "is_loaded", False):
        return state

    point = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
    if point.shape == (3,) and np.isfinite(point).all():
        state.position = point
        agent.set_state(state)
        state = agent.get_state()
    return state


def _agent_state_to_bev_pose(agent_state) -> dict:
    position = np.asarray(getattr(agent_state, "position", [12.0, 0.0, 12.0]), dtype=np.float32)
    return {
        "x": float(position[0]),
        "y": float(position[2] if position.size > 2 else position[-1]),
        "heading_deg": _heading_deg_from_rotation(getattr(agent_state, "rotation", None)),
    }


def _agent_state_log(agent_state) -> dict:
    position = np.asarray(getattr(agent_state, "position", []), dtype=np.float32)
    return {
        "position": [float(v) for v in position.tolist()],
        "heading_deg": _heading_deg_from_rotation(getattr(agent_state, "rotation", None)),
    }


def _heading_deg_from_rotation(rotation) -> float:
    if rotation is None:
        return 0.0
    try:
        vector = np.asarray(rotation.transform_vector([0.0, 0.0, -1.0]), dtype=np.float32)
    except Exception:
        return 0.0
    if vector.size < 3 or not np.isfinite(vector).all():
        return 0.0
    return float(np.degrees(np.arctan2(vector[0], -vector[2])))


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    return int(raw)


def _rgb_array(raw_rgb) -> np.ndarray:
    if raw_rgb is None:
        raise RuntimeError("Habitat-Sim did not return an RGB observation")
    rgb = np.asarray(raw_rgb)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    return rgb.astype(np.uint8)


def _valid_depth(raw_depth) -> np.ndarray:
    if raw_depth is None:
        raise RuntimeError("Habitat-Sim did not return a depth observation")
    depth = np.asarray(raw_depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise RuntimeError("Habitat-Sim returned no positive finite depth values")
    return depth


def _validate_live_render(rgb: np.ndarray, depth: np.ndarray) -> dict:
    valid_depth = np.isfinite(depth) & (depth > 0)
    if not valid_depth.any():
        raise RuntimeError("Live Habitat render has no valid depth pixels")

    rgb_std = float(np.std(rgb.astype(np.float32)))
    valid_depth_values = depth[valid_depth]
    depth_span = float(np.nanmax(valid_depth_values) - np.nanmin(valid_depth_values))
    valid_depth_fraction = float(valid_depth.mean())
    if rgb_std <= 0.1 and depth_span <= 1e-4:
        raise RuntimeError("Live Habitat render appears blank: RGB and depth are both near-constant")

    return {
        "rgb_std": rgb_std,
        "depth_span": depth_span,
        "valid_depth_fraction": valid_depth_fraction,
    }


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(rgb).save(path)


def _save_depth(depth: np.ndarray, path: Path) -> None:
    norm = depth.copy()
    finite = np.isfinite(norm)
    if finite.any():
        low, high = float(np.nanmin(norm[finite])), float(np.nanmax(norm[finite]))
        if high > low:
            norm = (norm - low) / (high - low)
        else:
            norm = np.zeros_like(norm)
    Image.fromarray(np.uint8(np.clip(norm, 0, 1) * 255)).save(path)


def _save_semantic(semantic: np.ndarray, path: Path) -> None:
    values = semantic.astype(np.float32)
    if values.max() > values.min():
        values = (values - values.min()) / (values.max() - values.min())
    Image.fromarray(np.uint8(values * 255)).save(path)


def _plot_bev(bev: BEVMemory, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    state = bev.occupancy_state().T
    cmap = mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#333333"])
    ax.imshow(state, origin="lower", cmap=cmap, vmin=0, vmax=2, alpha=0.82)
    ax.set_title("Phase 2.2 Live Habitat Scene -> BEV")
    ax.set_xlim(-0.5, bev.grid_size[0] - 0.5)
    ax.set_ylim(-0.5, bev.grid_size[1] - 0.5)
    ax.set_xticks(range(0, bev.grid_size[0], 2))
    ax.set_yticks(range(0, bev.grid_size[1], 2))
    ax.grid(color="#dddddd", linewidth=0.5)
    if bev.trajectory:
        xs, ys = zip(*bev.trajectory)
        ax.plot(xs, ys, color="#1f77b4", marker="*", markersize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
