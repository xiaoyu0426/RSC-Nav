from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bev_memory import BEVMemory
from observation_adapter import HabitatObservationAdapter


OUT_DIR = ROOT / "outputs" / "phase22_sim"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.2 Habitat-Sim NONE-scene smoke.")
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=_env_int("RSCNAV_HABITAT_GPU_DEVICE_ID"),
        help="Habitat-Sim SimulatorConfiguration.gpu_device_id.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import habitat_sim
    from habitat_sim import SensorSubType, SensorType

    sim, sensor_uuids = _make_simulator(
        habitat_sim,
        SensorSubType,
        SensorType,
        gpu_device_id=args.gpu_device_id,
    )
    try:
        observations = sim.get_sensor_observations()
    finally:
        sim.close()

    rgb = observations.get("rgb")
    depth = _valid_depth(observations.get("depth"))
    semantic = observations.get("semantic") if "semantic" in sensor_uuids else None

    raw_observation = {
        "frame_id": "phase22_habitat_sim_none_001",
        "time": 1,
        "scene_id": "habitat_sim_NONE",
        "episode_id": "phase22_none_scene_smoke",
        "pose": {"x": 10.0, "y": 10.0, "heading_deg": 0.0},
        "rgb": rgb,
        "depth": depth,
        "semantic": semantic,
    }
    adapter = HabitatObservationAdapter(
        scene_id="habitat_sim_NONE",
        episode_id="phase22_none_scene_smoke",
        hfov_deg=90.0,
        max_rays=9,
        max_depth=10.0,
    )
    frame = adapter.to_frame(raw_observation)
    bev = BEVMemory(grid_size=(24, 24), resolution=1.0)
    projected = bev.update_from_frame(frame)

    assert frame.rgb_shape is not None
    assert frame.depth_shape == (64, 64)
    assert len(frame.rays) > 0
    assert int(bev.explored.sum()) > 1

    log = {
        "phase": "phase22_habitat_sim_none_smoke",
        "goal": "start Habitat-Sim headless without dataset, read RGB-D, convert to ObservationFrame, update BEV",
        "gpu_device_id": args.gpu_device_id,
        "sensor_uuids": sensor_uuids,
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
    (OUT_DIR / "phase22_habitat_sim_none_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR / 'phase22_habitat_sim_none_log.json'}")
    print("Sensors:", sensor_uuids)
    print("Rays:", len(frame.rays))
    print("Explored cells:", int(bev.explored.sum()))


def _make_simulator(habitat_sim, SensorSubType, SensorType, gpu_device_id: int | None):
    sensor_specs = [
        _camera_spec(habitat_sim, SensorSubType, SensorType.COLOR, "rgb"),
        _camera_spec(habitat_sim, SensorSubType, SensorType.DEPTH, "depth"),
    ]
    try:
        sensor_specs.append(
            _camera_spec(habitat_sim, SensorSubType, SensorType.SEMANTIC, "semantic")
        )
        return _build_simulator(habitat_sim, sensor_specs, gpu_device_id), [spec.uuid for spec in sensor_specs]
    except Exception:
        sensor_specs = sensor_specs[:2]
        return _build_simulator(habitat_sim, sensor_specs, gpu_device_id), [spec.uuid for spec in sensor_specs]


def _camera_spec(habitat_sim, SensorSubType, sensor_type, uuid: str):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = SensorSubType.PINHOLE
    spec.resolution = [64, 64]
    spec.position = [0.0, 1.5, 0.0]
    return spec


def _build_simulator(habitat_sim, sensor_specs, gpu_device_id: int | None):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = False
    if gpu_device_id is not None:
        sim_cfg.gpu_device_id = gpu_device_id
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    return int(raw)


def _valid_depth(raw_depth) -> np.ndarray:
    if raw_depth is None:
        raise RuntimeError("Habitat-Sim did not return a depth observation")
    depth = np.asarray(raw_depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    valid = np.isfinite(depth) & (depth > 0)
    if valid.any():
        return depth
    return np.full(depth.shape, 10.0, dtype=np.float32)


if __name__ == "__main__":
    main()
