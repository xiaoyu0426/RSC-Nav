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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.4 dense BEV geometry evaluation.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "phase24_bev_eval"))
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--bev-resolution", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--move-amount", type=float, default=0.25)
    parser.add_argument("--turn-amount", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=len(DEFAULT_ACTIONS))
    args = parser.parse_args()

    scene = Path(args.scene).expanduser().resolve()
    if not scene.exists():
        raise FileNotFoundError(scene)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import habitat_sim
    from habitat_sim import SensorSubType, SensorType

    sim = _make_simulator(
        habitat_sim=habitat_sim,
        SensorSubType=SensorSubType,
        SensorType=SensorType,
        scene_path=str(scene),
        resolution=args.resolution,
        move_amount=args.move_amount,
        turn_amount=args.turn_amount,
    )
    try:
        agent_state = _reset_agent(sim)
        pose = _pose_xz(agent_state)
        config = DenseBEVConfig(
            grid_size=(args.grid_size, args.grid_size),
            resolution=args.bev_resolution,
            sample_stride=args.sample_stride,
            max_depth_m=6.0,
        )
        origin = (
            pose[0] - (config.grid_size[0] // 2) * config.resolution,
            pose[1] - (config.grid_size[1] // 2) * config.resolution,
        )
        mapper = DenseBEVMapper(origin_world_xz=origin, config=config)

        frames = []
        actions = DEFAULT_ACTIONS[: args.max_steps]
        for step, action in enumerate(["__initial__"] + actions):
            if action != "__initial__":
                sim.step(action)
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
            if step in {0, len(actions) // 2, len(actions)}:
                frame_prefix = f"frame_{step:03d}"
                rgb = _rgb_array(observations.get("rgb"))
                _save_rgb(rgb, out_dir / f"{frame_prefix}_rgb.png")
                _save_depth(depth, out_dir / f"{frame_prefix}_depth.png")
                frames.append(
                    {
                        "step": step,
                        "action": action,
                        "rgb": f"{frame_prefix}_rgb.png",
                        "depth": f"{frame_prefix}_depth.png",
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
        _write_summary_html(out_dir, scene, mapper.snapshot(), metrics, frames)
        report = {
            "phase": "phase24_bev_geometry_eval",
            "scene": str(scene),
            "actions": actions,
            "config": {
                "rgbd_resolution": args.resolution,
                "bev_resolution": args.bev_resolution,
                "grid_size": args.grid_size,
                "sample_stride": args.sample_stride,
                "move_amount": args.move_amount,
                "turn_amount": args.turn_amount,
            },
            "mapper_snapshot": mapper.snapshot(),
            "metrics": metrics,
            "frames": frames,
            "outputs": {
                "summary_html": str(out_dir / "summary.html"),
                "ours_bev": str(out_dir / "ours_bev.png"),
                "oracle_bev": str(out_dir / "oracle_bev.png"),
                "diff_bev": str(out_dir / "diff_bev.png"),
                "confidence": str(out_dir / "confidence.png"),
            },
        }
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["metrics"], indent=2))
        print(f"Wrote {out_dir / 'summary.html'}")
    finally:
        sim.close()


def _make_simulator(habitat_sim, SensorSubType, SensorType, scene_path: str, resolution: int, move_amount: float, turn_amount: float):
    sensor_specs = [
        _camera_spec(habitat_sim, SensorSubType, SensorType.COLOR, "rgb", resolution),
        _camera_spec(habitat_sim, SensorSubType, SensorType.DEPTH, "depth", resolution),
    ]
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
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


def _write_summary_html(out_dir: Path, scene: Path, snapshot: dict, metrics: dict, frames: list[dict]) -> None:
    rows = "\n".join(
        f"<tr><td>{key}</td><td>{value}</td></tr>"
        for key, value in metrics.items()
    )
    frame_html = "\n".join(
        f"<section><h3>Frame {frame['step']}</h3><img src='{frame['rgb']}'><img src='{frame['depth']}'></section>"
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
</div>
<h2>Sample Frames</h2>
<div class="grid">{frame_html}</div>
</body></html>"""
    (out_dir / "summary.html").write_text(html, encoding="utf-8")


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


if __name__ == "__main__":
    main()
