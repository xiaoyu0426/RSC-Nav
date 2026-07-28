from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from dense_bev_mapper import oracle_navmesh_mask  # noqa: E402
from phase23_habitat_control_server import HabitatControlSession, ensure_conda_nvidia_egl_vendor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc navmesh coverage audit for an online exploration run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dataset-config")
    args = parser.parse_args()

    ensure_conda_nvidia_egl_vendor()
    run_dir = Path(args.run_dir).expanduser().resolve()
    state = np.load(run_dir / "online_bev_state.npz")
    explored = np.asarray(state["explored"], dtype=bool)
    origin_world_xz = tuple(float(value) for value in state["origin_world_xz"])
    resolution = float(np.asarray(state["resolution"]).reshape(-1)[0])

    metadata = _read_json(run_dir / "frames_metadata.json")
    frames = metadata.get("frames", [])
    if not frames:
        raise ValueError("frames_metadata.json contains no frames")
    floor_y = float(frames[0]["agent_position_xyz"][1])

    session = HabitatControlSession(
        scene=Path(args.scene).expanduser().resolve(),
        scene_dataset_config=(
            Path(args.scene_dataset_config).expanduser().resolve()
            if args.scene_dataset_config
            else None
        ),
        resolution=64,
        move_amount=0.25,
        turn_amount=30.0,
        enable_oracle_metrics=False,
        enable_semantic_sensor=False,
        enable_autopilot_start_path=False,
    )
    try:
        navmesh = oracle_navmesh_mask(
            session.sim.pathfinder,
            origin_world_xz=origin_world_xz,
            grid_size=explored.shape,
            resolution=resolution,
            height=floor_y,
        )
    finally:
        session.close()

    denominator = int(navmesh.sum())
    covered = int((explored & navmesh).sum())
    metrics = {
        "evaluation_role": "posthoc_only_not_available_to_online_policy",
        "scene": str(Path(args.scene).expanduser().resolve()),
        "floor_y": floor_y,
        "grid_size": list(explored.shape),
        "resolution": resolution,
        "navmesh_cells_in_grid": denominator,
        "explored_navmesh_cells": covered,
        "navmesh_observation_coverage": covered / denominator if denominator else 0.0,
        "online_policy_semantic_oracle_access": False,
        "online_policy_navmesh_role": metadata.get("online_contract", {}).get(
            "navmesh_online_usage"
        ),
    }
    output = run_dir / "posthoc_coverage_metrics.json"
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
