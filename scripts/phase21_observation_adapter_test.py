from __future__ import annotations

import json
import sys
from math import cos, radians, sin
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bev_memory import BEVMemory, make_phase02_synthetic_observations
from observation_adapter import MockHabitatObservationAdapter, SyntheticObservationAdapter
from observation_types import ObservationFrame
from rsc_nav_memory import SemanticSpatialMemory


OUT_DIR = ROOT / "outputs" / "phase21"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_adapter = SyntheticObservationAdapter(
        scene_id="phase21_scene",
        episode_id="phase21_synthetic_episode",
    )
    synthetic_observations = make_phase02_synthetic_observations()
    frames = [synthetic_adapter.to_frame(obs) for obs in synthetic_observations]

    mock_habitat_raw = _make_mock_habitat_observation(frames[0])
    mock_habitat_frame = MockHabitatObservationAdapter().to_frame(mock_habitat_raw)
    assert len(frames) == 4
    assert mock_habitat_frame.rgb_shape == (64, 64, 3)
    assert mock_habitat_frame.depth_shape == (64, 64)
    assert len(mock_habitat_frame.rays) == 3

    bev = BEVMemory(grid_size=(16, 16), resolution=1.0)
    long_term_memory = SemanticSpatialMemory(grid_size=(16, 16), scene_id="phase21_scene")
    sequence = []

    for frame in frames:
        projected = bev.update_from_frame(frame)
        for evidence in projected:
            long_term_memory.observe(
                label=evidence.label,
                position=evidence.grid_coord,
                confidence=evidence.confidence,
                time=evidence.time,
                source_view_id=evidence.source_view_id,
            )
        sequence.append(
            {
                "frame": frame.to_dict(),
                "projected_semantic_evidence": [evidence.to_dict() for evidence in projected],
                "bev_snapshot": bev.snapshot(),
                "long_term_memory": long_term_memory.snapshot(),
            }
        )

    retrieval = long_term_memory.retrieve("sofa", current_position=(1, 1), top_k=3)
    assert int(bev.explored.sum()) == 33
    assert int((bev.occupancy_logodds > 0.2).sum()) == 6
    assert int((bev.semantic_confidence > 0).sum()) == 4
    assert retrieval and retrieval[0].item.semantic_label == "sofa"
    assert retrieval[0].item.bev_position == (7, 9)

    log = {
        "phase": "phase21_observation_interface",
        "goal": "convert source-specific observations into a unified ObservationFrame before BEV memory update",
        "adapters": {
            "synthetic_adapter": frames[0].to_dict(),
            "mock_habitat_adapter": mock_habitat_frame.to_dict(),
        },
        "contract_checks": {
            "synthetic_frame_count": len(frames),
            "mock_habitat_frame_has_rgb_shape": mock_habitat_frame.rgb_shape is not None,
            "mock_habitat_frame_has_depth_shape": mock_habitat_frame.depth_shape is not None,
            "mock_habitat_frame_ray_count": len(mock_habitat_frame.rays),
            "bev_update_entrypoint": "BEVMemory.update_from_frame",
        },
        "bev_snapshot": bev.snapshot(),
        "semantic_spatial_memory": long_term_memory.snapshot(),
        "sofa_retrieval": [result.to_dict() for result in retrieval],
        "sequence": sequence,
    }
    (OUT_DIR / "phase21_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_frame_debug(frames, mock_habitat_frame)
    _plot_adapter_overlay(bev, long_term_memory, retrieval)

    print(f"Wrote {OUT_DIR / 'phase21_log.json'}")
    print(f"Wrote {OUT_DIR / 'observation_frame_debug.png'}")
    print(f"Wrote {OUT_DIR / 'adapter_bev_overlay.png'}")
    print("Unified frames:", len(frames))
    print("Mock Habitat frame rays:", len(mock_habitat_frame.rays))
    print("Sofa retrieval top-1:", retrieval[0].item.id, retrieval[0].item.bev_position, retrieval[0].score)


def _make_mock_habitat_observation(frame: ObservationFrame) -> dict:
    object_rays = [ray for ray in frame.rays if ray.hit_type == "object" and ray.semantic_label]
    non_object_rays = [ray for ray in frame.rays if ray.hit_type != "object"]
    return {
        "frame_id": "mock_habitat_001",
        "time": frame.time,
        "scene_id": "mock_habitat_scene",
        "episode_id": "mock_habitat_episode",
        "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
        "depth": np.ones((64, 64), dtype=np.float32),
        "camera_intrinsics": {
            "width": 64,
            "height": 64,
            "fx": 60.0,
            "fy": 60.0,
            "cx": 32.0,
            "cy": 32.0,
        },
        "pose": frame.pose.to_dict(),
        "rays": [
            {
                "angle_deg": ray.angle_deg,
                "distance": ray.distance,
                "hit_type": ray.hit_type,
            }
            for ray in non_object_rays
        ],
        "semantic_detections": [
            {
                "angle_deg": ray.angle_deg,
                "distance": ray.distance,
                "label": ray.semantic_label,
                "confidence": ray.semantic_confidence,
            }
            for ray in object_rays
        ],
    }


def _plot_frame_debug(frames: list[ObservationFrame], mock_habitat_frame: ObservationFrame) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(16, 4.5))
    colors = {"free": "#4c78a8", "obstacle": "#333333", "object": "#2ca02c"}
    for ax, frame in zip(axes, frames):
        ax.set_title(f"{frame.frame_id}\n{frame.scene_id}")
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(-0.5, 15.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, 16, 2))
        ax.set_yticks(range(0, 16, 2))
        ax.grid(color="#dddddd", linewidth=0.5)
        ax.scatter(frame.pose.x, frame.pose.y, marker="*", s=120, color="#1f77b4")
        heading_end = _endpoint(frame.pose.x, frame.pose.y, frame.pose.heading_deg, 1.1)
        ax.annotate(
            "",
            xy=heading_end,
            xytext=(frame.pose.x, frame.pose.y),
            arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 2},
        )
        for ray in frame.rays:
            endpoint = _endpoint(
                frame.pose.x,
                frame.pose.y,
                frame.pose.heading_deg + ray.angle_deg,
                ray.distance,
            )
            ax.plot(
                [frame.pose.x, endpoint[0]],
                [frame.pose.y, endpoint[1]],
                color=colors.get(ray.hit_type, "#999999"),
                linewidth=1.5,
            )
            ax.scatter(endpoint[0], endpoint[1], color=colors.get(ray.hit_type, "#999999"), s=35)
            if ray.semantic_label:
                ax.text(endpoint[0] + 0.2, endpoint[1] + 0.2, ray.semantic_label, fontsize=8)
    fig.suptitle(
        "Phase 2.1 ObservationFrame debug: pose, heading, projected rays, semantic endpoints\n"
        f"mock Habitat adapter sample: {mock_habitat_frame.frame_id}, "
        f"rgb={mock_habitat_frame.rgb_shape}, depth={mock_habitat_frame.depth_shape}"
    )
    _save(fig, "observation_frame_debug.png", rect=(0, 0, 1, 0.82))


def _plot_adapter_overlay(bev: BEVMemory, memory: SemanticSpatialMemory, retrieval) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    state = bev.occupancy_state().T
    cmap = mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#333333"])
    ax.imshow(state, origin="lower", cmap=cmap, vmin=0, vmax=2, alpha=0.8)
    ax.set_title("Phase 2.1 BEV updated only through ObservationFrame")
    ax.set_xlim(-0.5, bev.grid_size[0] - 0.5)
    ax.set_ylim(-0.5, bev.grid_size[1] - 0.5)
    ax.set_xticks(range(bev.grid_size[0]))
    ax.set_yticks(range(bev.grid_size[1]))
    ax.grid(color="#dddddd", linewidth=0.5)
    if bev.trajectory:
        xs, ys = zip(*bev.trajectory)
        ax.plot(xs, ys, color="#1f77b4", linewidth=2.0, marker="*", markersize=9)
    for item in memory.items.values():
        x, y = item.bev_position
        ax.scatter(x, y, s=350, facecolors="none", edgecolors="#d62728", linewidth=2.0)
        ax.text(x + 0.15, y + 0.15, item.id, color="#d62728", fontsize=8)
    if retrieval:
        top = retrieval[0].item
        ax.annotate(
            "retrieval top-1",
            xy=top.bev_position,
            xytext=(top.bev_position[0] + 1.2, top.bev_position[1] + 1.0),
            arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 2},
            color="#1f77b4",
            fontsize=9,
        )
    _save(fig, "adapter_bev_overlay.png")


def _endpoint(x: float, y: float, angle_deg: float, distance: float) -> tuple[float, float]:
    theta = radians(angle_deg)
    return (x + distance * cos(theta), y + distance * sin(theta))


def _save(fig, name: str, rect: tuple[float, float, float, float] | None = None) -> None:
    fig.tight_layout(rect=rect)
    fig.savefig(OUT_DIR / name, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
