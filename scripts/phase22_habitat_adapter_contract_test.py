from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bev_memory import BEVMemory
from observation_adapter import HabitatObservationAdapter
from rsc_nav_memory import SemanticSpatialMemory


OUT_DIR = ROOT / "outputs" / "phase22"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_observation = _make_habitat_like_rgbd_semantic_observation()
    adapter = HabitatObservationAdapter(
        scene_id="phase22_contract_scene",
        episode_id="phase22_contract_episode",
        hfov_deg=90.0,
        max_rays=9,
        max_depth=8.0,
        semantic_id_to_label={1: "sofa", 2: "table"},
    )
    frame = adapter.to_frame(raw_observation)

    bev = BEVMemory(grid_size=(20, 20), resolution=1.0)
    long_term_memory = SemanticSpatialMemory(grid_size=(20, 20), scene_id=frame.scene_id)
    projected = bev.update_from_frame(frame)
    for evidence in projected:
        long_term_memory.observe(
            label=evidence.label,
            position=evidence.grid_coord,
            confidence=evidence.confidence,
            time=evidence.time,
            source_view_id=evidence.source_view_id,
        )

    retrieval = long_term_memory.retrieve("sofa", current_position=(8, 6), top_k=3)
    _assert_contract(frame, projected, retrieval)

    log = {
        "phase": "phase22_habitat_adapter_contract",
        "goal": "convert Habitat-like rgb/depth/semantic/pose arrays into ObservationFrame and BEV memory",
        "environment": {
            "habitat_sim_importable": importlib.util.find_spec("habitat_sim") is not None,
            "habitat_lab_importable": importlib.util.find_spec("habitat") is not None,
        },
        "frame": frame.to_dict(),
        "projected_semantic_evidence": [evidence.to_dict() for evidence in projected],
        "bev_snapshot": bev.snapshot(),
        "semantic_spatial_memory": long_term_memory.snapshot(),
        "sofa_retrieval": [result.to_dict() for result in retrieval],
    }
    (OUT_DIR / "phase22_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "observation_frame.json").write_text(
        json.dumps(frame.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_habitat_like_inputs(raw_observation)
    _plot_phase22_overlay(bev, long_term_memory, retrieval)

    print(f"Wrote {OUT_DIR / 'phase22_log.json'}")
    print(f"Wrote {OUT_DIR / 'observation_frame.json'}")
    print(f"Wrote {OUT_DIR / 'habitat_like_inputs.png'}")
    print(f"Wrote {OUT_DIR / 'habitat_adapter_bev_overlay.png'}")
    print("Habitat-like rays:", len(frame.rays))
    print("Projected semantic evidence:", len(projected))
    print("Sofa retrieval top-1:", retrieval[0].item.id, retrieval[0].item.bev_position, retrieval[0].score)


def _make_habitat_like_rgbd_semantic_observation() -> dict:
    height, width = 64, 64
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, :] = np.array([40, 45, 52], dtype=np.uint8)

    depth = np.full((height, width), 6.0, dtype=np.float32)
    semantic = np.zeros((height, width), dtype=np.int32)

    rgb[18:46, 28:36] = np.array([70, 140, 210], dtype=np.uint8)
    depth[18:46, 28:36] = 4.0
    semantic[18:46, 28:36] = 1

    rgb[22:42, 44:52] = np.array([210, 170, 75], dtype=np.uint8)
    depth[22:42, 44:52] = 3.0
    semantic[22:42, 44:52] = 2

    return {
        "frame_id": "phase22_habitat_like_001",
        "time": 1,
        "scene_id": "phase22_contract_scene",
        "episode_id": "phase22_contract_episode",
        "pose": {"x": 8.0, "y": 8.0, "heading_deg": 90.0},
        "rgb": rgb,
        "depth": depth,
        "semantic": semantic,
    }


def _assert_contract(frame, projected, retrieval) -> None:
    assert frame.rgb_shape == (64, 64, 3)
    assert frame.depth_shape == (64, 64)
    assert len(frame.rays) == 9
    assert any(ray.semantic_label == "sofa" for ray in frame.rays)
    assert any(ray.semantic_label == "table" for ray in frame.rays)
    assert any(evidence.label == "sofa" for evidence in projected)
    assert retrieval and retrieval[0].item.semantic_label == "sofa"


def _plot_habitat_like_inputs(raw_observation: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw_observation["rgb"])
    axes[0].set_title("RGB")
    axes[1].imshow(raw_observation["depth"], cmap="viridis")
    axes[1].set_title("Depth")
    axes[2].imshow(raw_observation["semantic"], cmap=mcolors.ListedColormap(["#222222", "#4c78a8", "#f2c14e"]))
    axes[2].set_title("Semantic IDs")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "habitat_like_inputs.png", dpi=180)
    plt.close(fig)


def _plot_phase22_overlay(bev: BEVMemory, memory: SemanticSpatialMemory, retrieval) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    state = bev.occupancy_state().T
    cmap = mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#333333"])
    ax.imshow(state, origin="lower", cmap=cmap, vmin=0, vmax=2, alpha=0.78)
    ax.set_title("Phase 2.2 Habitat Adapter Contract -> BEV")
    ax.set_xlim(-0.5, bev.grid_size[0] - 0.5)
    ax.set_ylim(-0.5, bev.grid_size[1] - 0.5)
    ax.set_xticks(range(0, bev.grid_size[0], 2))
    ax.set_yticks(range(0, bev.grid_size[1], 2))
    ax.grid(color="#dddddd", linewidth=0.5)
    if bev.trajectory:
        xs, ys = zip(*bev.trajectory)
        ax.plot(xs, ys, color="#1f77b4", marker="*", markersize=10)
    for item in memory.items.values():
        x, y = item.bev_position
        ax.scatter(x, y, s=360, facecolors="none", edgecolors="#d62728", linewidth=2.0)
        ax.text(x + 0.2, y + 0.2, f"{item.semantic_label}:{item.id}", color="#d62728", fontsize=8)
    if retrieval:
        top = retrieval[0].item
        ax.annotate(
            "sofa top-1",
            xy=top.bev_position,
            xytext=(top.bev_position[0] + 1.2, top.bev_position[1] + 1.0),
            arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 2},
            color="#1f77b4",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "habitat_adapter_bev_overlay.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
