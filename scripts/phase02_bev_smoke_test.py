from __future__ import annotations

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

from bev_memory import BEVMemory, make_phase02_synthetic_observations
from rsc_nav_memory import SemanticSpatialMemory


OUT_DIR = ROOT / "outputs" / "phase02"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bev = BEVMemory(grid_size=(16, 16), resolution=1.0)
    long_term_memory = SemanticSpatialMemory(grid_size=(16, 16), scene_id="phase02_scene")
    observations = make_phase02_synthetic_observations()
    sequence_snapshots = []

    for obs in observations:
        projected = bev.update_from_observation(obs)
        for evidence in projected:
            long_term_memory.observe(
                label=evidence.label,
                position=evidence.grid_coord,
                confidence=evidence.confidence,
                time=evidence.time,
                source_view_id=evidence.source_view_id,
            )
        sequence_snapshots.append(
            {
                "time": obs.time,
                "view_id": obs.view_id,
                "pose": obs.pose.__dict__,
                "projected_semantic_evidence": [e.to_dict() for e in projected],
                "bev_snapshot": bev.snapshot(),
                "long_term_memory": long_term_memory.snapshot(),
            }
        )

    retrieval = long_term_memory.retrieve("sofa", current_position=(1, 1), top_k=3)
    log = {
        "phase": "phase02_synthetic_bev_memory",
        "pipeline": "synthetic observation + pose -> BEV projection -> occupancy/explored/semantic map -> long-term memory",
        "bev_snapshot": bev.snapshot(),
        "semantic_spatial_memory": long_term_memory.snapshot(),
        "sofa_retrieval": [result.to_dict() for result in retrieval],
        "sequence": sequence_snapshots,
    }
    (OUT_DIR / "phase02_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_occupancy(bev)
    _plot_semantic(bev)
    _plot_overlay(bev, long_term_memory, retrieval)
    _plot_sequence(sequence_snapshots)

    print(f"Wrote {OUT_DIR / 'phase02_log.json'}")
    print(f"Wrote {OUT_DIR / 'bev_occupancy.png'}")
    print(f"Wrote {OUT_DIR / 'bev_semantic.png'}")
    print(f"Wrote {OUT_DIR / 'bev_memory_overlay.png'}")
    print(f"Wrote {OUT_DIR / 'bev_update_sequence.png'}")
    print("Explored cells:", int(bev.explored.sum()))
    print("Occupied cells:", int((bev.occupancy_logodds > 0.2).sum()))
    print("Semantic cells:", int((bev.semantic_confidence > 0).sum()))
    print("Sofa retrieval top-1:", retrieval[0].item.id, retrieval[0].item.bev_position, retrieval[0].score)


def _plot_occupancy(bev: BEVMemory) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    state = bev.occupancy_state().T
    cmap = mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#333333"])
    ax.imshow(state, origin="lower", cmap=cmap, vmin=0, vmax=2)
    _draw_grid(ax, bev, "BEV Occupancy / Explored Map")
    _draw_trajectory(ax, bev)
    _save(fig, "bev_occupancy.png")


def _plot_semantic(bev: BEVMemory) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(bev.semantic_confidence.T, origin="lower", cmap="YlGn", vmin=0, vmax=1)
    _draw_grid(ax, bev, "BEV Semantic Evidence Confidence")
    _draw_semantic_labels(ax, bev)
    _draw_trajectory(ax, bev)
    _save(fig, "bev_semantic.png")


def _plot_overlay(bev: BEVMemory, memory: SemanticSpatialMemory, retrieval) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    state = bev.occupancy_state().T
    cmap = mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#333333"])
    ax.imshow(state, origin="lower", cmap=cmap, vmin=0, vmax=2, alpha=0.78)
    explored = np.ma.masked_where(~bev.explored.T, bev.explored.T)
    ax.imshow(explored, origin="lower", cmap=mcolors.ListedColormap(["#bcd7ff"]), alpha=0.22)
    _draw_grid(ax, bev, "BEV Memory Overlay: occupancy + explored + semantic memory")
    _draw_semantic_labels(ax, bev)
    _draw_trajectory(ax, bev)
    _draw_memory_items(ax, memory)
    if retrieval:
        top = retrieval[0].item
        ax.annotate(
            "retrieval top-1",
            xy=top.bev_position,
            xytext=(top.bev_position[0] + 1.4, top.bev_position[1] + 1.2),
            arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 2},
            color="#1f77b4",
            fontsize=9,
        )
    _save(fig, "bev_memory_overlay.png")


def _plot_sequence(sequence_snapshots: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(sequence_snapshots), figsize=(16, 4.5))
    for ax, snapshot in zip(axes, sequence_snapshots):
        grid_size = tuple(snapshot["bev_snapshot"]["grid_size"])
        explored = np.zeros(grid_size, dtype=bool)
        semantic = np.zeros(grid_size, dtype=np.float32)
        trajectory = [tuple(cell) for cell in snapshot["bev_snapshot"]["trajectory"]]
        for evidence in snapshot["bev_snapshot"]["semantic_evidence"]:
            x, y = evidence["grid_coord"]
            semantic[x, y] = max(semantic[x, y], evidence["confidence"])
        for x, y in trajectory:
            explored[x, y] = True
        ax.imshow(explored.T, origin="lower", cmap=mcolors.ListedColormap(["#f2f2f2", "#bcd7ff"]), vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(semantic.T == 0, semantic.T), origin="lower", cmap="YlGn", vmin=0, vmax=1)
        ax.set_title(f"t={snapshot['time']} | {snapshot['view_id']}")
        ax.set_xlim(-0.5, grid_size[0] - 0.5)
        ax.set_ylim(-0.5, grid_size[1] - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for evidence in snapshot["bev_snapshot"]["semantic_evidence"]:
            x, y = evidence["grid_coord"]
            ax.text(x + 0.2, y + 0.2, evidence["label"], fontsize=8)
        if trajectory:
            xs, ys = zip(*trajectory)
            ax.plot(xs, ys, color="#1f77b4", linewidth=1.8, marker="*", markersize=7)
    fig.suptitle("BEV Update Sequence: explored space and semantic evidence over time")
    _save(fig, "bev_update_sequence.png")


def _draw_grid(ax, bev: BEVMemory, title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(-0.5, bev.grid_size[0] - 0.5)
    ax.set_ylim(-0.5, bev.grid_size[1] - 0.5)
    ax.set_xticks(range(bev.grid_size[0]))
    ax.set_yticks(range(bev.grid_size[1]))
    ax.grid(color="#dddddd", linewidth=0.5)


def _draw_trajectory(ax, bev: BEVMemory) -> None:
    if not bev.trajectory:
        return
    xs, ys = zip(*bev.trajectory)
    ax.plot(xs, ys, color="#1f77b4", linewidth=2.0, marker="*", markersize=10, label="agent trajectory")
    ax.legend(loc="upper left", fontsize=8)


def _draw_semantic_labels(ax, bev: BEVMemory) -> None:
    for x in range(bev.grid_size[0]):
        for y in range(bev.grid_size[1]):
            label = bev.semantic_label[x, y]
            conf = float(bev.semantic_confidence[x, y])
            if label and conf > 0:
                ax.scatter(x, y, s=260 * conf, color="#2ca02c", edgecolor="black", marker="s", alpha=0.82)
                ax.text(
                    x + 0.22,
                    y + 0.20,
                    f"{label}\n{conf:.2f}",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
                )


def _draw_memory_items(ax, memory: SemanticSpatialMemory) -> None:
    for item in memory.items.values():
        x, y = item.bev_position
        ax.scatter(x, y, s=420, facecolors="none", edgecolors="#d62728", linewidth=2.0)
        ax.text(x - 0.45, y - 0.75, f"mem:{item.id}", color="#d62728", fontsize=8)


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
