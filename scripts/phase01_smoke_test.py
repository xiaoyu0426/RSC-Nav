from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsc_nav_memory import SemanticSpatialMemory


OUT_DIR = ROOT / "outputs" / "phase01"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stale_memory = SemanticSpatialMemory()
    adaptive_memory = SemanticSpatialMemory()

    for memory in (stale_memory, adaptive_memory):
        memory.observe("sofa", (2, 2), 0.92, time=1, source_view_id="view_001")
        memory.observe("table", (5, 3), 0.85, time=2, source_view_id="view_002")
        memory.observe("lamp", (8, 8), 0.70, time=3, source_view_id="view_003")

    initial_snapshot = adaptive_memory.snapshot()

    # Scene variant: sofa moved from (2, 2) to (8, 7). Stale memory does not update.
    adaptive_memory.weaken_expected_visible("sofa", (2, 2), time=4)
    adaptive_memory.relocate(
        "sofa",
        old_position=(2, 2),
        new_position=(8, 7),
        confidence=0.90,
        time=5,
        source_view_id="view_010",
    )
    adaptive_memory.observe("sofa", (8, 7), 0.95, time=6, source_view_id="view_011")

    stale_results = stale_memory.retrieve("sofa", current_position=(1, 1), top_k=3)
    adaptive_results = adaptive_memory.retrieve("sofa", current_position=(1, 1), top_k=3)

    log = {
        "episode_id": "phase01_synthetic_repeated_use_001",
        "claim": "write -> retrieve -> perturb -> stale retrieval -> reconfigured retrieval",
        "initial_snapshot": initial_snapshot,
        "stale_snapshot": stale_memory.snapshot(),
        "adaptive_snapshot": adaptive_memory.snapshot(),
        "stale_retrieval": [result.to_dict() for result in stale_results],
        "adaptive_retrieval": [result.to_dict() for result in adaptive_results],
        "expected_result": {
            "stale_top1": "old sofa at [2, 2]",
            "adaptive_top1": "relocated sofa at [8, 7]",
        },
    }
    (OUT_DIR / "phase01_smoke_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_smoke_test(stale_memory, adaptive_memory, stale_results, adaptive_results)
    print(f"Wrote {OUT_DIR / 'phase01_smoke_log.json'}")
    print(f"Wrote {OUT_DIR / 'phase01_smoke_visualization.png'}")
    print("Stale top-1:", stale_results[0].item.id, stale_results[0].item.bev_position, stale_results[0].score)
    print(
        "Adaptive top-1:",
        adaptive_results[0].item.id,
        adaptive_results[0].item.bev_position,
        adaptive_results[0].score,
    )


def _plot_smoke_test(stale_memory, adaptive_memory, stale_results, adaptive_results) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 11),
        gridspec_kw={"height_ratios": [1.05, 0.95], "hspace": 0.42, "wspace": 0.22},
    )
    fig.suptitle(
        "RSC-Nav Phase 0+1 Smoke Test: Long-Term Memory Reuse and Adaptive Update",
        fontsize=14,
    )

    _draw_memory(
        axes[0, 0],
        stale_memory,
        title="Carried-stale memory\n(old sofa remains active)",
        retrieval=stale_results,
    )
    _draw_memory(
        axes[0, 1],
        adaptive_memory,
        title="Carried-reconfigured memory\n(old sofa downgraded, new sofa active)",
        retrieval=adaptive_results,
    )
    _draw_retrieval_bar(axes[1, 0], stale_results, "Goal query: sofa | stale retrieval")
    _draw_retrieval_bar(axes[1, 1], adaptive_results, "Goal query: sofa | reconfigured retrieval")

    fig.savefig(OUT_DIR / "phase01_smoke_visualization.png", dpi=180)
    plt.close(fig)


def _draw_memory(ax, memory: SemanticSpatialMemory, title: str, retrieval) -> None:
    grid = np.zeros(memory.grid_size)
    ax.imshow(grid.T, origin="lower", cmap="Greys", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlim(-0.5, memory.grid_size[0] - 0.5)
    ax.set_ylim(-0.5, memory.grid_size[1] - 0.5)
    ax.set_xticks(range(memory.grid_size[0]))
    ax.set_yticks(range(memory.grid_size[1]))
    ax.grid(color="#dddddd", linewidth=0.6)

    colors = {
        "active": "#2ca02c",
        "stale": "#ff7f0e",
        "missing": "#d62728",
        "relocated": "#9467bd",
    }
    markers = {
        "sofa": "s",
        "table": "o",
        "lamp": "^",
    }
    for item in memory.items.values():
        x, y = item.bev_position
        ax.scatter(
            x,
            y,
            s=360 * max(item.confidence, 0.2),
            marker=markers.get(item.semantic_label, "o"),
            color=colors.get(item.status, "#7f7f7f"),
            edgecolor="black",
            linewidth=1.0,
            alpha=0.88,
        )
        dx, dy = _label_offset(item.semantic_label, item.status)
        ax.text(
            x + dx,
            y + dy,
            f"{item.semantic_label}\n{item.status}\nc={item.confidence:.2f}",
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )

    if retrieval:
        top = retrieval[0].item
        ax.annotate(
            "top-1",
            xy=top.bev_position,
            xytext=(top.bev_position[0] + 1.5, top.bev_position[1] + 1.5),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#1f77b4"},
            color="#1f77b4",
            fontsize=10,
        )

    ax.scatter(1, 1, marker="*", s=180, color="#1f77b4", edgecolor="black")
    ax.text(1.2, 1.2, "agent", fontsize=8)


def _draw_retrieval_bar(ax, results, title: str) -> None:
    labels = [f"{r.item.semantic_label}@{r.item.bev_position}\n{r.item.status}" for r in results]
    scores = [r.score for r in results]
    colors = ["#1f77b4"] + ["#bbbbbb"] * (len(results) - 1)
    ax.bar(range(len(results)), scores, color=colors)
    ax.set_ylim(0, max(scores) + 1.0)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("retrieval score")
    ax.set_title(title)
    for idx, result in enumerate(results):
        parts = result.score_parts
        ax.text(
            idx,
            scores[idx] + 0.08,
            f"sem={parts['semantic_match']:.1f}\nconf={parts['confidence']:.2f}\nfresh={parts['freshness']:.2f}\npen={parts['status_penalty']:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _label_offset(label: str, status: str) -> tuple[float, float]:
    if label == "lamp":
        return 0.20, 0.45
    if status == "relocated":
        return 0.25, -0.75
    if label == "sofa":
        return 0.25, 0.10
    return 0.20, 0.20


if __name__ == "__main__":
    main()
