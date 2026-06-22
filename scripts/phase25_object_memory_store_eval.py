from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_memory_store import ObjectMemoryStore, build_store_from_semantic_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.5 object memory store evaluation.")
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "phase25_object_memory"))
    parser.add_argument("--freshness-tau-steps", type=float, default=20.0)
    parser.add_argument("--decay-extra-steps", type=int, default=30)
    parser.add_argument("--current-x", type=float, default=0.0)
    parser.add_argument("--current-z", type=float, default=0.0)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    semantic_report = report.get("semantic_report")
    if not semantic_report or not semantic_report.get("tracks"):
        raise RuntimeError(f"No semantic tracks found in {metrics_path}")

    scene_id = str(report.get("scene") or metrics_path.parent.name)
    current_step = _max_last_seen(semantic_report["tracks"])
    store = build_store_from_semantic_report(
        semantic_report=semantic_report,
        scene_id=scene_id,
        freshness_tau_steps=args.freshness_tau_steps,
    )

    memory_path = out_dir / "object_memory.json"
    store.save(memory_path)
    loaded = ObjectMemoryStore.load(memory_path)
    reload_equal = loaded.to_dict() == store.to_dict()

    retrieval = {
        category: loaded.retrieve(category, current_xz=(args.current_x, args.current_z), top_k=5)
        for category in sorted({item.category for item in loaded.items.values()})
    }

    decayed = ObjectMemoryStore.load(memory_path)
    decayed.decay(current_step=current_step + args.decay_extra_steps)
    decayed_path = out_dir / "object_memory_decayed.json"
    decayed.save(decayed_path)

    replay = ObjectMemoryStore.load(memory_path)
    update_result = replay.update_from_tracks(semantic_report["tracks"], current_step=current_step + 1)
    replay_path = out_dir / "object_memory_replayed.json"
    replay.save(replay_path)

    eval_report = {
        "phase": "phase25_object_memory_store_eval",
        "source_metrics_json": str(metrics_path),
        "scene_id": scene_id,
        "freshness_tau_steps": args.freshness_tau_steps,
        "current_step": current_step,
        "decay_extra_steps": args.decay_extra_steps,
        "initial_summary": store.summary(),
        "reload_equal": reload_equal,
        "retrieval": retrieval,
        "decayed_summary": decayed.summary(),
        "replay_update_result": update_result,
        "replayed_summary": replay.summary(),
        "outputs": {
            "object_memory": str(memory_path),
            "object_memory_decayed": str(decayed_path),
            "object_memory_replayed": str(replay_path),
            "object_memory_plot": str(out_dir / "object_memory_plot.png"),
        },
    }
    (out_dir / "object_memory_eval.json").write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    _plot_memory(store, out_dir / "object_memory_plot.png")
    _write_summary_html(out_dir, eval_report)
    print(json.dumps(eval_report, indent=2))


def _max_last_seen(tracks: list[dict]) -> int:
    return max((int(track.get("last_seen_step", 0)) for track in tracks), default=0)


def _plot_memory(store: ObjectMemoryStore, path: Path) -> None:
    colors = {
        "wall": "#2f2f2f",
        "door": "#1f77b4",
        "table": "#f28e2b",
        "chair": "#59a14f",
    }
    fig, ax = plt.subplots(figsize=(7, 7))
    for item in store.items.values():
        color = colors.get(item.category, "#9467bd")
        size = 30 + 120 * item.confidence
        ax.scatter(item.centroid_xz[0], item.centroid_xz[1], s=size, c=color, alpha=max(0.25, item.freshness))
        ax.text(item.centroid_xz[0], item.centroid_xz[1], f"{item.category}:{item.semantic_id}", fontsize=7)
    ax.set_title("Object Memory Store")
    ax.set_xlabel("world x")
    ax.set_ylabel("world z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary_html(out_dir: Path, report: dict) -> None:
    rows = "\n".join(
        f"<tr><td>{key}</td><td>{json.dumps(value, ensure_ascii=False)}</td></tr>"
        for key, value in report.items()
        if key not in {"retrieval", "outputs"}
    )
    retrieval_rows = "\n".join(
        f"<tr><td>{category}</td><td><pre>{json.dumps(results, ensure_ascii=False, indent=2)}</pre></td></tr>"
        for category, results in report["retrieval"].items()
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase 2.5 Object Memory</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
td {{ border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }}
table {{ border-collapse: collapse; }}
img {{ max-width: 720px; border: 1px solid #ccc; }}
pre {{ margin: 0; white-space: pre-wrap; }}
</style></head><body>
<h1>Phase 2.5 Object Memory Store</h1>
<img src="object_memory_plot.png">
<h2>Summary</h2><table>{rows}</table>
<h2>Retrieval</h2><table>{retrieval_rows}</table>
</body></html>"""
    (out_dir / "summary.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
