from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


IMAGE_KEYS = {
    "rgb_jpeg": ("rgb", ".jpg"),
    "depth_png": ("depth", ".png"),
    "bev_png": ("bev", ".png"),
    "semantic_png": ("semantic_bev", ".png"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.9 live object-memory reuse evaluator.")
    parser.add_argument("--url", default="http://127.0.0.1:43901")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--initial-path-steps", type=int, default=36)
    parser.add_argument("--replay-path-steps", type=int, default=12)
    parser.add_argument("--checkpoint-interval", type=int, default=6)
    parser.add_argument("--sleep-sec", type=float, default=0.03)
    parser.add_argument("--min-retained-ratio", type=float, default=0.95)
    parser.add_argument("--max-duplicate-new-items", type=int, default=2)
    parser.add_argument("--min-updated-items", type=int, default=1)
    parser.add_argument("--min-replay-active-items", type=int, default=1)
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    state = _post_json(base_url, "/api/reset", {})
    _record("initial_reset", state, timeline, out_dir, checkpoints, save_images=True)

    for idx in range(1, args.initial_path_steps + 1):
        state = _post_json(base_url, "/api/action", {"action": "path_step"})
        save_images = idx % args.checkpoint_interval == 0 or idx == args.initial_path_steps
        _record(f"initial_path_{idx}", state, timeline, out_dir, checkpoints, save_images=save_images)
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    saved_state = _post_json(base_url, "/api/save_memory", {})
    _record("saved_memory", saved_state, timeline, out_dir, checkpoints, save_images=True)

    reset_state = _post_json(base_url, "/api/reset", {})
    _record("post_reset_before_load", reset_state, timeline, out_dir, checkpoints, save_images=True)

    loaded_state = _post_json(base_url, "/api/load_memory", {})
    _record("loaded_memory", loaded_state, timeline, out_dir, checkpoints, save_images=True)

    for idx in range(1, args.replay_path_steps + 1):
        state = _post_json(base_url, "/api/action", {"action": "path_step"})
        save_images = idx % args.checkpoint_interval == 0 or idx == args.replay_path_steps
        _record(f"replay_path_{idx}", state, timeline, out_dir, checkpoints, save_images=save_images)
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    replay_saved_state = _post_json(base_url, "/api/save_memory", {})
    _record("replay_saved_memory", replay_saved_state, timeline, out_dir, checkpoints, save_images=True)

    metrics = _compute_metrics(
        saved_state=saved_state,
        reset_state=reset_state,
        loaded_state=loaded_state,
        replay_state=replay_saved_state,
        min_retained_ratio=args.min_retained_ratio,
        max_duplicate_new_items=args.max_duplicate_new_items,
        min_updated_items=args.min_updated_items,
        min_replay_active_items=args.min_replay_active_items,
    )
    summary = {
        "url": base_url,
        "out_dir": str(out_dir),
        "initial_path_steps": args.initial_path_steps,
        "replay_path_steps": args.replay_path_steps,
        "checkpoints": checkpoints,
        "metrics": metrics,
    }
    (out_dir / "timeline_compact.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "live_memory_reuse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_html(out_dir, summary)
    print(json.dumps(metrics, indent=2))


def _record(
    label: str,
    state: dict,
    timeline: list[dict],
    out_dir: Path,
    checkpoints: list[dict],
    save_images: bool,
) -> None:
    compact = _compact_state(state)
    compact["label"] = label
    timeline.append(compact)
    if save_images:
        prefix = f"{len(timeline) - 1:03d}_{label}_step_{int(state['step']):04d}_mem_{int(state.get('memory_step', state['step'])):04d}"
        _save_checkpoint_images(state, out_dir, prefix=prefix)
        checkpoints.append(
            {
                "label": label,
                "step": int(state["step"]),
                "memory_step": int(state.get("memory_step", state["step"])),
                "prefix": prefix,
            }
        )


def _compute_metrics(
    saved_state: dict,
    reset_state: dict,
    loaded_state: dict,
    replay_state: dict,
    min_retained_ratio: float,
    max_duplicate_new_items: int,
    min_updated_items: int,
    min_replay_active_items: int,
) -> dict:
    saved_ids = _item_ids(saved_state)
    reset_ids = _item_ids(reset_state)
    loaded_ids = _item_ids(loaded_state)
    replay_ids = _item_ids(replay_state)

    retained_after_load = saved_ids & loaded_ids
    retained_after_replay = saved_ids & replay_ids
    new_after_replay = replay_ids - saved_ids
    lost_after_replay = saved_ids - replay_ids

    loaded_ratio = _safe_div(len(retained_after_load), len(saved_ids))
    replay_retained_ratio = _safe_div(len(retained_after_replay), len(saved_ids))
    updated_ids = _updated_ids(saved_state, replay_state, saved_ids)

    reset_did_not_keep_full_memory = len(reset_ids & saved_ids) < len(saved_ids)
    load_retained_ok = loaded_ratio >= min_retained_ratio
    replay_retained_ok = replay_retained_ratio >= min_retained_ratio
    duplicate_ok = len(new_after_replay) <= max_duplicate_new_items
    update_ok = len(updated_ids) >= min_updated_items
    replay_active_ok = int(replay_state.get("memory", {}).get("active_items", 0)) >= min_replay_active_items
    memory_step_monotonic = int(replay_state.get("memory_step", 0)) > int(saved_state.get("memory_step", -1))
    passed = (
        bool(saved_ids)
        and reset_did_not_keep_full_memory
        and load_retained_ok
        and replay_retained_ok
        and duplicate_ok
        and update_ok
        and replay_active_ok
        and memory_step_monotonic
    )

    return {
        "passed": passed,
        "criteria": {
            "reset_did_not_keep_full_memory": reset_did_not_keep_full_memory,
            "load_retained_ok": load_retained_ok,
            "replay_retained_ok": replay_retained_ok,
            "duplicate_ok": duplicate_ok,
            "update_ok": update_ok,
            "replay_active_ok": replay_active_ok,
            "memory_step_monotonic": memory_step_monotonic,
            "min_retained_ratio": min_retained_ratio,
            "max_duplicate_new_items": max_duplicate_new_items,
            "min_updated_items": min_updated_items,
            "min_replay_active_items": min_replay_active_items,
        },
        "saved": _memory_snapshot(saved_state),
        "reset_before_load": _memory_snapshot(reset_state),
        "loaded": _memory_snapshot(loaded_state),
        "replay": _memory_snapshot(replay_state),
        "retained_after_load": {
            "count": len(retained_after_load),
            "ratio": loaded_ratio,
        },
        "retained_after_replay": {
            "count": len(retained_after_replay),
            "ratio": replay_retained_ratio,
        },
        "new_after_replay": sorted(new_after_replay),
        "lost_after_replay": sorted(lost_after_replay),
        "updated_ids": sorted(updated_ids),
    }


def _updated_ids(saved_state: dict, replay_state: dict, candidate_ids: set[str]) -> set[str]:
    saved = _items_by_id(saved_state)
    replay = _items_by_id(replay_state)
    updated = set()
    for item_id in candidate_ids:
        before = saved.get(item_id)
        after = replay.get(item_id)
        if not before or not after:
            continue
        if int(after.get("last_seen_step", -1)) > int(before.get("last_seen_step", -1)):
            updated.add(item_id)
            continue
        if float(after.get("confidence", 0.0)) > float(before.get("confidence", 0.0)) + 1e-6:
            updated.add(item_id)
    return updated


def _memory_snapshot(state: dict) -> dict:
    memory = dict(state.get("memory", {}))
    return {
        "step": int(state.get("step", 0)),
        "memory_step": int(state.get("memory_step", state.get("step", 0))),
        "num_items": int(memory.get("num_items", 0)),
        "per_class": memory.get("per_class", {}),
        "mean_confidence": memory.get("mean_confidence", 0.0),
        "mean_freshness": memory.get("mean_freshness", 0.0),
        "active_items": int(memory.get("active_items", 0)),
        "stale_items": int(memory.get("stale_items", 0)),
        "missing_items": int(memory.get("missing_items", 0)),
        "ids": sorted(_item_ids(state)),
    }


def _item_ids(state: dict) -> set[str]:
    return set(_items_by_id(state))


def _items_by_id(state: dict) -> dict[str, dict]:
    return {str(item["id"]): item for item in state.get("memory_items", [])}


def _compact_state(state: dict) -> dict:
    return {
        key: value
        for key, value in state.items()
        if key not in {"rgb_jpeg", "depth_png", "bev_png", "semantic_png"}
    }


def _save_checkpoint_images(state: dict, out_dir: Path, prefix: str) -> None:
    for key, (label, suffix) in IMAGE_KEYS.items():
        value = state.get(key)
        if not value:
            continue
        (out_dir / f"{prefix}_{label}{suffix}").write_bytes(base64.b64decode(value))


def _get_json(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(base_url: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} failed: HTTP {exc.code}: {body}") from exc


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _write_summary_html(out_dir: Path, summary: dict) -> None:
    rows = []
    for checkpoint in summary["checkpoints"]:
        prefix = checkpoint["prefix"]
        bev = out_dir / f"{prefix}_bev.png"
        semantic = out_dir / f"{prefix}_semantic_bev.png"
        rows.append(
            f"<tr><td>{checkpoint['label']}</td><td>{checkpoint['step']}</td>"
            f"<td>{checkpoint['memory_step']}</td><td>{_img_tag(bev)}</td><td>{_img_tag(semantic)}</td></tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Phase 2.9 Live Memory Reuse Eval</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;line-height:1.4}}
code,pre{{background:#f4f4f4;padding:2px 4px}}
img{{width:260px;border:1px solid #ccc}}
td,th{{vertical-align:top;padding:8px;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse}}
</style></head>
<body>
<h1>Phase 2.9 Live Memory Reuse Eval</h1>
<p>URL: <code>{summary['url']}</code></p>
<p>Passed: <strong>{summary['metrics']['passed']}</strong></p>
<pre>{json.dumps(summary['metrics'], indent=2)}</pre>
<table>
<tr><th>Label</th><th>Episode Step</th><th>Memory Step</th><th>BEV</th><th>Semantic BEV</th></tr>
{''.join(rows)}
</table>
</body></html>
"""
    (out_dir / "summary.html").write_text(html, encoding="utf-8")


def _img_tag(path: Path) -> str:
    return f'<img src="{path.name}">' if path.exists() else ""


if __name__ == "__main__":
    main()
