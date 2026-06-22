from __future__ import annotations

import argparse
import base64
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ACTIONS = (
    ["move_forward"] * 6
    + ["turn_left"] * 2
    + ["move_forward"] * 6
    + ["turn_right"] * 3
    + ["move_forward"] * 6
    + ["turn_right"] * 2
    + ["move_forward"] * 6
)
DEFAULT_PATH_STEPS = 36

IMAGE_KEYS = {
    "rgb_jpeg": ("rgb", ".jpg"),
    "depth_png": ("depth", ".png"),
    "bev_png": ("bev", ".png"),
    "semantic_png": ("semantic_bev", ".png"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.7 live Habitat control memory evaluator.")
    parser.add_argument("--url", default="http://127.0.0.1:43901")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--actions", help="Comma-separated action sequence. Defaults to the Phase 2.6 sweep.")
    parser.add_argument("--trajectory-mode", choices=("path", "actions"), default="path")
    parser.add_argument("--path-steps", type=int, default=DEFAULT_PATH_STEPS)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--sleep-sec", type=float, default=0.03)
    parser.add_argument("--target-classes", default="wall,door,table,chair")
    parser.add_argument("--min-oracle-free-iou", type=float, default=0.2)
    parser.add_argument("--min-oracle-occupied-f1", type=float, default=0.05)
    parser.add_argument("--max-mean-step-drift-m", type=float, default=0.45)
    parser.add_argument("--max-tail-drift-m", type=float, default=0.8)
    parser.add_argument("--stability-window", type=int, default=6)
    parser.add_argument("--min-final-items", type=int, default=4)
    parser.add_argument("--min-active-items", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actions = _parse_actions(args.actions, args.trajectory_mode, args.path_steps)
    target_classes = [item.strip().lower() for item in args.target_classes.split(",") if item.strip()]

    base_url = args.url.rstrip("/")
    states: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    if args.reset:
        state = _post_json(base_url, "/api/reset", {})
    else:
        state = _get_json(base_url, "/api/state")
    states.append(_compact_state(state))
    _save_checkpoint_images(state, out_dir, prefix=f"step_{int(state['step']):04d}")
    checkpoints.append({"step": int(state["step"]), "reason": "initial"})

    for idx, action in enumerate(actions, start=1):
        state = _post_json(base_url, "/api/action", {"action": action})
        compact = _compact_state(state)
        compact["action"] = action
        states.append(compact)
        if idx % args.checkpoint_interval == 0 or idx == len(actions):
            _save_checkpoint_images(state, out_dir, prefix=f"step_{int(state['step']):04d}")
            checkpoints.append({"step": int(state["step"]), "reason": f"action_{idx}"})
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    final_state = _post_json(base_url, "/api/save_memory", {})
    states.append(_compact_state(final_state))
    _save_checkpoint_images(final_state, out_dir, prefix=f"step_{int(final_state['step']):04d}_final")
    checkpoints.append({"step": int(final_state["step"]), "reason": "final_save"})

    object_history = _object_history(states)
    metrics = _compute_metrics(
        states=states,
        object_history=object_history,
        target_classes=target_classes,
        min_oracle_free_iou=args.min_oracle_free_iou,
        min_oracle_occupied_f1=args.min_oracle_occupied_f1,
        max_mean_step_drift_m=args.max_mean_step_drift_m,
        max_tail_drift_m=args.max_tail_drift_m,
        stability_window=args.stability_window,
        min_final_items=args.min_final_items,
        min_active_items=args.min_active_items,
    )

    (out_dir / "states_compact.json").write_text(json.dumps(states, indent=2), encoding="utf-8")
    (out_dir / "object_history.json").write_text(json.dumps(object_history, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "live_eval_summary.json").write_text(
        json.dumps(
            {
                "url": base_url,
                "out_dir": str(out_dir),
                "num_actions": len(actions),
                "checkpoints": checkpoints,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_summary_html(out_dir, base_url, metrics, checkpoints)
    print(json.dumps(metrics, indent=2))


def _parse_actions(actions: str | None, trajectory_mode: str, path_steps: int) -> list[str]:
    if actions is None:
        if trajectory_mode == "path":
            return ["path_step"] * max(1, int(path_steps))
        return list(DEFAULT_ACTIONS)
    return [item.strip() for item in actions.split(",") if item.strip()]


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


def _object_history(states: list[dict]) -> dict[str, list[dict]]:
    history: dict[str, list[dict]] = {}
    for state in states:
        step = int(state.get("step", 0))
        for item in state.get("memory_items", []):
            item_id = str(item["id"])
            entry = {
                "step": step,
                "category": item.get("category"),
                "centroid_xz": item.get("centroid_xz"),
                "confidence": item.get("confidence"),
                "freshness": item.get("freshness"),
                "status": item.get("status"),
                "last_seen_step": item.get("last_seen_step"),
            }
            history.setdefault(item_id, []).append(entry)
    return history


def _compute_metrics(
    states: list[dict],
    object_history: dict[str, list[dict]],
    target_classes: list[str],
    min_oracle_free_iou: float,
    min_oracle_occupied_f1: float,
    max_mean_step_drift_m: float,
    max_tail_drift_m: float,
    stability_window: int,
    min_final_items: int,
    min_active_items: int,
) -> dict:
    final = states[-1]
    final_memory = final.get("memory", {})
    final_semantic = final.get("semantic", {})
    final_oracle = final.get("geometry_oracle", {})
    final_per_class = final_memory.get("per_class", {})
    class_coverage = {category: int(final_per_class.get(category, 0)) for category in target_classes}

    stability_rows = []
    step_drifts = []
    total_drifts = []
    tail_drifts = []
    confidence_values = []
    freshness_values = []
    for item_id, observations in sorted(object_history.items()):
        centroids = [
            tuple(float(value) for value in obs["centroid_xz"])
            for obs in observations
            if obs.get("centroid_xz") and len(obs["centroid_xz"]) >= 2
        ]
        if len(centroids) < 2:
            continue
        item_step_drifts = [_distance(prev, cur) for prev, cur in zip(centroids[:-1], centroids[1:])]
        total_drift = max(_distance(centroids[0], centroid) for centroid in centroids[1:])
        tail_centroids = centroids[-max(2, int(stability_window)):]
        tail_drift = max(_distance(tail_centroids[0], centroid) for centroid in tail_centroids[1:])
        step_drifts.extend(item_step_drifts)
        total_drifts.append(total_drift)
        tail_drifts.append(tail_drift)
        confidence_values.extend(float(obs["confidence"]) for obs in observations if obs.get("confidence") is not None)
        freshness_values.extend(float(obs["freshness"]) for obs in observations if obs.get("freshness") is not None)
        stability_rows.append(
            {
                "id": item_id,
                "category": observations[-1].get("category"),
                "num_observations": len(observations),
                "mean_step_drift_m": _mean(item_step_drifts),
                "max_step_drift_m": max(item_step_drifts) if item_step_drifts else 0.0,
                "total_drift_m": total_drift,
                "tail_drift_m": tail_drift,
                "final_status": observations[-1].get("status"),
            }
        )

    mean_step_drift = _mean(step_drifts)
    max_total_drift = max(total_drifts) if total_drifts else 0.0
    max_tail_drift = max(tail_drifts) if tail_drifts else 0.0
    final_items = int(final_memory.get("num_items", 0))
    active_items = int(final_memory.get("active_items", 0))
    covered_all_classes = all(count > 0 for count in class_coverage.values())
    bev_nonempty = int(final.get("bev", {}).get("num_explored_cells", 0)) > 0 and int(final.get("bev", {}).get("num_occupied_cells", 0)) > 0
    oracle_enabled = bool(final_oracle.get("enabled"))
    oracle_free_iou = float(final_oracle.get("free_iou_observed", 0.0))
    oracle_occupied_f1 = float(final_oracle.get("occupied_f1_observed", 0.0))
    geometry_ok = (
        bev_nonempty
        and oracle_enabled
        and oracle_free_iou >= min_oracle_free_iou
        and oracle_occupied_f1 >= min_oracle_occupied_f1
    )
    semantic_ok = int(final_semantic.get("observed_target_instances", 0)) > 0 and int(final_semantic.get("semantic_cells", 0)) > 0
    stability_ok = bool(stability_rows) and mean_step_drift <= max_mean_step_drift_m and max_tail_drift <= max_tail_drift_m
    memory_ok = final_items >= min_final_items and active_items >= min_active_items
    passed = geometry_ok and semantic_ok and covered_all_classes and memory_ok and stability_ok

    return {
        "passed": passed,
        "criteria": {
            "geometry_ok": geometry_ok,
            "bev_nonempty": bev_nonempty,
            "oracle_enabled": oracle_enabled,
            "min_oracle_free_iou": min_oracle_free_iou,
            "min_oracle_occupied_f1": min_oracle_occupied_f1,
            "semantic_ok": semantic_ok,
            "covered_all_classes": covered_all_classes,
            "memory_ok": memory_ok,
            "stability_ok": stability_ok,
            "max_mean_step_drift_m": max_mean_step_drift_m,
            "max_tail_drift_m": max_tail_drift_m,
            "stability_window": int(stability_window),
            "min_final_items": min_final_items,
            "min_active_items": min_active_items,
        },
        "final_step": int(final.get("step", 0)),
        "class_coverage": class_coverage,
        "final_memory": final_memory,
        "final_semantic": final_semantic,
        "final_geometry_oracle": final_oracle,
        "final_bev": final.get("bev", {}),
        "object_stability": {
            "tracked_items": len(stability_rows),
            "mean_step_drift_m": mean_step_drift,
            "max_step_drift_m": max(step_drifts) if step_drifts else 0.0,
            "max_total_drift_m": max_total_drift,
            "max_tail_drift_m": max_tail_drift,
            "mean_confidence": _mean(confidence_values),
            "mean_freshness": _mean(freshness_values),
            "items": stability_rows,
        },
    }


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_summary_html(out_dir: Path, url: str, metrics: dict, checkpoints: list[dict]) -> None:
    image_rows = []
    for checkpoint in checkpoints:
        step = int(checkpoint["step"])
        for stem in (f"step_{step:04d}", f"step_{step:04d}_final"):
            semantic = out_dir / f"{stem}_semantic_bev.png"
            bev = out_dir / f"{stem}_bev.png"
            if semantic.exists() or bev.exists():
                image_rows.append(
                    f"<tr><td>{step}</td><td>{checkpoint['reason']}</td>"
                    f"<td>{_img_tag(bev)}</td><td>{_img_tag(semantic)}</td></tr>"
                )
                break
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Phase 2.7 Live Memory Eval</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;line-height:1.4}}
code,pre{{background:#f4f4f4;padding:2px 4px}}
img{{width:260px;border:1px solid #ccc}}
td,th{{vertical-align:top;padding:8px;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse}}
</style></head>
<body>
<h1>Phase 2.7 Live Memory Eval</h1>
<p>URL: <code>{url}</code></p>
<p>Passed: <strong>{metrics['passed']}</strong></p>
<pre>{json.dumps(metrics, indent=2)}</pre>
<table>
<tr><th>Step</th><th>Reason</th><th>BEV</th><th>Semantic BEV</th></tr>
{''.join(image_rows)}
</table>
</body></html>
"""
    (out_dir / "summary.html").write_text(html, encoding="utf-8")


def _img_tag(path: Path) -> str:
    return f'<img src="{path.name}">' if path.exists() else ""


if __name__ == "__main__":
    main()
