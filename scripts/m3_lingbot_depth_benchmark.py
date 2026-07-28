from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dense_bev_mapper import DenseBEVMapper
from m3_vggt_geometry_eval import (
    MatrixRotation,
    _compare_bev,
    _grid_config,
    _render_bev,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LingBot-Depth on degraded Habitat RGB-D and BEV.")
    parser.add_argument("--frames-metadata", required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=/absolute/path/to/model.pt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--resolution-level", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    import torch
    from mdm.model.v2 import MDMModel

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata, frames = _load_frames(Path(args.frames_metadata), args.max_frames)
    oracle_depths = [np.load(frame["depth_npy"]).astype(np.float32) for frame in frames]
    degraded_depths = [
        _degrade_depth(depth, np.random.default_rng(args.seed + index))
        for index, depth in enumerate(oracle_depths)
    ]
    nearest_depths = [_nearest_fill(depth) for depth in degraded_depths]
    oracle_mapper = _build_mapper(metadata, frames, oracle_depths)
    degraded_mapper = _build_mapper(metadata, frames, degraded_depths)
    nearest_mapper = _build_mapper(metadata, frames, nearest_depths)
    _render_bev(oracle_mapper, out_dir / "bev_oracle.png", "Habitat oracle depth BEV")
    _render_bev(degraded_mapper, out_dir / "bev_degraded.png", "Degraded sensor depth BEV")
    _render_bev(nearest_mapper, out_dir / "bev_nearest_fill.png", "Nearest-fill depth BEV")

    raw_metrics = _depth_metrics(oracle_depths, degraded_depths, observed_only=True)
    nearest_metrics = _depth_metrics(oracle_depths, nearest_depths)
    results = []
    predicted_by_name: dict[str, list[np.ndarray]] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_spec in args.model:
        name, checkpoint = model_spec.split("=", 1)
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        model = MDMModel.from_pretrained(checkpoint_path).to(device).eval()
        load_seconds = time.perf_counter() - load_started
        predicted: list[np.ndarray] = []
        inference_seconds = 0.0
        for frame, degraded in zip(frames, degraded_depths):
            rgb = cv2.cvtColor(cv2.imread(frame["rgb_path"]), cv2.COLOR_BGR2RGB)
            height, width = degraded.shape
            image_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            depth_t = torch.from_numpy(degraded).unsqueeze(0).to(device)
            intrinsics = _normalized_intrinsics(width, height, float(metadata.get("hfov_deg", 90.0)))
            intrinsics_t = torch.from_numpy(intrinsics).unsqueeze(0).to(device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model.infer(
                image_t,
                depth_in=depth_t,
                intrinsics=intrinsics_t,
                resolution_level=args.resolution_level,
                apply_mask=False,
                use_fp16=True,
            )
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started
            pred = output["depth"][0].detach().float().cpu().numpy()
            pred[~np.isfinite(pred)] = 0.0
            predicted.append(pred.astype(np.float32))

        mapper = _build_mapper(metadata, frames, predicted)
        bev_path = out_dir / f"bev_{name}.png"
        _render_bev(mapper, bev_path, f"{name} refined depth BEV")
        metrics = _depth_metrics(oracle_depths, predicted)
        metrics.update(
            {
                "model": name,
                "checkpoint": str(checkpoint_path),
                "num_frames": len(frames),
                "load_seconds": float(load_seconds),
                "inference_seconds": float(inference_seconds),
                "fps": float(len(frames) / max(inference_seconds, 1e-8)),
                "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1e9),
                "bev_comparison": _compare_bev(mapper, oracle_mapper),
                "bev_image": bev_path.name,
            }
        )
        results.append(metrics)
        predicted_by_name[name] = predicted
        del model
        torch.cuda.empty_cache()

    contact_path = out_dir / "depth_refinement_contact_sheet.png"
    _render_contact_sheet(
        frames[0],
        oracle_depths[0],
        degraded_depths[0],
        nearest_depths[0],
        {name: values[0] for name, values in predicted_by_name.items()},
        contact_path,
    )
    payload = {
        "phase": "m3_lingbot_foundation_benchmark",
        "benchmark": "lingbot_depth_sensor_recovery",
        "protocol": {
            "degradation": "Gaussian noise plus random dropout and rectangular missing regions.",
            "seed": args.seed,
            "num_frames": len(frames),
            "resolution_level": args.resolution_level,
        },
        "baselines": {
            "degraded_observed_only": raw_metrics,
            "nearest_fill": {
                **nearest_metrics,
                "bev_comparison": _compare_bev(nearest_mapper, oracle_mapper),
            },
            "degraded_bev_comparison": _compare_bev(degraded_mapper, oracle_mapper),
        },
        "results": results,
    }
    (out_dir / "lingbot_depth_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(out_dir, payload, contact_path.name)
    print(json.dumps(payload, indent=2))


def _load_frames(metadata_path: Path, max_frames: int) -> tuple[dict, list[dict]]:
    metadata_path = metadata_path.expanduser().resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = [dict(frame) for frame in metadata.get("frames", [])[:max_frames]]
    base = metadata_path.parent
    for frame in frames:
        for key in ("rgb_path", "depth_npy"):
            path = Path(frame[key])
            if not path.is_file():
                candidate = base / "frames" / path.name
                if candidate.is_file():
                    path = candidate
            frame[key] = str(path)
    return metadata, frames


def _degrade_depth(depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = depth.copy()
    valid = np.isfinite(output) & (output > 0.05)
    noise = rng.normal(0.0, 0.03, size=output.shape).astype(np.float32)
    output[valid] = np.maximum(0.05, output[valid] + noise[valid])
    output[rng.random(output.shape) < 0.12] = 0.0
    height, width = output.shape
    for _ in range(10):
        rect_w = int(rng.integers(max(12, width // 16), max(20, width // 5)))
        rect_h = int(rng.integers(max(12, height // 16), max(20, height // 5)))
        x0 = int(rng.integers(0, max(1, width - rect_w)))
        y0 = int(rng.integers(0, max(1, height - rect_h)))
        output[y0 : y0 + rect_h, x0 : x0 + rect_w] = 0.0
    return output


def _nearest_fill(depth: np.ndarray) -> np.ndarray:
    invalid = ~np.isfinite(depth) | (depth <= 0.05)
    if invalid.all():
        return depth.copy()
    indices = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return depth[tuple(indices)].astype(np.float32)


def _normalized_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = fx
    intrinsics = np.array(
        [[fx / width, 0.0, (width - 1) / 2.0 / width], [0.0, fy / height, (height - 1) / 2.0 / height], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return intrinsics


def _build_mapper(metadata: dict, frames: list[dict], depths: list[np.ndarray]) -> DenseBEVMapper:
    grid_cfg = _grid_config(metadata, frames)
    mapper = DenseBEVMapper(origin_world_xz=grid_cfg["origin"], config=grid_cfg["config"])
    for frame, depth in zip(frames, depths):
        mapper.update_from_depth(
            depth=depth,
            agent_position_xyz=frame["agent_position_xyz"],
            sensor_position_xyz=frame["sensor_position_xyz"],
            sensor_rotation=MatrixRotation(np.asarray(frame["sensor_rotation_matrix"], dtype=np.float32)),
            hfov_deg=float(metadata.get("hfov_deg", 90.0)),
        )
    return mapper


def _depth_metrics(
    targets: list[np.ndarray],
    predictions: list[np.ndarray],
    observed_only: bool = False,
) -> dict:
    target_values = []
    pred_values = []
    coverage = []
    for target, prediction in zip(targets, predictions):
        valid_target = np.isfinite(target) & (target > 0.05)
        valid_prediction = np.isfinite(prediction) & (prediction > 0.05)
        coverage.append(float(np.logical_and(valid_target, valid_prediction).sum() / max(1, valid_target.sum())))
        mask = valid_target & valid_prediction if observed_only else valid_target
        pred = prediction[mask]
        if not observed_only:
            pred = np.where(np.isfinite(pred) & (pred > 0.05), pred, 10.0)
        target_values.append(target[mask])
        pred_values.append(pred)
    target = np.concatenate(target_values)
    prediction = np.concatenate(pred_values)
    error = prediction - target
    ratio = np.maximum(target / np.maximum(prediction, 1e-6), prediction / np.maximum(target, 1e-6))
    return {
        "coverage": float(np.mean(coverage)),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "abs_rel": float(np.mean(np.abs(error) / np.maximum(target, 1e-6))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def _depth_color(depth: np.ndarray, maximum: float = 7.0) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0.05)
    normalized = np.clip(depth / maximum, 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return Image.fromarray(colored)


def _render_contact_sheet(
    frame: dict,
    oracle: np.ndarray,
    degraded: np.ndarray,
    nearest: np.ndarray,
    predictions: dict[str, np.ndarray],
    output: Path,
) -> None:
    panels = [("RGB", Image.open(frame["rgb_path"]).convert("RGB")), ("Oracle depth", _depth_color(oracle))]
    panels.extend([("Degraded input", _depth_color(degraded)), ("Nearest fill", _depth_color(nearest))])
    panels.extend((name, _depth_color(depth)) for name, depth in predictions.items())
    width = 420
    height = 420
    columns = 3
    rows = int(math.ceil(len(panels) / columns))
    canvas = Image.new("RGB", (columns * width, rows * (height + 34)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = (index % columns) * width
        y = (index // columns) * (height + 34)
        canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (x, y + 34))
        draw.text((x + 10, y + 10), label, fill="#202124")
    canvas.save(output)


def _write_report(out_dir: Path, payload: dict, contact_name: str) -> None:
    nearest = payload["baselines"]["nearest_fill"]
    rows = [
        "<tr><td>Nearest fill</td>"
        f"<td>{nearest['coverage']:.3f}</td><td>{nearest['mae_m']:.3f}</td><td>{nearest['rmse_m']:.3f}</td>"
        f"<td>{nearest['abs_rel']:.3f}</td><td>{nearest['delta1']:.3f}</td>"
        f"<td>{nearest['bev_comparison']['free_iou']:.3f}</td><td>{nearest['bev_comparison']['occupied_iou']:.3f}</td>"
        "<td>-</td><td>-</td></tr>"
    ]
    bev_cards = ["<section><h2>Oracle / degraded baselines</h2><div class='grid'>"
                 "<img src='bev_oracle.png'><img src='bev_degraded.png'><img src='bev_nearest_fill.png'></div></section>"]
    for item in payload["results"]:
        rows.append(
            f"<tr><td>{html.escape(item['model'])}</td><td>{item['coverage']:.3f}</td>"
            f"<td>{item['mae_m']:.3f}</td><td>{item['rmse_m']:.3f}</td><td>{item['abs_rel']:.3f}</td>"
            f"<td>{item['delta1']:.3f}</td><td>{item['bev_comparison']['free_iou']:.3f}</td>"
            f"<td>{item['bev_comparison']['occupied_iou']:.3f}</td><td>{item['fps']:.2f}</td>"
            f"<td>{item['peak_vram_gb']:.2f} GB</td></tr>"
        )
        bev_cards.append(
            f"<section><h2>{html.escape(item['model'])} BEV</h2><img src='{html.escape(item['bev_image'])}'></section>"
        )
    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>LingBot-Depth Habitat Benchmark</title><style>
body{{font-family:Arial,sans-serif;max-width:1400px;margin:28px auto;padding:0 22px;color:#202124}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d5d8dc;padding:8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;border:1px solid #d5d8dc}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.note{{background:#f4f6f8;border-left:4px solid #3b82f6;padding:14px}}
</style></head><body><h1>LingBot-Depth Habitat 传感器退化恢复</h1>
<p class="note">以 Habitat depth 为 gold，加入噪声、随机掉点和块状空洞；比较简单填充与 LingBot-Depth，
并把恢复后的 depth 送入相同 pose 和 DenseBEVMapper。Oracle depth 本身不需要修复。</p>
<table><thead><tr><th>Method</th><th>Coverage</th><th>MAE m</th><th>RMSE m</th><th>AbsRel</th><th>delta1</th>
<th>Free IoU</th><th>Occupied IoU</th><th>FPS</th><th>Peak VRAM</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<section><h2>Depth evidence</h2><img src="{html.escape(contact_name)}"></section>
{''.join(bev_cards)}<p><a href="lingbot_depth_metrics.json">Raw metrics JSON</a></p></body></html>"""
    (out_dir / "lingbot_depth_report.html").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
