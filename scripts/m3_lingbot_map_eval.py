from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m3_vggt_geometry_eval import (
    _align_vggt_to_oracle,
    _build_oracle_mapper,
    _build_vggt_mapper,
    _compare_bev,
    _grid_config,
    _oracle_sensor_centers,
    _render_bev,
    _save_bev_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LingBot-Map against VGGT and Habitat geometry.")
    parser.add_argument("--frames-metadata", required=True)
    parser.add_argument("--lingbot-map-repo", required=True)
    parser.add_argument("--model-pt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vggt-metrics")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata, frames = _load_frames(Path(args.frames_metadata), args.max_frames)
    map_repo = Path(args.lingbot_map_repo).expanduser().resolve()
    sys.path.insert(0, str(map_repo))
    import demo as map_demo

    image_paths = [frame["rgb_path"] for frame in frames]
    images = map_demo.load_and_preprocess_images(
        image_paths,
        mode="crop",
        image_size=518,
        patch_size=14,
    )
    model_args = SimpleNamespace(
        mode="streaming",
        image_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        num_scale_frames=min(args.num_scale_frames, len(frames) - 1),
        use_sdpa=True,
        camera_num_iterations=args.camera_num_iterations,
        model_path=str(Path(args.model_pt).expanduser().resolve()),
    )
    device = torch.device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = map_demo.load_model(model_args, device)
    load_seconds = time.perf_counter() - load_started
    dtype = torch.bfloat16
    model.aggregator = model.aggregator.to(dtype=dtype)
    images = images.to(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        raw_prediction = model.inference_streaming(
            images,
            num_scale_frames=model_args.num_scale_frames,
            keyframe_interval=1,
            output_device=torch.device("cpu"),
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsic, intrinsic = pose_encoding_to_extri_intri(raw_prediction["pose_enc"], images.shape[-2:])
    prediction = _normalize_prediction(raw_prediction, extrinsic, intrinsic)

    align = _align_vggt_to_oracle(prediction["camera_centers"], _oracle_sensor_centers(frames))
    grid_cfg = _grid_config(metadata, frames)
    mapper_args = SimpleNamespace()
    oracle_mapper = _build_oracle_mapper(frames, grid_cfg, metadata, mapper_args)
    map_mapper, _ = _build_vggt_mapper(frames, prediction, align, grid_cfg, metadata, mapper_args)
    _save_bev_state(out_dir / "lingbot_map_bev_state.npz", map_mapper, metadata)
    oracle_bev = _render_bev(oracle_mapper, out_dir / "oracle_bev.png", "Habitat oracle RGB-D/pose BEV")
    map_bev = _render_bev(map_mapper, out_dir / "lingbot_map_bev.png", "LingBot-Map RGB-only BEV")

    scaled_depth = prediction["depth"] * float(align["scale"])
    depth_metrics = _depth_metrics(frames, scaled_depth)
    depth_contact = out_dir / "lingbot_map_depth_contact_sheet.png"
    _render_depth_contact(frames, scaled_depth, depth_contact)
    np.savez_compressed(
        out_dir / "lingbot_map_geometry.npz",
        extrinsic=prediction["extrinsic"],
        intrinsic=prediction["intrinsic"],
        depth=prediction["depth"],
        depth_conf=prediction["depth_conf"],
        camera_centers=prediction["camera_centers"],
        camera_rotations=prediction["camera_rotations"],
        alignment=json.dumps(align),
    )
    baseline = None
    if args.vggt_metrics:
        baseline = json.loads(Path(args.vggt_metrics).expanduser().read_text(encoding="utf-8"))
    metrics = {
        "phase": "m3_lingbot_foundation_benchmark",
        "benchmark": "lingbot_map_vs_vggt",
        "model": "LingBot-Map-long",
        "num_frames": len(frames),
        "alignment": align,
        "depth": depth_metrics,
        "bev_comparison": _compare_bev(map_mapper, oracle_mapper),
        "runtime": {
            "load_seconds": float(load_seconds),
            "inference_seconds": float(inference_seconds),
            "fps": float(len(frames) / max(inference_seconds, 1e-8)),
            "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1e9),
        },
        "vggt_baseline": {
            "backend": baseline.get("backend"),
            "alignment": baseline.get("alignment"),
            "bev_comparison": baseline.get("bev_comparison"),
        }
        if baseline
        else None,
        "outputs": {
            "oracle_bev": oracle_bev.name,
            "lingbot_map_bev": map_bev.name,
            "depth_contact_sheet": depth_contact.name,
            "geometry_npz": "lingbot_map_geometry.npz",
        },
    }
    (out_dir / "lingbot_map_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_report(out_dir, metrics)
    print(json.dumps(metrics, indent=2))


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


def _normalize_prediction(prediction: dict, extrinsic, intrinsic) -> dict:
    def array(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)

    depth = array(prediction["depth"])
    depth_conf = array(prediction["depth_conf"])
    extrinsic = array(extrinsic)
    intrinsic = array(intrinsic)
    if depth.ndim == 5 and depth.shape[0] == 1:
        depth = depth[0]
    if depth_conf.ndim == 4 and depth_conf.shape[0] == 1:
        depth_conf = depth_conf[0]
    if extrinsic.ndim == 4 and extrinsic.shape[0] == 1:
        extrinsic = extrinsic[0]
    if intrinsic.ndim == 4 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    elif depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth_conf.ndim == 4 and depth_conf.shape[1] == 1:
        depth_conf = depth_conf[:, 0]
    elif depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[..., 0]
    return {
        "extrinsic": extrinsic.astype(np.float32),
        "intrinsic": intrinsic.astype(np.float32),
        "depth": depth.astype(np.float32),
        "depth_conf": depth_conf.astype(np.float32),
        "camera_centers": extrinsic[:, :3, 3].astype(np.float64),
        "camera_rotations": extrinsic[:, :3, :3].astype(np.float64),
    }


def _depth_metrics(frames: list[dict], predictions: np.ndarray) -> dict:
    targets = []
    values = []
    for frame, prediction in zip(frames, predictions):
        target = np.load(frame["depth_npy"]).astype(np.float32)
        resized = cv2.resize(prediction, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = np.isfinite(target) & (target > 0.05) & np.isfinite(resized) & (resized > 0.05)
        targets.append(target[mask])
        values.append(resized[mask])
    target = np.concatenate(targets)
    prediction = np.concatenate(values)
    error = prediction - target
    ratio = np.maximum(target / np.maximum(prediction, 1e-6), prediction / np.maximum(target, 1e-6))
    return {
        "valid_pixels": int(target.size),
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


def _render_depth_contact(frames: list[dict], predictions: np.ndarray, output: Path) -> None:
    indices = np.linspace(0, len(frames) - 1, min(4, len(frames)), dtype=int)
    panel_size = 360
    canvas = Image.new("RGB", (panel_size * len(indices), panel_size * 3 + 90), "white")
    draw = ImageDraw.Draw(canvas)
    for column, index in enumerate(indices):
        target = np.load(frames[index]["depth_npy"]).astype(np.float32)
        prediction = cv2.resize(predictions[index], (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
        panels = [
            ("RGB", Image.open(frames[index]["rgb_path"]).convert("RGB")),
            ("Oracle depth", _depth_color(target)),
            ("LingBot-Map depth", _depth_color(prediction)),
        ]
        for row, (label, image) in enumerate(panels):
            x = column * panel_size
            y = row * (panel_size + 30)
            canvas.paste(image.resize((panel_size, panel_size), Image.Resampling.BILINEAR), (x, y + 30))
            draw.text((x + 10, y + 8), f"F{index:02d} {label}", fill="#202124")
    canvas.save(output)


def _write_report(out_dir: Path, metrics: dict) -> None:
    current = metrics["bev_comparison"]
    baseline = metrics.get("vggt_baseline") or {}
    baseline_align = baseline.get("alignment") or {}
    baseline_bev = baseline.get("bev_comparison") or {}
    rows = [
        "<tr><td>VGGT-1B baseline</td>"
        f"<td>{baseline_align.get('ate_rmse_m', float('nan')):.3f}</td>"
        f"<td>{baseline_bev.get('explored_iou', float('nan')):.3f}</td>"
        f"<td>{baseline_bev.get('free_iou', float('nan')):.3f}</td>"
        f"<td>{baseline_bev.get('occupied_iou', float('nan')):.3f}</td><td>-</td><td>-</td></tr>",
        "<tr><td>LingBot-Map-long</td>"
        f"<td>{metrics['alignment']['ate_rmse_m']:.3f}</td><td>{current['explored_iou']:.3f}</td>"
        f"<td>{current['free_iou']:.3f}</td><td>{current['occupied_iou']:.3f}</td>"
        f"<td>{metrics['runtime']['fps']:.2f}</td><td>{metrics['runtime']['peak_vram_gb']:.2f} GB</td></tr>",
    ]
    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>LingBot-Map vs VGGT</title><style>
body{{font-family:Arial,sans-serif;max-width:1400px;margin:28px auto;padding:0 22px;color:#202124}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #d5d8dc;padding:9px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
img{{width:100%;border:1px solid #d5d8dc}}.note{{background:#f4f6f8;border-left:4px solid #3b82f6;padding:14px}}
</style></head><body><h1>LingBot-Map 与现有 VGGT 几何前端对照</h1>
<p class="note">同一 Habitat 连续 16 帧 RGB，均用 Sim(3) 对齐到 Habitat world frame，再进入相同 DenseBEVMapper。
这是几何前端对照，不含 GroundingDINO/SAM 语义误差。</p>
<table><thead><tr><th>Method</th><th>ATE RMSE m</th><th>Explored IoU</th><th>Free IoU</th>
<th>Occupied IoU</th><th>FPS</th><th>Peak VRAM</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="grid"><section><h2>Oracle</h2><img src="oracle_bev.png"></section>
<section><h2>LingBot-Map</h2><img src="lingbot_map_bev.png"></section></div>
<section><h2>Depth evidence</h2><img src="lingbot_map_depth_contact_sheet.png"></section>
<p><a href="lingbot_map_metrics.json">Raw metrics JSON</a></p></body></html>"""
    (out_dir / "lingbot_map_report.html").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
