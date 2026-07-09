from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class ObjectTrack:
    label: str
    position_sum: np.ndarray
    confidence_sum: float
    weight_sum: float
    detections: list[dict[str, Any]] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        return self.position_sum / max(1e-6, self.weight_sum)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OWLv2/OwlViT on Habitat RGB-D frames and export M2.5 grounding candidates.")
    parser.add_argument("--frames-metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-id", default="m25_owlv2_habitat")
    parser.add_argument("--labels", default="chair,table,door,bed,sofa")
    parser.add_argument("--model-id", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--backend", choices=["owlv2", "owlvit"], default="owlv2")
    parser.add_argument("--box-threshold", type=float, default=0.08)
    parser.add_argument("--max-detections-per-frame", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--merge-radius-m", type=float, default=0.85)
    parser.add_argument("--min-track-confidence", type=float, default=0.05)
    parser.add_argument("--mask-backend", default="box", choices=["box"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.frames_metadata).read_text(encoding="utf-8"))
    labels = [item.strip().lower() for item in args.labels.split(",") if item.strip()]
    detector = _load_detector(args.backend, args.model_id)

    all_detections = []
    projection_debug = []
    tracks: list[ObjectTrack] = []
    frames = metadata.get("frames", [])[: max(1, int(args.max_frames))]
    for frame in frames:
        rgb_path = Path(frame["rgb_path"])
        image = Image.open(rgb_path).convert("RGB")
        detections = _detect(detector, image, labels, threshold=args.box_threshold, max_detections=args.max_detections_per_frame)
        projected = []
        depth = np.load(frame["depth_npy"]).astype(np.float32)
        for det in detections:
            projection = _project_box_detection(det, depth, frame, hfov_deg=float(metadata.get("hfov_deg", 90.0)))
            if projection is None:
                projection_debug.append(
                    {
                        "frame_index": int(frame["frame_index"]),
                        "label": det["label"],
                        "score": float(det["score"]),
                        "box": det["box"],
                        "status": "rejected",
                        "reason": "insufficient_valid_depth_or_invalid_projection",
                    }
                )
                continue
            det = {**det, **projection, "frame_index": int(frame["frame_index"]), "rgb_path": str(rgb_path)}
            projected.append(det)
            projection_debug.append(
                {
                    "frame_index": int(frame["frame_index"]),
                    "label": det["label"],
                    "score": float(det["score"]),
                    "box": det["box"],
                    "status": "projected",
                    "position_3d": det["position_3d"],
                    "depth_median": det["depth_median"],
                    "projected_points": det["projected_points"],
                }
            )
            _merge_detection(tracks, det, merge_radius_m=args.merge_radius_m)
        all_detections.extend(projected)
        _write_overlay(image, projected, overlays_dir / f"frame_{int(frame['frame_index']):04d}_overlay.jpg")

    candidates = []
    for index, track in enumerate(tracks):
        confidence = float(track.confidence_sum / max(1e-6, track.weight_sum))
        if confidence < float(args.min_track_confidence):
            continue
        centroid = track.centroid
        candidates.append(
            {
                "id": f"owlv2_{track.label}_{index:03d}",
                "label": track.label,
                "position_3d": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                "confidence": max(0.0, min(1.0, confidence)),
                "context_id": args.context_id,
                "source": args.backend,
                "source_view_ids": [f"frame_{int(det['frame_index']):04d}" for det in track.detections],
                "bbox": _best_bbox(track.detections),
                "mask_ref": None,
                "raw": {
                    "num_detections": len(track.detections),
                    "mask_backend": args.mask_backend,
                    "model_id": args.model_id,
                },
            }
        )

    payload = {
        "source": "m25_owlv2_grounding_export",
        "metadata": {
            "backend": args.backend,
            "model_id": args.model_id,
            "labels": labels,
            "box_threshold": args.box_threshold,
            "mask_backend": args.mask_backend,
            "frames_metadata": str(Path(args.frames_metadata).expanduser().resolve()),
            "num_frames": len(frames),
            "num_projected_detections": len(all_detections),
        },
        "items": candidates,
    }
    (out_dir / "grounding_candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "detections.json").write_text(json.dumps({"detections": all_detections}, indent=2), encoding="utf-8")
    (out_dir / "projection_debug.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "backend": args.backend,
                    "model_id": args.model_id,
                    "mask_backend": args.mask_backend,
                    "frames_metadata": str(Path(args.frames_metadata).expanduser().resolve()),
                },
                "items": projection_debug,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(out_dir, payload, all_detections)
    print(json.dumps({"num_candidates": len(candidates), "num_projected_detections": len(all_detections), "out_dir": str(out_dir)}, indent=2))


def _load_detector(backend: str, model_id: str):
    import torch
    from transformers import OwlViTForObjectDetection, OwlViTProcessor, Owlv2ForObjectDetection, Owlv2Processor

    if backend == "owlvit":
        processor = OwlViTProcessor.from_pretrained(model_id)
        model = OwlViTForObjectDetection.from_pretrained(model_id)
    else:
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return {"processor": processor, "model": model, "device": device}


def _detect(detector, image: Image.Image, labels: list[str], threshold: float, max_detections: int) -> list[dict[str, Any]]:
    import torch

    prompts = [f"a photo of a {label}" for label in labels]
    inputs = detector["processor"](text=[prompts], images=image, return_tensors="pt")
    inputs = {key: value.to(detector["device"]) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = detector["model"](**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=detector["device"])
    results = detector["processor"].post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=float(threshold))[0]
    rows = []
    for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
        idx = int(label_idx.detach().cpu().item())
        if idx < 0 or idx >= len(labels):
            continue
        x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append({"label": labels[idx], "score": float(score.detach().cpu().item()), "box": [x1, y1, x2, y2]})
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[: max(1, int(max_detections))]


def _project_box_detection(det: dict[str, Any], depth: np.ndarray, frame: dict[str, Any], hfov_deg: float) -> dict[str, Any] | None:
    h, w = depth.shape
    x1, y1, x2, y2 = det["box"]
    x1 = max(0, min(w - 1, int(round(x1))))
    x2 = max(0, min(w, int(round(x2))))
    y1 = max(0, min(h - 1, int(round(y1))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    pad_x = int((x2 - x1) * 0.2)
    pad_y = int((y2 - y1) * 0.2)
    crop = depth[y1 + pad_y : max(y1 + pad_y + 1, y2 - pad_y), x1 + pad_x : max(x1 + pad_x + 1, x2 - pad_x)]
    valid = crop[np.isfinite(crop) & (crop > 0.05) & (crop < 6.0)]
    if valid.size < 8:
        return None
    low, high = np.percentile(valid, [10, 70])
    mask = np.isfinite(crop) & (crop >= low) & (crop <= high)
    rows, cols = np.where(mask)
    if rows.size < 8:
        return None
    stride = max(1, int(math.sqrt(rows.size / 160.0)))
    rows = rows[::stride] + y1 + pad_y
    cols = cols[::stride] + x1 + pad_x
    z = depth[rows, cols].astype(np.float32)
    fx = w / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    fy = fx
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    x_cam = (cols.astype(np.float32) - cx) / fx * z
    y_cam = -(rows.astype(np.float32) - cy) / fy * z
    z_cam = -z
    camera_points = np.stack([x_cam, y_cam, z_cam], axis=1)
    rotation = np.asarray(frame["sensor_rotation_matrix"], dtype=np.float32)
    sensor_position = np.asarray(frame["sensor_position_xyz"], dtype=np.float32).reshape(1, 3)
    world = sensor_position + camera_points @ rotation.T
    centroid = np.median(world, axis=0)
    if not np.isfinite(centroid).all():
        return None
    return {
        "position_3d": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
        "depth_median": float(np.median(z)),
        "projected_points": int(world.shape[0]),
    }


def _merge_detection(tracks: list[ObjectTrack], det: dict[str, Any], merge_radius_m: float) -> None:
    position = np.asarray(det["position_3d"], dtype=np.float32)
    best = None
    best_dist = math.inf
    for track in tracks:
        if track.label != det["label"]:
            continue
        dist = float(np.linalg.norm(track.centroid[[0, 2]] - position[[0, 2]]))
        if dist < best_dist:
            best = track
            best_dist = dist
    weight = max(0.01, float(det["score"]))
    if best is not None and best_dist <= float(merge_radius_m):
        best.position_sum += position * weight
        best.confidence_sum += float(det["score"]) * weight
        best.weight_sum += weight
        best.detections.append(det)
        return
    tracks.append(ObjectTrack(label=det["label"], position_sum=position * weight, confidence_sum=float(det["score"]) * weight, weight_sum=weight, detections=[det]))


def _write_overlay(image: Image.Image, detections: list[dict[str, Any]], path: Path) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    colors = {"chair": "#59a14f", "table": "#f28e2b", "door": "#e15759", "bed": "#1f77b4", "sofa": "#9467bd", "wall": "#444444"}
    for det in detections:
        color = colors.get(det["label"], "#4c78a8")
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 3, y1 + 3), f"{det['label']} {det['score']:.2f}", fill=color, font=font)
    image.save(path, quality=92)


def _best_bbox(detections: list[dict[str, Any]]) -> dict[str, float] | None:
    if not detections:
        return None
    best = max(detections, key=lambda item: float(item.get("score", 0.0)))
    x1, y1, x2, y2 = best["box"]
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "frame_index": int(best["frame_index"])}


def _write_report(out_dir: Path, payload: dict[str, Any], detections: list[dict[str, Any]]) -> None:
    overlays = sorted((out_dir / "overlays").glob("*_overlay.jpg"))[:12]
    image_tags = "\n".join(f'<img src="overlays/{path.name}" alt="{path.name}">' for path in overlays)
    rows = "\n".join(
        "<tr>"
        f"<td>{item['id']}</td><td>{item['label']}</td><td>{item['confidence']:.3f}</td>"
        f"<td>{item['position_3d'][0]:.2f}, {item['position_3d'][1]:.2f}, {item['position_3d'][2]:.2f}</td>"
        f"<td>{len(item.get('source_view_ids', []))}</td>"
        "</tr>"
        for item in payload["items"]
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>M2.5 OWLv2 Grounding Export</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202124}}img{{max-width:360px;border:1px solid #ddd;margin:6px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #ddd;padding:6px}}</style>
</head><body>
<h1>M2.5 OWLv2 Grounding Export</h1>
<p>Model: {payload['metadata']['model_id']} | backend: {payload['metadata']['backend']} | mask backend: {payload['metadata']['mask_backend']}</p>
<p>Projected detections: {len(detections)} | object candidates: {len(payload['items'])}</p>
<h2>Detection Overlays</h2>{image_tags}
<h2>Object Inventory</h2><table><tr><th>ID</th><th>Label</th><th>Confidence</th><th>Position XYZ</th><th>Views</th></tr>{rows}</table>
</body></html>"""
    (out_dir / "owlv2_grounding_report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
