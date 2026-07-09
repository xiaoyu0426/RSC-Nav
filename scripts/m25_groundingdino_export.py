from __future__ import annotations

import argparse
import json
import math
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
    parser = argparse.ArgumentParser(description="Run GroundingDINO on Habitat RGB-D frames and export M2.5 grounding candidates.")
    parser.add_argument("--frames-metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-id", default="m25_groundingdino_habitat")
    parser.add_argument("--labels", default="chair,table,door,bed,sofa")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-detections-per-frame", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--merge-radius-m", type=float, default=0.85)
    parser.add_argument("--min-track-confidence", type=float, default=0.10)
    parser.add_argument("--mask-backend", default="box", choices=["box", "sam"])
    parser.add_argument("--sam-model-id", default="facebook/sam-vit-base")
    parser.add_argument("--sam-min-iou", type=float, default=0.0)
    parser.add_argument("--sam-max-mask-area-ratio", type=float, default=1.0)
    parser.add_argument("--sam-min-mask-area-px", type=int, default=0)
    parser.add_argument("--depth-min-m", type=float, default=0.05)
    parser.add_argument("--depth-max-m", type=float, default=6.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.frames_metadata).read_text(encoding="utf-8"))
    labels = [item.strip().lower() for item in args.labels.split(",") if item.strip()]
    detector = _load_detector(args.model_id)
    sam = _load_sam(args.sam_model_id) if args.mask_backend == "sam" else None

    all_detections: list[dict[str, Any]] = []
    projection_debug: list[dict[str, Any]] = []
    tracks: list[ObjectTrack] = []
    frames = metadata.get("frames", [])[:: max(1, int(args.frame_stride))][: max(1, int(args.max_frames))]
    for frame in frames:
        rgb_path = Path(frame["rgb_path"])
        image = Image.open(rgb_path).convert("RGB")
        detections = _detect(
            detector,
            image,
            labels,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_detections=args.max_detections_per_frame,
        )
        projected: list[dict[str, Any]] = []
        depth = np.load(frame["depth_npy"]).astype(np.float32)
        sam_masks = (
            _sam_masks_for_detections(
                sam,
                image,
                detections,
                min_iou=float(args.sam_min_iou),
                max_mask_area_ratio=float(args.sam_max_mask_area_ratio),
                min_mask_area_px=int(args.sam_min_mask_area_px),
            )
            if sam is not None and detections
            else []
        )
        for det in detections:
            if sam is not None:
                mask_info = sam_masks.pop(0) if sam_masks else None
                if mask_info and mask_info.get("reject_reason"):
                    projection_debug.append(
                        {
                            "frame_index": int(frame["frame_index"]),
                            "label": det["label"],
                            "score": float(det["score"]),
                            "box": det["box"],
                            "raw_label": det.get("raw_label"),
                            "status": "rejected",
                            "reason": mask_info["reject_reason"],
                            "sam_iou_score": mask_info.get("iou_score"),
                            "mask_area_px": mask_info.get("mask_area_px"),
                        }
                    )
                    continue
                projection = (
                    _project_mask_detection(
                        det,
                        mask_info["mask"] if mask_info else None,
                        depth,
                        frame,
                        hfov_deg=float(metadata.get("hfov_deg", 90.0)),
                        depth_min_m=float(args.depth_min_m),
                        depth_max_m=float(args.depth_max_m),
                    )
                    if mask_info
                    else None
                )
                if projection is not None:
                    projection["sam_iou_score"] = float(mask_info["iou_score"])
                    projection["mask_area_px"] = int(mask_info["mask_area_px"])
            else:
                projection = _project_box_detection(
                    det,
                    depth,
                    frame,
                    hfov_deg=float(metadata.get("hfov_deg", 90.0)),
                    depth_min_m=float(args.depth_min_m),
                    depth_max_m=float(args.depth_max_m),
                )
            if projection is None:
                projection_debug.append(
                    {
                        "frame_index": int(frame["frame_index"]),
                        "label": det["label"],
                        "score": float(det["score"]),
                        "box": det["box"],
                        "raw_label": det.get("raw_label"),
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
                        "raw_label": det.get("raw_label"),
                        "status": "projected",
                        "position_3d": det["position_3d"],
                        "depth_median": det["depth_median"],
                        "depth_valid_ratio": det["depth_valid_ratio"],
                        "projected_points": det["projected_points"],
                        "sam_iou_score": det.get("sam_iou_score"),
                        "mask_area_px": det.get("mask_area_px"),
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
                "id": f"groundingdino_{track.label}_{index:03d}",
                "label": track.label,
                "position_3d": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                "confidence": max(0.0, min(1.0, confidence)),
                "context_id": args.context_id,
                "source": "grounding_dino",
                "source_view_ids": [f"frame_{int(det['frame_index']):04d}" for det in track.detections],
                "bbox": _best_bbox(track.detections),
                "mask_ref": None,
                "raw": {
                    "num_detections": len(track.detections),
                    "mask_backend": args.mask_backend,
                    "model_id": args.model_id,
                    "detector_backend": "grounding_dino",
                    "segmenter_backend": "sam" if args.mask_backend == "sam" else None,
                    "sam_model_id": args.sam_model_id if args.mask_backend == "sam" else None,
                    "prompt": _prompt(labels),
                    "mean_depth_valid_ratio": float(np.mean([det.get("depth_valid_ratio", 0.0) for det in track.detections])),
                    "mean_sam_iou_score": float(np.mean([det.get("sam_iou_score", 0.0) for det in track.detections])) if args.mask_backend == "sam" else None,
                    "mean_mask_area_px": float(np.mean([det.get("mask_area_px", 0.0) for det in track.detections])) if args.mask_backend == "sam" else None,
                },
            }
        )

    payload = {
        "source": "m25_groundingdino_export",
        "metadata": {
            "backend": "grounding_dino",
            "model_id": args.model_id,
            "sam_model_id": args.sam_model_id if args.mask_backend == "sam" else None,
            "sam_min_iou": args.sam_min_iou if args.mask_backend == "sam" else None,
            "sam_max_mask_area_ratio": args.sam_max_mask_area_ratio if args.mask_backend == "sam" else None,
            "sam_min_mask_area_px": args.sam_min_mask_area_px if args.mask_backend == "sam" else None,
            "labels": labels,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "mask_backend": args.mask_backend,
            "frames_metadata": str(Path(args.frames_metadata).expanduser().resolve()),
            "num_frames": len(frames),
            "num_projected_detections": len(all_detections),
            "frame_stride": int(args.frame_stride),
        },
        "items": candidates,
    }
    (out_dir / "grounding_candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "detections.json").write_text(json.dumps({"detections": all_detections}, indent=2), encoding="utf-8")
    (out_dir / "projection_debug.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "backend": "grounding_dino",
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


def _load_detector(model_id: str):
    import torch
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

    processor = GroundingDinoProcessor.from_pretrained(model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return {"processor": processor, "model": model, "device": device}


def _load_sam(model_id: str):
    import torch
    from transformers import SamModel, SamProcessor

    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return {"processor": processor, "model": model, "device": device}


def _sam_masks_for_detections(
    sam,
    image: Image.Image,
    detections: list[dict[str, Any]],
    min_iou: float,
    max_mask_area_ratio: float,
    min_mask_area_px: int,
) -> list[dict[str, Any]]:
    if sam is None or not detections:
        return []
    import torch

    boxes = [[float(v) for v in det["box"]] for det in detections]
    inputs = sam["processor"](image, input_boxes=[boxes], return_tensors="pt")
    inputs = {key: value.to(sam["device"]) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        outputs = sam["model"](**inputs)
    masks = sam["processor"].image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        inputs["reshaped_input_sizes"].detach().cpu(),
        return_tensors="pt",
    )[0]
    iou_scores = outputs.iou_scores.detach().cpu()[0]
    rows: list[dict[str, Any]] = []
    image_area = max(1, image.width * image.height)
    for index in range(len(detections)):
        best_idx = int(torch.argmax(iou_scores[index]).item())
        mask = masks[index, best_idx].numpy().astype(bool)
        iou_score = float(iou_scores[index, best_idx].item())
        area = int(mask.sum())
        reject_reason = None
        if iou_score < float(min_iou):
            reject_reason = "low_sam_iou"
        elif area < int(min_mask_area_px):
            reject_reason = "small_sam_mask"
        elif area / image_area > float(max_mask_area_ratio):
            reject_reason = "large_sam_mask"
        rows.append({"mask": mask, "iou_score": iou_score, "mask_area_px": area, "reject_reason": reject_reason})
    return rows


def _detect(
    detector,
    image: Image.Image,
    labels: list[str],
    box_threshold: float,
    text_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    import torch

    text = _prompt(labels)
    inputs = detector["processor"](images=image, text=text, return_tensors="pt")
    inputs = {key: value.to(detector["device"]) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = detector["model"](**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=detector["device"])
    results = detector["processor"].post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.get("input_ids"),
        threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        target_sizes=target_sizes,
    )[0]
    rows = []
    for score, box, raw_label in zip(results["scores"], results["boxes"], results["labels"]):
        x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
        x1 = max(0.0, min(float(image.width - 1), x1))
        x2 = max(0.0, min(float(image.width), x2))
        y1 = max(0.0, min(float(image.height - 1), y1))
        y2 = max(0.0, min(float(image.height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        label = _match_label(str(raw_label), labels)
        if label is None:
            continue
        rows.append(
            {
                "label": label,
                "score": float(score.detach().cpu().item()),
                "box": [x1, y1, x2, y2],
                "raw_label": str(raw_label),
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[: max(1, int(max_detections))]


def _prompt(labels: list[str]) -> str:
    return ". ".join(labels) + "."


def _match_label(raw_label: str, labels: list[str]) -> str | None:
    text = raw_label.lower().strip()
    if not text:
        return None
    best_label = None
    best_pos = 10**9
    for label in labels:
        pos = text.find(label)
        if pos >= 0 and pos < best_pos:
            best_label = label
            best_pos = pos
    return best_label


def _project_box_detection(
    det: dict[str, Any],
    depth: np.ndarray,
    frame: dict[str, Any],
    hfov_deg: float,
    depth_min_m: float,
    depth_max_m: float,
) -> dict[str, Any] | None:
    h, w = depth.shape
    x1, y1, x2, y2 = det["box"]
    x1 = max(0, min(w - 1, int(round(x1))))
    x2 = max(0, min(w, int(round(x2))))
    y1 = max(0, min(h - 1, int(round(y1))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    crop = depth[y1 + pad_y : max(y1 + pad_y + 1, y2 - pad_y), x1 + pad_x : max(x1 + pad_x + 1, x2 - pad_x)]
    valid_mask = np.isfinite(crop) & (crop > depth_min_m) & (crop < depth_max_m)
    valid = crop[valid_mask]
    if valid.size < 8:
        return None
    low, high = np.percentile(valid, [10, 65])
    mask = np.isfinite(crop) & (crop >= low) & (crop <= high)
    rows, cols = np.where(mask)
    if rows.size < 8:
        return None
    stride = max(1, int(math.sqrt(rows.size / 220.0)))
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
        "depth_valid_ratio": float(valid.size / max(1, crop.size)),
        "projected_points": int(world.shape[0]),
    }


def _project_mask_detection(
    det: dict[str, Any],
    mask: np.ndarray | None,
    depth: np.ndarray,
    frame: dict[str, Any],
    hfov_deg: float,
    depth_min_m: float,
    depth_max_m: float,
) -> dict[str, Any] | None:
    if mask is None:
        return None
    if mask.shape != depth.shape:
        return None
    rows, cols = np.where(mask)
    if rows.size < 8:
        return None
    z_all = depth[rows, cols].astype(np.float32)
    valid_mask = np.isfinite(z_all) & (z_all > depth_min_m) & (z_all < depth_max_m)
    if int(valid_mask.sum()) < 8:
        return None
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    z = z_all[valid_mask]
    low, high = np.percentile(z, [5, 85])
    fg = (z >= low) & (z <= high)
    rows = rows[fg]
    cols = cols[fg]
    z = z[fg]
    if z.size < 8:
        return None
    stride = max(1, int(math.sqrt(z.size / 260.0)))
    rows = rows[::stride]
    cols = cols[::stride]
    z = z[::stride]
    h, w = depth.shape
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
        "depth_valid_ratio": float(valid_mask.sum() / max(1, int(mask.sum()))),
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
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    colors = {
        "chair": "#59a14f",
        "table": "#f28e2b",
        "door": "#e15759",
        "bed": "#1f77b4",
        "sofa": "#9467bd",
    }
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = colors.get(det["label"], "#00a6d6")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f"{det['label']} {float(det['score']):.2f}"
        tw = draw.textlength(text, font=font)
        draw.rectangle((x1, max(0, y1 - 18), x1 + tw + 6, y1), fill=color)
        draw.text((x1 + 3, max(0, y1 - 17)), text, fill="white", font=font)
    image.save(path, quality=92)


def _best_bbox(detections: list[dict[str, Any]]) -> list[float] | None:
    if not detections:
        return None
    best = max(detections, key=lambda item: float(item.get("score", 0.0)))
    return [float(value) for value in best.get("box", [])]


def _write_report(out_dir: Path, payload: dict[str, Any], detections: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f"<tr><td>{item['id']}</td><td>{item['label']}</td><td>{item['confidence']:.3f}</td>"
        f"<td>{item['position_3d'][0]:.2f}, {item['position_3d'][1]:.2f}, {item['position_3d'][2]:.2f}</td>"
        f"<td>{len(item['source_view_ids'])}</td></tr>"
        for item in payload["items"]
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GroundingDINO Export</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202124;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d7dce2;padding:6px}}th{{background:#f1f4f8}}
code{{background:#f1f3f4;padding:2px 4px;border-radius:4px}}
</style></head><body>
<h1>GroundingDINO RGB-D Grounding Export</h1>
<p>Model: <code>{payload['metadata']['model_id']}</code> | backend: <code>grounding_dino</code> | mask backend: <code>{payload['metadata']['mask_backend']}</code></p>
<p>Frames: {payload['metadata']['num_frames']} | projected detections: {len(detections)} | object candidates: {len(payload['items'])}</p>
<p>This is the high-quality semantic grounding branch entry point. The current run may still use box evidence; SAM/SAM2 mask evidence should reuse the same candidate schema.</p>
<p><a href="grounding_candidates.json">grounding_candidates.json</a> |
<a href="detections.json">detections.json</a> |
<a href="projection_debug.json">projection_debug.json</a></p>
<h2>Object Candidates</h2>
<table><tr><th>ID</th><th>Label</th><th>Confidence</th><th>3D Position</th><th>Views</th></tr>{rows}</table>
</body></html>
"""
    (out_dir / "groundingdino_export_report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
