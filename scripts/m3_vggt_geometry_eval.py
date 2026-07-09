from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_bev_mapper import DenseBEVConfig, DenseBEVMapper
from semantic_grounding_adapter import GroundingCandidate, candidates_from_habitat_memory, compare_candidates, write_grounding_candidates


SEMANTIC_COLORS = {
    "wall": "#2f2f2f",
    "door": "#e15759",
    "table": "#f28e2b",
    "chair": "#59a14f",
    "bed": "#1f77b4",
    "sofa": "#9467bd",
}


@dataclass
class MatrixRotation:
    matrix: np.ndarray

    def transform_vector(self, point) -> np.ndarray:
        return np.asarray(point, dtype=np.float32) @ self.matrix.T


@dataclass
class Track:
    label: str
    position_sum: np.ndarray
    confidence_sum: float
    weight_sum: float
    detections: list[dict[str, Any]] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        return self.position_sum / max(1e-6, self.weight_sum)


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 VGGT RGB geometry frontend -> BEV/semantic BEV validation.")
    parser.add_argument("--frames-metadata", required=True)
    parser.add_argument("--detections-json", required=True, help="2D OWLv2/GroundingDINO detections from the same frames.")
    parser.add_argument("--gold-memory-json", required=True, help="Habitat semantic-oracle object memory for validation.")
    parser.add_argument("--vggt-repo", default=str(ROOT / "downloads" / "third_party" / "vggt"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "m3_rgb_geometry_frontend" / "vggt_habitat_validation"))
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--model-pt", help="Optional local VGGT model.pt path. Use this when the remote host cannot reach HuggingFace.")
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--selection-mode", choices=["even", "first"], default="even")
    parser.add_argument("--image-load-resolution", type=int, default=1024)
    parser.add_argument("--vggt-resolution", type=int, default=518)
    parser.add_argument("--depth-conf-threshold", type=float, default=3.0)
    parser.add_argument("--point-sample-stride", type=int, default=8)
    parser.add_argument("--bev-sample-stride", type=int, default=4)
    parser.add_argument("--max-depth-m", type=float, default=7.0)
    parser.add_argument("--merge-radius-m", type=float, default=0.85)
    parser.add_argument("--min-track-confidence", type=float, default=0.05)
    parser.add_argument("--semantic-radius-m", type=float, default=0.25)
    parser.add_argument("--context-id", default="m3_vggt_habitat_A")
    parser.add_argument("--scene-id", default="habitat_mp3d_A_vggt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_json(args.frames_metadata)
    frames = _select_frames(metadata.get("frames", []), max_frames=args.max_frames, stride=args.frame_stride, mode=args.selection_mode)
    if len(frames) < 3:
        raise SystemExit("VGGT alignment needs at least 3 selected frames.")
    _localize_frame_paths(frames, Path(args.frames_metadata).expanduser().resolve())
    _prepare_input_images(frames, out_dir / "vggt_input" / "images")

    prediction = _run_vggt(
        frames=frames,
        vggt_repo=Path(args.vggt_repo).expanduser().resolve(),
        model_id=args.model_id,
        model_pt=Path(args.model_pt).expanduser().resolve() if args.model_pt else None,
        image_load_resolution=args.image_load_resolution,
        vggt_resolution=args.vggt_resolution,
    )
    align = _align_vggt_to_oracle(prediction["camera_centers"], _oracle_sensor_centers(frames))
    _save_geometry_outputs(out_dir, frames, prediction, align, args)

    grid_cfg = _grid_config(metadata, frames)
    oracle_mapper = _build_oracle_mapper(frames, grid_cfg, metadata, args)
    vggt_mapper, aligned_camera = _build_vggt_mapper(frames, prediction, align, grid_cfg, metadata, args)

    oracle_bev_state = _save_bev_state(out_dir / "oracle_bev_state_selected.npz", oracle_mapper, metadata)
    vggt_bev_state = _save_bev_state(out_dir / "vggt_bev_state.npz", vggt_mapper, metadata)
    oracle_bev_png = _render_bev(oracle_mapper, out_dir / "oracle_bev_selected.png", "Oracle RGB-D/Pose BEV (selected frames)")
    vggt_bev_png = _render_bev(vggt_mapper, out_dir / "vggt_traditional_bev.png", "VGGT Estimated Depth/Pose BEV")
    depth_contact = _render_depth_contact_sheet(out_dir, frames, prediction)

    candidates, projected_debug = _reproject_detections_with_vggt(
        frames=frames,
        detections_json=Path(args.detections_json).expanduser().resolve(),
        prediction=prediction,
        align=align,
        aligned_camera=aligned_camera,
        context_id=args.context_id,
        merge_radius_m=args.merge_radius_m,
        min_track_confidence=args.min_track_confidence,
    )
    candidates_path = out_dir / "vggt_grounding_candidates.json"
    write_grounding_candidates(
        candidates_path,
        candidates,
        metadata={
            "backend": "owlv2_boxes_vggt_depth_pose",
            "context_id": args.context_id,
            "source_detections": str(Path(args.detections_json).expanduser().resolve()),
        },
    )
    (out_dir / "vggt_projection_debug.json").write_text(json.dumps({"items": projected_debug}, indent=2), encoding="utf-8")

    gold = candidates_from_habitat_memory(args.gold_memory_json, context_id=args.context_id)
    validation = compare_candidates(candidates, gold, distance_threshold_m=0.75)
    metrics = {
        "phase": "m3_rgb_geometry_frontend",
        "status": "passed" if candidates and align["ate_rmse_m"] < 1.5 else "needs_review",
        "backend": "VGGT-1B + OWLv2 boxes",
        "context_id": args.context_id,
        "num_frames": len(frames),
        "num_candidates": len(candidates),
        "alignment": align,
        "bev_comparison": _compare_bev(vggt_mapper, oracle_mapper),
        "validation": validation,
        "outputs": {},
    }
    metrics_path = out_dir / "m3_vggt_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bridge_dir = out_dir / "bridge"
    bridge_metadata = _run_bridge(candidates_path, out_dir / "vggt_pointcloud_aligned.pcd", bridge_dir, args)
    metrics["bridge"] = bridge_metadata
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    semantic_mvp_dir = out_dir / "rgb_to_semantic_bev_mvp"
    sample_frame = frames[min(1, len(frames) - 1)]
    overlay_path = _best_overlay_for_frame(Path(args.detections_json).resolve(), sample_frame["frame_index"])
    if overlay_path is None:
        overlay_path = Path(sample_frame["rgb_path"])
    _run_semantic_mvp(
        bev_state=vggt_bev_state,
        candidates_json=candidates_path,
        metrics_json=metrics_path,
        rgb=Path(sample_frame["rgb_path"]),
        overlay=overlay_path,
        out_dir=semantic_mvp_dir,
    )

    metrics["outputs"] = {
        "report": str(out_dir / "m3_vggt_geometry_report.html"),
        "metrics": str(metrics_path),
        "vggt_bev_state": str(vggt_bev_state),
        "oracle_bev_state_selected": str(oracle_bev_state),
        "vggt_traditional_bev": str(vggt_bev_png),
        "oracle_bev_selected": str(oracle_bev_png),
        "depth_contact_sheet": str(depth_contact),
        "pointcloud_pcd": str(out_dir / "vggt_pointcloud_aligned.pcd"),
        "grounding_candidates": str(candidates_path),
        "semantic_mvp": str(semantic_mvp_dir / "rgb_to_semantic_bev_mvp.html"),
        "bridge_report": str(bridge_dir / "bridge_report.html"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_report(out_dir, metrics, frames)
    print(json.dumps(metrics, indent=2))


def _run_vggt(
    frames: list[dict[str, Any]],
    vggt_repo: Path,
    model_id: str,
    model_pt: Path | None,
    image_load_resolution: int,
    vggt_resolution: int,
) -> dict[str, Any]:
    if not vggt_repo.exists():
        raise SystemExit(f"VGGT repo not found: {vggt_repo}")
    sys.path.insert(0, str(vggt_repo))

    import torch
    import torch.nn.functional as F
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_paths = [str(frame["rgb_path"]) for frame in frames]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("VGGT-1B run requires CUDA for this pipeline.")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    if model_pt is not None:
        if not model_pt.exists():
            raise SystemExit(f"VGGT model checkpoint not found: {model_pt}")
        model = VGGT()
        model.load_state_dict(torch.load(model_pt, map_location="cpu"))
    else:
        model = VGGT.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    images, original_coords = load_and_preprocess_images_square(image_paths, int(image_load_resolution))
    images = images.to(device)
    images_518 = F.interpolate(images, size=(int(vggt_resolution), int(vggt_resolution)), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            batch = images_518[None]
            aggregated_tokens_list, ps_idx = model.aggregator(batch)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batch, ps_idx)

    extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()
    depth_np = depth_map.squeeze(0).detach().cpu().numpy()
    depth_conf_np = depth_conf.squeeze(0).detach().cpu().numpy()
    points_np = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)
    cam_to_world = closed_form_inverse_se3(extrinsic_np)
    camera_centers = cam_to_world[:, :3, 3].astype(np.float64)
    camera_rotations = cam_to_world[:, :3, :3].astype(np.float64)

    return {
        "extrinsic": extrinsic_np,
        "intrinsic": intrinsic_np,
        "depth": depth_np.squeeze(-1) if depth_np.ndim == 4 else depth_np,
        "depth_conf": depth_conf_np.squeeze(-1) if depth_conf_np.ndim == 4 else depth_conf_np,
        "points": points_np,
        "camera_centers": camera_centers,
        "camera_rotations": camera_rotations,
        "image_resolution": int(vggt_resolution),
        "original_coords": original_coords.detach().cpu().numpy(),
    }


def _align_vggt_to_oracle(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mu_src = source.mean(axis=0)
    mu_dst = target.mean(axis=0)
    src_c = source - mu_src
    dst_c = target - mu_dst
    cov = (dst_c.T @ src_c) / max(1, len(source))
    u, _, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1
    rotation = u @ d @ vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.trace(np.diag(np.linalg.svd(cov, compute_uv=False)) @ d) / max(1e-9, var_src))
    translation = mu_dst - scale * rotation @ mu_src
    aligned = (scale * (rotation @ source.T)).T + translation
    errors = np.linalg.norm(aligned - target, axis=1)
    return {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "ate_rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "ate_mean_m": float(errors.mean()),
        "ate_max_m": float(errors.max()),
    }


def _transform_points(points: np.ndarray, align: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(align["rotation"], dtype=np.float64)
    translation = np.asarray(align["translation"], dtype=np.float64)
    scale = float(align["scale"])
    flat = points.reshape(-1, 3)
    out = (scale * (rotation @ flat.T)).T + translation
    return out.reshape(points.shape).astype(np.float32)


def _build_vggt_mapper(frames, prediction, align, grid_cfg, metadata, args) -> tuple[DenseBEVMapper, list[dict[str, Any]]]:
    mapper = DenseBEVMapper(origin_world_xz=grid_cfg["origin"], config=grid_cfg["config"])
    align_rot = np.asarray(align["rotation"], dtype=np.float32)
    scale = float(align["scale"])
    axis_fix = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    aligned_camera = []
    centers = _transform_points(prediction["camera_centers"], align)
    rotations_cv = prediction["camera_rotations"]
    for idx, frame in enumerate(frames):
        sensor_position = centers[idx].astype(np.float32)
        camera_height = float(metadata.get("camera_height_m", 1.5))
        agent_position = np.asarray([sensor_position[0], sensor_position[1] - camera_height, sensor_position[2]], dtype=np.float32)
        sensor_rotation = align_rot @ rotations_cv[idx].astype(np.float32) @ axis_fix
        depth = np.asarray(prediction["depth"][idx], dtype=np.float32) * scale
        mapper.update_from_depth(
            depth=depth,
            agent_position_xyz=agent_position,
            sensor_position_xyz=sensor_position,
            sensor_rotation=MatrixRotation(sensor_rotation),
            hfov_deg=float(metadata.get("hfov_deg", 90.0)),
        )
        aligned_camera.append(
            {
                "frame_index": int(frame["frame_index"]),
                "sensor_position_xyz": sensor_position.tolist(),
                "agent_position_xyz": agent_position.tolist(),
                "sensor_rotation_matrix": sensor_rotation.tolist(),
            }
        )
    return mapper, aligned_camera


def _build_oracle_mapper(frames, grid_cfg, metadata, args) -> DenseBEVMapper:
    mapper = DenseBEVMapper(origin_world_xz=grid_cfg["origin"], config=grid_cfg["config"])
    for frame in frames:
        depth = np.load(frame["depth_npy"]).astype(np.float32)
        mapper.update_from_depth(
            depth=depth,
            agent_position_xyz=frame["agent_position_xyz"],
            sensor_position_xyz=frame["sensor_position_xyz"],
            sensor_rotation=MatrixRotation(np.asarray(frame["sensor_rotation_matrix"], dtype=np.float32)),
            hfov_deg=float(metadata.get("hfov_deg", 90.0)),
        )
    return mapper


def _reproject_detections_with_vggt(frames, detections_json: Path, prediction, align, aligned_camera, context_id, merge_radius_m, min_track_confidence):
    payload = _read_json(detections_json)
    detections = payload.get("detections", payload.get("items", []))
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for det in detections:
        by_frame.setdefault(int(det["frame_index"]), []).append(det)

    tracks: list[Track] = []
    debug = []
    scale = float(align["scale"])
    align_rot = np.asarray(align["rotation"], dtype=np.float32)
    translations = np.asarray(align["translation"], dtype=np.float32)
    for seq_idx, frame in enumerate(frames):
        frame_index = int(frame["frame_index"])
        image = Image.open(frame["rgb_path"])
        src_w, src_h = image.size
        depth = np.asarray(prediction["depth"][seq_idx], dtype=np.float32)
        conf = np.asarray(prediction["depth_conf"][seq_idx], dtype=np.float32)
        intr = np.asarray(prediction["intrinsic"][seq_idx], dtype=np.float32)
        cam_center = np.asarray(prediction["camera_centers"][seq_idx], dtype=np.float32)
        cam_rot_cv = np.asarray(prediction["camera_rotations"][seq_idx], dtype=np.float32)
        for det in by_frame.get(frame_index, []):
            projected = _project_box_vggt(det, depth, conf, intr, cam_center, cam_rot_cv, align_rot, translations, scale, src_w, src_h)
            if projected is None:
                debug.append({"frame_index": frame_index, "label": det.get("label"), "status": "rejected", "box": det.get("box")})
                continue
            merged = {**det, **projected, "frame_index": frame_index, "source": "owlv2_box_vggt_geometry"}
            _merge_detection(tracks, merged, merge_radius_m=float(merge_radius_m))
            debug.append({"frame_index": frame_index, "label": det.get("label"), "status": "projected", **projected})

    candidates = []
    for index, track in enumerate(tracks):
        confidence = float(track.confidence_sum / max(1e-6, track.weight_sum))
        if confidence < float(min_track_confidence):
            continue
        centroid = track.centroid
        candidates.append(
            GroundingCandidate(
                id=f"vggt_owlv2_{track.label}_{index:03d}",
                label=track.label,
                position_3d=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                confidence=max(0.0, min(1.0, confidence)),
                context_id=context_id,
                source="owlv2_boxes_vggt_depth_pose",
                source_view_ids=[f"frame_{int(det['frame_index']):04d}" for det in track.detections],
                bbox=_best_bbox(track.detections),
                raw={"num_detections": len(track.detections)},
            )
        )
    return candidates, debug


def _project_box_vggt(det, depth, conf, intrinsic, cam_center, cam_rot_cv, align_rot, translation, scale, src_w, src_h):
    h, w = depth.shape
    box = det.get("box")
    if not box:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = int(round(x1 / max(1, src_w) * w))
    x2 = int(round(x2 / max(1, src_w) * w))
    y1 = int(round(y1 / max(1, src_h) * h))
    y2 = int(round(y2 / max(1, src_h) * h))
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    pad_x = int((x2 - x1) * 0.2)
    pad_y = int((y2 - y1) * 0.2)
    crop = depth[y1 + pad_y : max(y1 + pad_y + 1, y2 - pad_y), x1 + pad_x : max(x1 + pad_x + 1, x2 - pad_x)]
    crop_conf = conf[y1 + pad_y : max(y1 + pad_y + 1, y2 - pad_y), x1 + pad_x : max(x1 + pad_x + 1, x2 - pad_x)]
    valid = np.isfinite(crop) & (crop > 0.05) & (crop_conf > 1.0)
    if valid.sum() < 8:
        return None
    rows, cols = np.where(valid)
    values = crop[rows, cols]
    low, high = np.percentile(values, [10, 70])
    keep = (values >= low) & (values <= high)
    rows = rows[keep] + y1 + pad_y
    cols = cols[keep] + x1 + pad_x
    z = depth[rows, cols].astype(np.float32)
    if rows.size < 8:
        return None
    stride = max(1, int(math.sqrt(rows.size / 180.0)))
    rows, cols, z = rows[::stride], cols[::stride], z[::stride]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x_cam = (cols.astype(np.float32) - cx) / fx * z
    y_cam = (rows.astype(np.float32) - cy) / fy * z
    cv_points = np.stack([x_cam, y_cam, z], axis=1)
    vggt_world = cam_center.reshape(1, 3) + cv_points @ cam_rot_cv.T
    habitat_world = scale * (vggt_world @ align_rot.T) + translation.reshape(1, 3)
    centroid = np.median(habitat_world, axis=0)
    if not np.isfinite(centroid).all():
        return None
    return {
        "position_3d": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
        "depth_median": float(np.median(z) * scale),
        "projected_points": int(habitat_world.shape[0]),
    }


def _merge_detection(tracks: list[Track], det: dict[str, Any], merge_radius_m: float) -> None:
    label = str(det.get("label", "")).lower()
    if not label:
        return
    position = np.asarray(det["position_3d"], dtype=np.float32)
    best = None
    best_dist = math.inf
    for track in tracks:
        if track.label != label:
            continue
        dist = float(np.linalg.norm(track.centroid[[0, 2]] - position[[0, 2]]))
        if dist < best_dist:
            best, best_dist = track, dist
    score = float(det.get("score", det.get("confidence", 0.1)))
    weight = max(0.05, score)
    if best is not None and best_dist <= float(merge_radius_m):
        best.position_sum += position * weight
        best.confidence_sum += score * weight
        best.weight_sum += weight
        best.detections.append(det)
    else:
        tracks.append(Track(label=label, position_sum=position * weight, confidence_sum=score * weight, weight_sum=weight, detections=[det]))


def _save_geometry_outputs(out_dir: Path, frames, prediction, align, args) -> None:
    np.savez_compressed(
        out_dir / "vggt_predictions_geometry.npz",
        extrinsic=prediction["extrinsic"],
        intrinsic=prediction["intrinsic"],
        depth=prediction["depth"],
        depth_conf=prediction["depth_conf"],
        camera_centers=prediction["camera_centers"],
        camera_rotations=prediction["camera_rotations"],
        alignment=json.dumps(align),
    )
    points = _transform_points(prediction["points"], align)
    conf = prediction["depth_conf"]
    sampled = []
    stride = max(1, int(args.point_sample_stride))
    for idx in range(points.shape[0]):
        frame_points = points[idx, ::stride, ::stride].reshape(-1, 3)
        frame_conf = conf[idx, ::stride, ::stride].reshape(-1)
        mask = np.isfinite(frame_points).all(axis=1) & (frame_conf >= float(args.depth_conf_threshold))
        sampled.append(frame_points[mask])
    all_points = np.concatenate(sampled, axis=0) if sampled else np.empty((0, 3), dtype=np.float32)
    _write_ascii_pcd(out_dir / "vggt_pointcloud_aligned.pcd", all_points)
    (out_dir / "selected_frames.json").write_text(json.dumps(_json_safe_frames(frames), indent=2), encoding="utf-8")
    (out_dir / "alignment.json").write_text(json.dumps(align, indent=2), encoding="utf-8")


def _write_ascii_pcd(path: Path, points: np.ndarray, max_points: int = 180000) -> None:
    points = np.asarray(points, dtype=np.float32)
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[idx]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# .PCD v0.7 - Point Cloud Data file format\n")
        handle.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        handle.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        for x, y, z in points:
            handle.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")


def _grid_config(metadata: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
    fit = metadata.get("route_plan", {}).get("bev_fit", {})
    if fit:
        grid_size = tuple(int(v) for v in fit.get("grid_size", [574, 574]))
        origin = tuple(float(v) for v in fit.get("origin_world_xz", [-17.7, -15.6]))
        resolution = float(fit.get("resolution", 0.05))
    else:
        centers = np.asarray([frame["sensor_position_xyz"] for frame in frames], dtype=float)
        mins = centers[:, [0, 2]].min(axis=0) - 7.0
        maxs = centers[:, [0, 2]].max(axis=0) + 7.0
        resolution = 0.05
        size = int(math.ceil(max(maxs - mins) / resolution))
        grid_size = (size, size)
        origin = (float(mins[0]), float(mins[1]))
    return {
        "origin": origin,
        "config": DenseBEVConfig(
            grid_size=grid_size,
            resolution=resolution,
            sample_stride=4,
            max_depth_m=7.0,
            obstacle_dilation_radius_cells=2,
        ),
    }


def _save_bev_state(path: Path, mapper: DenseBEVMapper, metadata: dict[str, Any]) -> Path:
    np.savez_compressed(
        path,
        occupancy_logodds=mapper.occupancy_logodds.astype(np.float32),
        explored=mapper.explored.astype(bool),
        trajectory=np.asarray(mapper.trajectory, dtype=np.int32),
        metadata=json.dumps(
            {
                "grid_size": list(mapper.config.grid_size),
                "resolution": mapper.config.resolution,
                "origin_world_xz": list(mapper.origin_world_xz),
                "semantic_categories": metadata.get("semantic_categories", ["wall", "door", "table", "chair", "bed", "sofa"]),
            }
        ),
    )
    return path


def _compare_bev(vggt_mapper: DenseBEVMapper, oracle_mapper: DenseBEVMapper) -> dict[str, Any]:
    v_free = vggt_mapper.free_mask()
    o_free = oracle_mapper.free_mask()
    v_occ = vggt_mapper.occupied_mask()
    o_occ = oracle_mapper.occupied_mask()
    return {
        "explored_iou": _iou(vggt_mapper.explored, oracle_mapper.explored),
        "free_iou": _iou(v_free, o_free),
        "occupied_iou": _iou(v_occ, o_occ),
        "vggt_explored_cells": int(vggt_mapper.explored.sum()),
        "oracle_explored_cells": int(oracle_mapper.explored.sum()),
        "vggt_occupied_cells": int(v_occ.sum()),
        "oracle_occupied_cells": int(o_occ.sum()),
    }


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _run_bridge(candidates_path: Path, pointcloud_pcd: Path, bridge_dir: Path, args) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "m2_nlmap_bev_bridge_eval.py"),
        "--objects-json",
        str(candidates_path),
        "--pointcloud-pcd",
        str(pointcloud_pcd),
        "--out-dir",
        str(bridge_dir),
        "--context-id",
        args.context_id,
        "--scene-id",
        args.scene_id,
        "--resolution-m",
        "0.10",
        "--padding-m",
        "1.0",
        "--semantic-radius-m",
        str(args.semantic_radius_m),
    ]
    subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=True)
    return _read_json(bridge_dir / "bridge_metadata.json")


def _run_semantic_mvp(bev_state: Path, candidates_json: Path, metrics_json: Path, rgb: Path, overlay: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "m25_make_semantic_bev_mvp_evidence.py"),
        "--bev-state",
        str(bev_state),
        "--candidates-json",
        str(candidates_json),
        "--metrics-json",
        str(metrics_json),
        "--rgb",
        str(rgb),
        "--owlv2-overlay",
        str(overlay),
        "--out-dir",
        str(out_dir),
        "--scale",
        "2",
    ]
    subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=True)


def _render_bev(mapper: DenseBEVMapper, path: Path, title: str) -> Path:
    scale = max(1, min(3, int(1100 / max(mapper.config.grid_size))))
    state = mapper.occupancy_state()
    img = Image.new("RGB", (mapper.config.grid_size[0] * scale, mapper.config.grid_size[1] * scale), "#d9d9d9")
    pix = img.load()
    colors = {0: (217, 217, 217), 1: (255, 255, 255), 2: (45, 45, 45)}
    for gx in range(mapper.config.grid_size[0]):
        for gy in range(mapper.config.grid_size[1]):
            color = colors[int(state[gx, gy])]
            px = gx * scale
            py = (mapper.config.grid_size[1] - 1 - gy) * scale
            for dx in range(scale):
                for dy in range(scale):
                    pix[px + dx, py + dy] = color
    draw = ImageDraw.Draw(img)
    if len(mapper.trajectory) > 1:
        pts = [(x * scale + scale / 2, (mapper.config.grid_size[1] - 1 - y) * scale + scale / 2) for x, y in mapper.trajectory]
        draw.line(pts, fill="#1f77b4", width=max(2, scale * 2))
    draw.rectangle((0, 0, min(img.width, 640), 28), fill="white")
    draw.text((8, 7), title, fill="#111111", font=_font(14))
    img.save(path)
    return path


def _render_depth_contact_sheet(out_dir: Path, frames, prediction) -> Path:
    chosen = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
    panels = []
    for idx in chosen:
        rgb = Image.open(frames[int(idx)]["rgb_path"]).convert("RGB").resize((220, 220))
        depth = prediction["depth"][int(idx)]
        depth_img = _depth_to_image(depth).resize((220, 220))
        panels.extend([rgb, depth_img])
    sheet = Image.new("RGB", (440, 220 * len(chosen)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, idx in enumerate(chosen):
        sheet.paste(panels[row * 2], (0, row * 220))
        sheet.paste(panels[row * 2 + 1], (220, row * 220))
        draw.text((8, row * 220 + 8), f"frame {frames[int(idx)]['frame_index']:04d}", fill="white", font=_font(14))
    path = out_dir / "vggt_depth_contact_sheet.jpg"
    sheet.save(path, quality=92)
    return path


def _depth_to_image(depth: np.ndarray) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return Image.new("RGB", depth.shape[::-1], "black")
    lo, hi = np.percentile(depth[valid], [2, 95])
    norm = np.clip((depth - lo) / max(1e-6, hi - lo), 0, 1)
    arr = (255 * (1 - norm)).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _write_report(out_dir: Path, metrics: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in {
            "Status": metrics["status"],
            "Frames": metrics["num_frames"],
            "VGGT ATE RMSE": f"{metrics['alignment']['ate_rmse_m']:.3f} m",
            "BEV free IoU": f"{metrics['bev_comparison']['free_iou']:.3f}",
            "BEV occupied IoU": f"{metrics['bev_comparison']['occupied_iou']:.3f}",
            "Semantic candidates": metrics["num_candidates"],
            "Oracle F1": f"{metrics['validation'].get('f1', 0.0):.3f}",
            "Bridge": metrics.get("bridge", {}).get("status", "unknown"),
        }.items()
    )
    selected = ", ".join(str(int(frame["frame_index"])) for frame in frames)
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>M3 VGGT Geometry Frontend Validation</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; align-items: start; }}
img {{ max-width: 100%; border: 1px solid #d7dce2; border-radius: 6px; background: white; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #d7dce2; padding: 7px 9px; text-align: left; }}
th {{ background: #f4f6f8; }}
code {{ background: #f1f3f4; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>M3 VGGT Geometry Frontend Validation</h1>
<p>Pipeline: Habitat coverage-loop RGB sequence -> VGGT depth/pose/pointcloud -> Sim(3) alignment to Habitat world -> DenseBEVMapper -> OWLv2 box reprojection with VGGT geometry -> semantic BEV MVP -> oracle comparison.</p>
<table>{rows}</table>
<p><strong>Selected frame indices:</strong> {html.escape(selected)}</p>
<section class="grid">
  <div><h2>Oracle RGB-D/Pose BEV</h2><img src="oracle_bev_selected.png"></div>
  <div><h2>VGGT Estimated BEV</h2><img src="vggt_traditional_bev.png"></div>
  <div><h2>VGGT Depth Samples</h2><img src="vggt_depth_contact_sheet.jpg"></div>
</section>
<section class="grid">
  <div><h2>Semantic BEV MVP</h2><a href="rgb_to_semantic_bev_mvp/rgb_to_semantic_bev_mvp.html"><img src="rgb_to_semantic_bev_mvp/rgb_to_semantic_bev_mvp_pipeline.png"></a></div>
  <div><h2>M2 Bridge</h2><a href="bridge/bridge_report.html"><img src="bridge/occupancy_bev.png"></a></div>
  <div><h2>Phase3 Retrieval</h2><a href="bridge/phase3_retrieval/retrieval_report.html"><img src="bridge/phase3_retrieval/retrieval_bev.svg"></a></div>
</section>
<h2>Artifacts</h2>
<ul>
  <li><a href="m3_vggt_metrics.json">m3_vggt_metrics.json</a></li>
  <li><a href="vggt_predictions_geometry.npz">vggt_predictions_geometry.npz</a></li>
  <li><a href="vggt_pointcloud_aligned.pcd">vggt_pointcloud_aligned.pcd</a></li>
  <li><a href="vggt_grounding_candidates.json">vggt_grounding_candidates.json</a></li>
</ul>
</body>
</html>
"""
    (out_dir / "m3_vggt_geometry_report.html").write_text(report, encoding="utf-8")


def _select_frames(frames: list[dict[str, Any]], max_frames: int, stride: int, mode: str) -> list[dict[str, Any]]:
    subset = frames[:: max(1, int(stride))]
    if mode == "first":
        subset = subset[: max(1, int(max_frames))]
    elif len(subset) > max_frames:
        idx = np.linspace(0, len(subset) - 1, int(max_frames), dtype=int)
        subset = [subset[int(i)] for i in idx]
    return [dict(frame) for frame in subset]


def _localize_frame_paths(frames: list[dict[str, Any]], metadata_path: Path) -> None:
    frames_dir = metadata_path.parent / "frames"
    for frame in frames:
        for key in ["rgb_path", "depth_npy", "depth_png", "semantic_npy"]:
            if key not in frame:
                continue
            path = Path(frame[key])
            if not path.exists():
                path = frames_dir / Path(frame[key]).name
            frame[key] = str(path.resolve())


def _prepare_input_images(frames: list[dict[str, Any]], image_dir: Path) -> None:
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    for seq_idx, frame in enumerate(frames):
        shutil.copy2(frame["rgb_path"], image_dir / f"{seq_idx:04d}_frame_{int(frame['frame_index']):04d}.jpg")


def _oracle_sensor_centers(frames: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([frame["sensor_position_xyz"] for frame in frames], dtype=np.float64)


def _json_safe_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["frame_index", "action", "rgb_path", "depth_npy", "sensor_position_xyz", "agent_position_xyz"]
    return [{key: frame.get(key) for key in keys if key in frame} for frame in frames]


def _best_overlay_for_frame(detections_json: Path, frame_index: int) -> Path | None:
    candidate = detections_json.parent / "overlays" / f"frame_{int(frame_index):04d}_overlay.jpg"
    return candidate if candidate.exists() else None


def _best_bbox(detections: list[dict[str, Any]]) -> dict[str, float] | None:
    if not detections:
        return None
    best = max(detections, key=lambda item: float(item.get("score", 0.0)))
    box = best.get("box")
    if not box:
        return None
    return {"x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3])}


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
