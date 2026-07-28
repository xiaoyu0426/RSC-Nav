from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CupConfirmationConfig:
    min_task_views: int = 2
    min_visual_passes: int = 2
    min_visual_negatives: int = 2
    min_depth_relief_passes: int = 2
    min_depth_relief_m: float = 0.025
    max_position_spread_m: float = 0.30
    independent_translation_m: float = 0.35


def append_independent_observation(
    observations: list[dict[str, Any]],
    observation: dict[str, Any],
    config: CupConfirmationConfig,
) -> bool:
    signature = _signature(observation)
    for existing in observations:
        if str(existing.get("crop_verifier_status", "")) == "error":
            continue
        other = _signature(existing)
        translation = math.hypot(signature[0] - other[0], signature[1] - other[1])
        if translation < float(config.independent_translation_m):
            return False
    observations.append(observation)
    return True


def evaluate_cup_confirmation(
    observations: list[dict[str, Any]],
    config: CupConfirmationConfig,
) -> dict[str, Any]:
    evidence = [
        item
        for item in observations
        if str(item.get("crop_verifier_status", "")) != "error"
    ]
    positions = [_position(item) for item in evidence]
    raw_position_spread_m = _max_pairwise_distance(positions)
    inlier_indices = _largest_consistent_subset(
        positions,
        max_distance_m=float(config.max_position_spread_m),
    )
    consistent_evidence = [evidence[index] for index in inlier_indices]
    consistent_positions = [positions[index] for index in inlier_indices]
    position_spread_m = _max_pairwise_distance(consistent_positions)
    visual_passes = sum(
        str(item.get("crop_verifier_status", "")) == "pass"
        or bool(item.get("crop_verifier_pass"))
        for item in consistent_evidence
    )
    visual_negatives = sum(
        str(item.get("crop_verifier_status", "")) == "negative"
        for item in consistent_evidence
    )
    visual_inconclusive = sum(
        str(item.get("crop_verifier_status", "")) == "inconclusive"
        for item in consistent_evidence
    )
    verifier_errors = sum(
        str(item.get("crop_verifier_status", "")) == "error"
        for item in observations
    )
    depth_relief_values = [
        float(item["depth_surface_relief_m"])
        for item in consistent_evidence
        if item.get("depth_surface_relief_m") is not None
    ]
    depth_relief_passes = sum(
        value >= float(config.min_depth_relief_m)
        for value in depth_relief_values
    )
    task_views = len(evidence)

    if task_views == 0:
        status = "not_observed"
    elif task_views < int(config.min_task_views):
        status = "insufficient_task_views"
    elif len(consistent_evidence) < int(config.min_task_views):
        status = "rejected_geometry_inconsistent"
    elif len(depth_relief_values) < int(config.min_depth_relief_passes):
        status = "insufficient_depth_evidence"
    elif depth_relief_passes < int(config.min_depth_relief_passes):
        status = "rejected_planar_surface"
    elif (
        visual_passes >= int(config.min_visual_passes)
        and visual_negatives >= int(config.min_visual_negatives)
    ):
        status = "conflicting_visual_evidence"
    elif visual_passes >= int(config.min_visual_passes):
        status = "verified"
    elif visual_negatives >= int(config.min_visual_negatives):
        status = "rejected_visual_verifier"
    else:
        status = "insufficient_visual_evidence"

    positive_scores = [
        float(item.get("crop_positive_score", 0.0))
        for item in consistent_evidence
    ]
    negative_scores = [
        float(item.get("crop_negative_score", 0.0))
        for item in consistent_evidence
    ]
    return {
        "status": status,
        "verified": status == "verified",
        "task_independent_views": task_views,
        "geometry_inlier_views": len(consistent_evidence),
        "visual_passes": visual_passes,
        "visual_negatives": visual_negatives,
        "visual_inconclusive": visual_inconclusive,
        "verifier_errors": verifier_errors,
        "depth_relief_passes": depth_relief_passes,
        "mean_depth_surface_relief_m": _mean(depth_relief_values),
        "position_spread_m": float(position_spread_m),
        "raw_position_spread_m": float(raw_position_spread_m),
        "mean_crop_positive_score": _mean(positive_scores),
        "mean_crop_negative_score": _mean(negative_scores),
        "evidence_steps": [
            int(item.get("step", -1)) for item in consistent_evidence
        ],
        "crop_paths": [
            str(item["crop_path"])
            for item in observations
            if item.get("crop_path")
        ],
    }


def estimate_depth_surface_relief(
    depth: np.ndarray,
    box: list[float],
    *,
    padding_ratio: float = 0.40,
) -> dict[str, Any]:
    if depth.ndim != 2 or len(box) != 4:
        return {"valid": False, "relief_m": None}
    height, width = depth.shape
    x1, y1, x2, y2 = [float(value) for value in box]
    ix1 = max(0, min(width - 1, int(math.floor(x1))))
    iy1 = max(0, min(height - 1, int(math.floor(y1))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(x2))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(y2))))
    box_width = max(1, ix2 - ix1)
    box_height = max(1, iy2 - iy1)
    pad_x = max(2, int(round(float(padding_ratio) * box_width)))
    pad_y = max(2, int(round(float(padding_ratio) * box_height)))
    ox1 = max(0, ix1 - pad_x)
    oy1 = max(0, iy1 - pad_y)
    ox2 = min(width, ix2 + pad_x)
    oy2 = min(height, iy2 + pad_y)

    inner = np.asarray(depth[iy1:iy2, ix1:ix2], dtype=np.float32)
    outer = np.asarray(depth[oy1:oy2, ox1:ox2], dtype=np.float32)
    ring_mask = np.ones(outer.shape, dtype=bool)
    ring_mask[
        iy1 - oy1 : iy2 - oy1,
        ix1 - ox1 : ix2 - ox1,
    ] = False
    inner_valid = inner[np.isfinite(inner) & (inner > 0.05)]
    ring_values = outer[ring_mask]
    ring_valid = ring_values[
        np.isfinite(ring_values) & (ring_values > 0.05)
    ]
    if inner_valid.size < 8 or ring_valid.size < 12:
        return {
            "valid": False,
            "relief_m": None,
            "inner_valid_pixels": int(inner_valid.size),
            "ring_valid_pixels": int(ring_valid.size),
        }
    foreground_depth_m = float(np.percentile(inner_valid, 25.0))
    background_depth_m = float(np.median(ring_valid))
    return {
        "valid": True,
        "relief_m": background_depth_m - foreground_depth_m,
        "foreground_depth_m": foreground_depth_m,
        "background_depth_m": background_depth_m,
        "inner_valid_pixels": int(inner_valid.size),
        "ring_valid_pixels": int(ring_valid.size),
    }


def score_crop_verifier(
    *,
    detections: list[dict[str, Any]],
    positive_labels: set[str],
    target_box: list[float],
    min_positive_score: float,
    min_score_margin: float,
) -> dict[str, Any]:
    normalized_positive = {
        str(label).strip().lower() for label in positive_labels
    }
    positive_score = 0.0
    negative_score = 0.0
    associated: list[dict[str, Any]] = []
    for detection in detections:
        box = detection.get("box", ())
        if len(box) != 4 or not _box_matches_target(box, target_box):
            continue
        label = str(detection.get("label", "")).strip().lower()
        score = float(detection.get("score", 0.0))
        associated.append(
            {
                "label": label,
                "score": score,
                "box": [float(value) for value in box],
            }
        )
        if label in normalized_positive:
            positive_score = max(positive_score, score)
        else:
            negative_score = max(negative_score, score)

    if (
        positive_score >= float(min_positive_score)
        and positive_score >= negative_score + float(min_score_margin)
    ):
        status = "pass"
    elif (
        negative_score >= float(min_positive_score)
        and negative_score >= positive_score + float(min_score_margin)
    ):
        status = "negative"
    else:
        status = "inconclusive"
    return {
        "crop_verifier_status": status,
        "crop_verifier_pass": status == "pass",
        "crop_positive_score": positive_score,
        "crop_negative_score": negative_score,
        "crop_score_margin": positive_score - negative_score,
        "crop_associated_detections": associated,
        "crop_associated_detection_count": len(associated),
    }


def _signature(observation: dict[str, Any]) -> tuple[float, float, float]:
    values = observation.get("camera_xzyaw", ())
    if len(values) != 3:
        raise ValueError("confirmation observation requires camera_xzyaw")
    return float(values[0]), float(values[1]), float(values[2])


def _position(observation: dict[str, Any]) -> tuple[float, float, float]:
    values = observation.get("position_3d", ())
    if len(values) != 3:
        raise ValueError("confirmation observation requires position_3d")
    return float(values[0]), float(values[1]), float(values[2])


def _max_pairwise_distance(
    positions: list[tuple[float, float, float]],
) -> float:
    maximum = 0.0
    for index, first in enumerate(positions):
        for second in positions[index + 1 :]:
            distance = math.sqrt(
                (first[0] - second[0]) ** 2
                + (first[1] - second[1]) ** 2
                + (first[2] - second[2]) ** 2
            )
            maximum = max(maximum, distance)
    return maximum


def _largest_consistent_subset(
    positions: list[tuple[float, float, float]],
    *,
    max_distance_m: float,
) -> list[int]:
    for subset_size in range(len(positions), 0, -1):
        for indices in combinations(range(len(positions)), subset_size):
            subset = [positions[index] for index in indices]
            if _max_pairwise_distance(subset) <= float(max_distance_m):
                return list(indices)
    return []


def _box_matches_target(
    candidate_box: Any,
    target_box: Any,
) -> bool:
    candidate = [float(value) for value in candidate_box]
    target = [float(value) for value in target_box]
    candidate_center = (
        0.5 * (candidate[0] + candidate[2]),
        0.5 * (candidate[1] + candidate[3]),
    )
    target_width = max(1.0, target[2] - target[0])
    target_height = max(1.0, target[3] - target[1])
    expanded_target = [
        target[0] - 0.20 * target_width,
        target[1] - 0.20 * target_height,
        target[2] + 0.20 * target_width,
        target[3] + 0.20 * target_height,
    ]
    center_inside = (
        expanded_target[0] <= candidate_center[0] <= expanded_target[2]
        and expanded_target[1] <= candidate_center[1] <= expanded_target[3]
    )
    intersection_width = max(
        0.0,
        min(candidate[2], target[2]) - max(candidate[0], target[0]),
    )
    intersection_height = max(
        0.0,
        min(candidate[3], target[3]) - max(candidate[1], target[1]),
    )
    intersection = intersection_width * intersection_height
    candidate_area = max(
        1.0,
        (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]),
    )
    return center_inside or intersection / candidate_area >= 0.25


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
