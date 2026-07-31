#!/usr/bin/env python3
"""Deterministic offline audit for open-vocabulary boxes and online tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TARGET_CLASSES = ("door", "window")
AP_IOU_THRESHOLDS = (0.50, 0.75)
HARD_NEGATIVE_CATEGORIES = (
    "window",
    "cabinet door",
    "refrigerator door",
    "mirror",
    "wall panel",
)
XZ_HISTOGRAM_EDGES_M = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, math.inf)
XZ_DEPTH_STRATUM_EDGES_M = (0.0, 1.0, 2.0, 4.0, 6.0, math.inf)


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """Return IoU for two XYXY boxes."""
    intersection_w = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_h = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_w * intersection_h
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(
        0.0, right[3] - right[1]
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def interpolated_average_precision(
    tp_flags: Sequence[int],
    fp_flags: Sequence[int],
    gt_count: int,
) -> float | None:
    """Compute all-point interpolated AP from score-ordered decisions."""
    if gt_count <= 0:
        return None
    if len(tp_flags) != len(fp_flags):
        raise ValueError("tp_flags and fp_flags must have equal length")
    if not tp_flags:
        return 0.0

    recalls: list[float] = []
    precisions: list[float] = []
    tp_total = 0
    fp_total = 0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        tp_total += int(bool(tp_flag))
        fp_total += int(bool(fp_flag))
        recalls.append(tp_total / gt_count)
        precisions.append(tp_total / (tp_total + fp_total))

    recall_points = [0.0, *recalls, 1.0]
    precision_points = [0.0, *precisions, 0.0]
    for index in range(len(precision_points) - 2, -1, -1):
        precision_points[index] = max(
            precision_points[index], precision_points[index + 1]
        )

    average_precision = 0.0
    for index in range(len(recall_points) - 1):
        recall_delta = recall_points[index + 1] - recall_points[index]
        if recall_delta > 0.0:
            average_precision += recall_delta * precision_points[index + 1]
    return average_precision


def classwise_nms(
    detections: Sequence[Mapping[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply deterministic NMS independently per frame and canonical class."""
    if iou_threshold >= 1.0:
        return [dict(item) for item in detections], []
    if iou_threshold < 0.0:
        raise ValueError("nms_iou must be non-negative")

    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for detection in detections:
        grouped[
            (int(detection["frame_index"]), str(detection["canonical_label"]))
        ].append(detection)

    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        group = sorted(
            grouped[group_key],
            key=lambda item: (-float(item["score"]), int(item["_source_index"])),
        )
        group_kept: list[Mapping[str, Any]] = []
        for detection in group:
            if any(
                box_iou(detection["box"], accepted["box"]) > iou_threshold
                for accepted in group_kept
            ):
                suppressed.append(dict(detection))
            else:
                group_kept.append(detection)
                kept.append(dict(detection))

    kept.sort(key=lambda item: int(item["_source_index"]))
    suppressed.sort(key=lambda item: int(item["_source_index"]))
    return kept, suppressed


def audit_payloads(
    detections_payload: Mapping[str, Any],
    semantic_gt_payload: Mapping[str, Any],
    *,
    score_threshold: float = 0.0,
    match_iou: float = 0.50,
    nms_iou: float = 1.0,
) -> dict[str, Any]:
    """Audit baseline detections and an optional canonical-class NMS variant."""
    _validate_threshold("score_threshold", score_threshold, lower=0.0, upper=1.0)
    _validate_threshold("match_iou", match_iou, lower=0.0, upper=1.0)
    if not math.isfinite(nms_iou) or nms_iou < 0.0:
        raise ValueError("nms_iou must be a finite non-negative number")

    raw_detections = detections_payload.get("detections")
    raw_frames = semantic_gt_payload.get("frames")
    if not isinstance(raw_detections, list):
        raise ValueError("detections.json must contain a detections list")
    if not isinstance(raw_frames, list):
        raise ValueError("semantic_gt.json must contain a frames list")

    detections, detection_issues = _normalize_detections(raw_detections)
    (
        ground_truth,
        gt_issues,
        raw_gt_instances,
        evaluated_frames,
    ) = _normalize_ground_truth(raw_frames)
    evaluated_detections = [
        item
        for item in detections
        if item["frame_index"] in evaluated_frames
    ]
    target_detections = [
        item
        for item in evaluated_detections
        if item["canonical_label"] in TARGET_CLASSES
    ]
    target_ground_truth = [
        item for item in ground_truth if item["canonical_label"] in TARGET_CLASSES
    ]

    variants: dict[str, Any] = {
        "baseline": _evaluate_variant(
            target_detections,
            target_ground_truth,
            ground_truth,
            evaluated_frame_count=len(evaluated_frames),
            score_threshold=score_threshold,
            match_iou=match_iou,
            online_tracks_causal=True,
        )
    }
    nms_enabled = nms_iou < 1.0
    nms_summary: dict[str, Any] = {
        "enabled": nms_enabled,
        "iou_threshold": nms_iou,
        "input_evaluable_detections": len(evaluated_detections),
        "kept_evaluable_detections": len(evaluated_detections),
        "suppressed_evaluable_detections": 0,
        "suppressed_target_detections": 0,
        "suppressed_by_canonical_class": {},
    }
    if nms_enabled:
        nms_kept, nms_suppressed = classwise_nms(
            evaluated_detections, nms_iou
        )
        nms_target = [
            item
            for item in nms_kept
            if item["canonical_label"] in TARGET_CLASSES
        ]
        variants["class_nms"] = _evaluate_variant(
            nms_target,
            target_ground_truth,
            ground_truth,
            evaluated_frame_count=len(evaluated_frames),
            score_threshold=score_threshold,
            match_iou=match_iou,
            online_tracks_causal=False,
        )
        suppressed_counts = Counter(
            str(item["canonical_label"]) for item in nms_suppressed
        )
        nms_summary.update(
            {
                "kept_evaluable_detections": len(nms_kept),
                "suppressed_evaluable_detections": len(nms_suppressed),
                "suppressed_target_detections": sum(
                    suppressed_counts[label] for label in TARGET_CLASSES
                ),
                "suppressed_by_canonical_class": dict(
                    sorted(suppressed_counts.items())
                ),
            }
        )

    unavailable_metrics = []
    for variant_name, variant in variants.items():
        for item in variant["unavailable_metrics"]:
            unavailable_metrics.append(
                {
                    "metric": f"variants.{variant_name}.{item['metric']}",
                    "reason": item["reason"],
                }
            )

    coverage = _coverage_report(
        raw_detections=raw_detections,
        raw_frames=raw_frames,
        raw_gt_instances=raw_gt_instances,
        detections=detections,
        evaluated_detections=evaluated_detections,
        target_detections=target_detections,
        ground_truth=ground_truth,
        target_ground_truth=target_ground_truth,
        evaluated_frames=evaluated_frames,
        detection_issues=detection_issues,
        gt_issues=gt_issues,
    )
    parameters = {
            "target_classes": list(TARGET_CLASSES),
            "box_format": "xyxy",
            "score_threshold": score_threshold,
            "fixed_operating_point_iou": match_iou,
            "ap_iou_thresholds": list(AP_IOU_THRESHOLDS),
            "ap_method": "all_point_interpolated_precision_envelope",
            "nms": {
                **nms_summary,
                "scope": "within_frame_and_canonical_class",
                "suppression_rule": "IoU strictly greater than nms_iou",
                "disabled_when": "nms_iou >= 1",
            },
            "label_policy": (
                "canonical_label when non-empty, otherwise label; "
                "lowercase and trim only"
            ),
            "ground_truth_policy": (
                "semantic_gt instances are the only truth source; "
                "VLM verdicts are never read"
            ),
            "matching_policy": (
                "greedy score-ordered one-to-one matching within frame and "
                "canonical class"
            ),
            "evaluated_frame_policy": (
                "unique semantic_gt frames with a valid frame_index and an "
                "instances list; detections on other frames are excluded"
            ),
            "frame_normalized_rate_policy": (
                "count / evaluated_frame_count * 100"
            ),
            "physical_instance_recall_policy": (
                "unique canonical-target semantic_id values matched at the "
                "fixed operating point / all unique canonical-target "
                "semantic_id values; unavailable if any target GT annotation "
                "lacks semantic_id"
            ),
            "tp_iou_distribution_policy": (
                "IoUs of fixed-operating-point one-to-one TP matches; "
                "quantiles use linear interpolation"
            ),
            "track_policy": (
                "associations use fixed-operating-point TP matches with both "
                "semantic_id and online_track_id; wrong_merge_rate is "
                "wrong-merge tracks / associated tracks; fragmentation "
                "tracks_per_gt is unique associated tracks / recalled GT"
            ),
            "xz_error_policy": (
                "Euclidean XZ distance for fixed-operating-point TP matches "
                "with valid detection position_3d and GT "
                "world_visible_center_xyz; legacy object-AABB centers are "
                "never used because merged semantic meshes can make them "
                "geometrically misleading"
            ),
            "hard_negative_policy": {
                "door_fp_only": True,
                "categories": list(HARD_NEGATIVE_CATEGORIES),
                "source": "semantic_gt raw_category and box only",
                "target_exclusion": (
                    "instances whose canonical_label is door are excluded"
                ),
                "attribution": (
                    "highest-IoU recognized non-door GT in the same frame, "
                    "requiring IoU >= fixed_operating_point_iou"
                ),
                "raw_category_never_creates_target_gt": True,
            },
            "variant_causality": {
                "baseline": "recorded online detections and track IDs",
                "class_nms": (
                    "posthoc detector-only counterfactual; track IDs were "
                    "assigned by the baseline before NMS and are not scored"
                ),
            },
        }
    evaluation_contract = {
        key: value
        for key, value in parameters.items()
        if key not in {"nms", "variant_causality"}
    }
    variant_algorithm_contracts: dict[str, dict[str, Any]] = {
        "baseline": {
            "operation": "identity",
            "input": "recorded online detections and track IDs",
            "causal_track_metrics": True,
        }
    }
    if "class_nms" in variants:
        variant_algorithm_contracts["class_nms"] = {
            "operation": "canonical_label_classwise_greedy_nms",
            "nms_iou": float(nms_iou),
            "scope": "within_frame_and_canonical_class",
            "suppression_rule": "IoU strictly greater than nms_iou",
            "input_order": "descending_score_then_original_index",
            "causal_track_metrics": False,
        }
    variant_algorithms = {
        name: {
            "contract": contract,
            "sha256": _canonical_payload_sha256(contract),
        }
        for name, contract in variant_algorithm_contracts.items()
    }
    return {
        "schema_version": "grounding_box_audit_v1",
        "parameters": parameters,
        "evaluation_contract": evaluation_contract,
        "evaluation_parameters_sha256": _canonical_payload_sha256(
            evaluation_contract
        ),
        "variant_algorithms": variant_algorithms,
        "coverage": coverage,
        "variants": variants,
        "unavailable_metrics": unavailable_metrics,
    }


def _evaluate_variant(
    detections: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    all_ground_truth: Sequence[Mapping[str, Any]],
    *,
    evaluated_frame_count: int,
    score_threshold: float,
    match_iou: float,
    online_tracks_causal: bool,
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    unavailable: list[dict[str, str]] = []
    fixed_matches: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = total_duplicate_fp = 0
    total_predictions = 0
    total_gt = 0
    door_fp_decisions: list[Mapping[str, Any]] = []

    for class_name in TARGET_CLASSES:
        class_detections = [
            item for item in detections if item["canonical_label"] == class_name
        ]
        class_gt = [
            item
            for item in ground_truth
            if item["canonical_label"] == class_name
        ]
        class_result: dict[str, Any] = {
            "ground_truth_instances": len(class_gt),
            "predictions_all_scores": len(class_detections),
        }
        for iou_threshold in AP_IOU_THRESHOLDS:
            decisions, _ = _match_detections(
                class_detections, class_gt, iou_threshold
            )
            ap_value = interpolated_average_precision(
                [item["tp"] for item in decisions],
                [item["fp"] for item in decisions],
                len(class_gt),
            )
            metric_name = f"ap{int(iou_threshold * 100)}"
            class_result[metric_name] = ap_value
            if ap_value is None:
                unavailable.append(
                    {
                        "metric": f"per_class.{class_name}.{metric_name}",
                        "reason": (
                            f"semantic_gt has no valid {class_name} instances"
                        ),
                    }
                )

        operating_detections = [
            item
            for item in class_detections
            if float(item["score"]) >= score_threshold
        ]
        decisions, matches = _match_detections(
            operating_detections, class_gt, match_iou
        )
        tp = sum(item["tp"] for item in decisions)
        fp = sum(item["fp"] for item in decisions)
        duplicate_fp = sum(item["duplicate_fp"] for item in decisions)
        fn = max(0, len(class_gt) - tp)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / len(class_gt) if class_gt else None
        tp_iou_distribution = _tp_iou_report(matches)
        if not tp_iou_distribution["available"]:
            unavailable.append(
                {
                    "metric": (
                        f"per_class.{class_name}.operating_point."
                        "tp_iou_distribution"
                    ),
                    "reason": (
                        f"no fixed-operating-point {class_name} TP matches"
                    ),
                }
            )
        physical_recall = _physical_instance_recall_report(
            matches,
            class_gt,
            metric_path=(
                f"per_class.{class_name}.operating_point."
                "physical_instance_recall"
            ),
            unavailable=unavailable,
        )
        class_result["operating_point"] = {
            "predictions": len(operating_detections),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "duplicate_fp": duplicate_fp,
            "precision": precision,
            "recall": recall,
            "evaluated_frames": evaluated_frame_count,
            "fp_per_100_evaluated_frames": _per_100_frames(
                fp, evaluated_frame_count
            ),
            "duplicate_fp_per_100_evaluated_frames": _per_100_frames(
                duplicate_fp, evaluated_frame_count
            ),
            "tp_iou_distribution": tp_iou_distribution,
            "physical_instance_recall": physical_recall,
        }
        if evaluated_frame_count == 0:
            unavailable.extend(
                [
                    {
                        "metric": (
                            f"per_class.{class_name}.operating_point."
                            "fp_per_100_evaluated_frames"
                        ),
                        "reason": "semantic_gt has no evaluated frames",
                    },
                    {
                        "metric": (
                            f"per_class.{class_name}.operating_point."
                            "duplicate_fp_per_100_evaluated_frames"
                        ),
                        "reason": "semantic_gt has no evaluated frames",
                    },
                ]
            )
        if precision is None:
            unavailable.append(
                {
                    "metric": (
                        f"per_class.{class_name}.operating_point.precision"
                    ),
                    "reason": (
                        f"no {class_name} predictions meet score_threshold"
                    ),
                }
            )
        if recall is None:
            unavailable.append(
                {
                    "metric": f"per_class.{class_name}.operating_point.recall",
                    "reason": (
                        f"semantic_gt has no valid {class_name} instances"
                    ),
                }
            )
        fixed_matches.extend(matches)
        if class_name == "door":
            door_fp_decisions = [
                item for item in decisions if int(item["fp"]) == 1
            ]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_duplicate_fp += duplicate_fp
        total_predictions += len(operating_detections)
        total_gt += len(class_gt)
        per_class[class_name] = class_result

    overall_precision = (
        total_tp / (total_tp + total_fp) if total_tp + total_fp else None
    )
    overall_recall = total_tp / total_gt if total_gt else None
    if overall_precision is None:
        unavailable.append(
            {
                "metric": "overall_operating_point.precision",
                "reason": "no target predictions meet score_threshold",
            }
        )
    if overall_recall is None:
        unavailable.append(
            {
                "metric": "overall_operating_point.recall",
                "reason": "semantic_gt has no valid target instances",
            }
        )

    overall_tp_iou = _tp_iou_report(fixed_matches)
    if not overall_tp_iou["available"]:
        unavailable.append(
            {
                "metric": "overall_operating_point.tp_iou_distribution",
                "reason": "no fixed-operating-point target TP matches",
            }
        )
    overall_physical_recall = _physical_instance_recall_report(
        fixed_matches,
        ground_truth,
        metric_path="overall_operating_point.physical_instance_recall",
        unavailable=unavailable,
    )
    if online_tracks_causal:
        track_association = _track_association_report(
            fixed_matches,
            ground_truth,
            unavailable,
        )
    else:
        track_association = _noncausal_track_report(unavailable)
    xz_error = _xz_error_report(fixed_matches, unavailable)
    hard_negative_door_fp = _hard_negative_door_fp_report(
        door_fp_decisions,
        all_ground_truth,
        evaluated_frame_count=evaluated_frame_count,
        attribution_iou=match_iou,
        unavailable=unavailable,
    )
    if evaluated_frame_count == 0:
        unavailable.extend(
            [
                {
                    "metric": (
                        "overall_operating_point."
                        "fp_per_100_evaluated_frames"
                    ),
                    "reason": "semantic_gt has no evaluated frames",
                },
                {
                    "metric": (
                        "overall_operating_point."
                        "duplicate_fp_per_100_evaluated_frames"
                    ),
                    "reason": "semantic_gt has no evaluated frames",
                },
            ]
        )
    return {
        "detections_all_scores": len(detections),
        "detections_at_operating_point": total_predictions,
        "per_class": per_class,
        "overall_operating_point": {
            "ground_truth_instances": total_gt,
            "predictions": total_predictions,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "duplicate_fp": total_duplicate_fp,
            "precision": overall_precision,
            "recall": overall_recall,
            "evaluated_frames": evaluated_frame_count,
            "fp_per_100_evaluated_frames": _per_100_frames(
                total_fp, evaluated_frame_count
            ),
            "duplicate_fp_per_100_evaluated_frames": _per_100_frames(
                total_duplicate_fp, evaluated_frame_count
            ),
            "tp_iou_distribution": overall_tp_iou,
            "physical_instance_recall": overall_physical_recall,
        },
        "hard_negative_door_fp": hard_negative_door_fp,
        "track_association": track_association,
        "xz_error_m": xz_error,
        "unavailable_metrics": unavailable,
    }


def _noncausal_track_report(
    unavailable: list[dict[str, str]],
) -> dict[str, Any]:
    reason = (
        "posthoc NMS reuses baseline online_track_id values; rerun the "
        "online detector-to-tracker pipeline for causal track metrics"
    )
    report = {}
    for scope in (*TARGET_CLASSES, "overall"):
        report[scope] = {
            "available": False,
            "reason": reason,
        }
        unavailable.append(
            {
                "metric": f"track_association.{scope}",
                "reason": reason,
            }
        )
    return report


def _tp_iou_report(matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(float(item["iou"]) for item in matches)
    if not values:
        return {"available": False, "distribution": None}
    return {
        "available": True,
        "distribution": {
            "count": len(values),
            "values": values,
            "min": values[0],
            "mean": sum(values) / len(values),
            "median": _quantile(values, 0.50),
            "p90": _quantile(values, 0.90),
            "p95": _quantile(values, 0.95),
            "max": values[-1],
            "quantile_method": "linear_interpolation",
        },
    }


def _physical_instance_recall_report(
    matches: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    metric_path: str,
    unavailable: list[dict[str, str]],
) -> dict[str, Any]:
    gt_ids: dict[str, Any] = {}
    missing_semantic_id = 0
    for instance in ground_truth:
        token = _id_token(instance.get("semantic_id"))
        if token is None:
            missing_semantic_id += 1
        else:
            gt_ids[token] = instance["semantic_id"]

    matched_ids = {
        token
        for item in matches
        if (token := _id_token(item["ground_truth"].get("semantic_id")))
        is not None
    }
    available = bool(gt_ids) and missing_semantic_id == 0
    if not available:
        reason = (
            "one or more canonical-target GT annotations lack semantic_id"
            if missing_semantic_id
            else "semantic_gt has no canonical-target semantic_id values"
        )
        unavailable.append({"metric": metric_path, "reason": reason})
    matched_count = len(set(gt_ids).intersection(matched_ids))
    return {
        "available": available,
        "gt_physical_instances": len(gt_ids),
        "matched_physical_instances": matched_count,
        "missed_physical_instances": len(gt_ids) - matched_count,
        "target_gt_annotations_missing_semantic_id": missing_semantic_id,
        "recall": matched_count / len(gt_ids) if available else None,
    }


def _per_100_frames(count: int, evaluated_frame_count: int) -> float | None:
    if evaluated_frame_count <= 0:
        return None
    return float(count) * 100.0 / evaluated_frame_count


def _hard_negative_door_fp_report(
    door_fp_decisions: Sequence[Mapping[str, Any]],
    all_ground_truth: Sequence[Mapping[str, Any]],
    *,
    evaluated_frame_count: int,
    attribution_iou: float,
    unavailable: list[dict[str, str]],
) -> dict[str, Any]:
    hard_gt_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    hard_gt_counts = Counter({name: 0 for name in HARD_NEGATIVE_CATEGORIES})
    for instance in all_ground_truth:
        if instance.get("canonical_label") == "door":
            continue
        category = _hard_negative_category(instance.get("raw_category"))
        if category is None:
            continue
        enriched = dict(instance)
        enriched["_hard_negative_category"] = category
        hard_gt_by_frame[int(instance["frame_index"])].append(enriched)
        hard_gt_counts[category] += 1

    attributed_counts = Counter(
        {name: 0 for name in HARD_NEGATIVE_CATEGORIES}
    )
    attribution_ious: list[float] = []
    for decision in door_fp_decisions:
        detection = decision["detection"]
        candidates = hard_gt_by_frame.get(int(detection["frame_index"]), [])
        overlaps = sorted(
            (
                (box_iou(detection["box"], instance["box"]), instance)
                for instance in candidates
            ),
            key=lambda pair: (-pair[0], int(pair[1]["_source_index"])),
        )
        if not overlaps or overlaps[0][0] < attribution_iou:
            continue
        overlap, instance = overlaps[0]
        attributed_counts[instance["_hard_negative_category"]] += 1
        attribution_ious.append(overlap)

    attributed_total = sum(attributed_counts.values())
    hard_gt_total = sum(hard_gt_counts.values())
    available = evaluated_frame_count > 0 and hard_gt_total > 0
    if not available:
        unavailable.append(
            {
                "metric": (
                    "hard_negative_door_fp."
                    "hard_negative_fp_per_100_frames"
                ),
                "reason": (
                    "semantic_gt has no evaluated frames"
                    if evaluated_frame_count <= 0
                    else "semantic_gt has no recognized hard-negative instances"
                ),
            }
        )
    return {
        "available": available,
        "evaluated_frames": evaluated_frame_count,
        "door_fp": len(door_fp_decisions),
        "recognized_hard_negative_gt_instances": hard_gt_total,
        "recognized_hard_negative_gt_by_category": dict(
            sorted(hard_gt_counts.items())
        ),
        "attributed_hard_negative_fp": attributed_total,
        "unattributed_door_fp": len(door_fp_decisions) - attributed_total,
        "attributed_fp_by_category": dict(sorted(attributed_counts.items())),
        "hard_negative_fp_per_100_frames": (
            _per_100_frames(attributed_total, evaluated_frame_count)
            if available
            else None
        ),
        "hard_negative_fp_per_100_evaluated_frames": (
            _per_100_frames(attributed_total, evaluated_frame_count)
            if available
            else None
        ),
        "attribution_iou_distribution": _tp_iou_report(
            [{"iou": value} for value in attribution_ious]
        ),
    }


def _match_detections(
    detections: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered_detections = sorted(
        detections,
        key=lambda item: (
            -float(item["score"]),
            int(item["frame_index"]),
            int(item["_source_index"]),
        ),
    )
    gt_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for instance in ground_truth:
        gt_by_frame[int(instance["frame_index"])].append(instance)
    for frame_instances in gt_by_frame.values():
        frame_instances.sort(key=lambda item: int(item["_source_index"]))

    used_gt: set[int] = set()
    decisions: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for detection in ordered_detections:
        candidates = gt_by_frame.get(int(detection["frame_index"]), [])
        overlaps = sorted(
            (
                (box_iou(detection["box"], instance["box"]), instance)
                for instance in candidates
            ),
            key=lambda pair: (-pair[0], int(pair[1]["_source_index"])),
        )
        available = [
            pair
            for pair in overlaps
            if pair[0] >= iou_threshold
            and int(pair[1]["_source_index"]) not in used_gt
        ]
        if available:
            overlap, instance = available[0]
            used_gt.add(int(instance["_source_index"]))
            decisions.append(
                {
                    "tp": 1,
                    "fp": 0,
                    "duplicate_fp": 0,
                    "detection": detection,
                }
            )
            matches.append(
                {
                    "detection": detection,
                    "ground_truth": instance,
                    "iou": overlap,
                }
            )
            continue

        is_duplicate = any(overlap >= iou_threshold for overlap, _ in overlaps)
        decisions.append(
            {
                "tp": 0,
                "fp": 1,
                "duplicate_fp": int(is_duplicate),
                "detection": detection,
            }
        )
    return decisions, matches


def _track_association_report(
    matches: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    unavailable: list[dict[str, str]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for scope in (*TARGET_CLASSES, "overall"):
        scope_matches = [
            item
            for item in matches
            if scope == "overall"
            or item["ground_truth"]["canonical_label"] == scope
        ]
        scope_gt = [
            item
            for item in ground_truth
            if scope == "overall" or item["canonical_label"] == scope
        ]
        gt_ids: dict[str, Any] = {}
        for instance in scope_gt:
            token = _id_token(instance.get("semantic_id"))
            if token is not None:
                gt_ids[token] = instance["semantic_id"]

        gt_to_tracks: dict[str, set[str]] = defaultdict(set)
        track_to_gt: dict[str, set[str]] = defaultdict(set)
        track_values: dict[str, Any] = {}
        usable_pairs = 0
        missing_semantic_id = 0
        missing_track_id = 0
        for match in scope_matches:
            semantic_id = match["ground_truth"].get("semantic_id")
            track_id = match["detection"].get("online_track_id")
            semantic_token = _id_token(semantic_id)
            track_token = _id_token(track_id)
            if semantic_token is None:
                missing_semantic_id += 1
                continue
            if track_token is None:
                missing_track_id += 1
                continue
            usable_pairs += 1
            gt_ids[semantic_token] = semantic_id
            track_values[track_token] = track_id
            gt_to_tracks[semantic_token].add(track_token)
            track_to_gt[track_token].add(semantic_token)

        per_gt = []
        for semantic_token in sorted(gt_ids):
            track_tokens = sorted(gt_to_tracks.get(semantic_token, set()))
            per_gt.append(
                {
                    "semantic_id": gt_ids[semantic_token],
                    "track_count": len(track_tokens),
                    "online_track_ids": [
                        track_values[token] for token in track_tokens
                    ],
                }
            )
        wrong_merges = []
        for track_token in sorted(track_to_gt):
            semantic_tokens = sorted(track_to_gt[track_token])
            if len(semantic_tokens) > 1:
                wrong_merges.append(
                    {
                        "online_track_id": track_values[track_token],
                        "semantic_ids": [
                            gt_ids[token] for token in semantic_tokens
                        ],
                    }
                )

        associated_track_count = len(track_to_gt)
        matched_gt_count = sum(
            1 for tracks in gt_to_tracks.values() if tracks
        )
        available = (
            usable_pairs > 0
            and missing_semantic_id == 0
            and missing_track_id == 0
        )
        wrong_merge_count = len(wrong_merges)
        fragmentation_tracks_per_gt = (
            sum(len(tracks) for tracks in gt_to_tracks.values() if tracks)
            / matched_gt_count
            if available and matched_gt_count
            else None
        )
        report[scope] = {
            "available": available,
            "matched_tp_pairs": len(scope_matches),
            "usable_id_pairs": usable_pairs,
            "matched_pairs_missing_semantic_id": missing_semantic_id,
            "matched_pairs_missing_online_track_id": missing_track_id,
            "gt_objects_with_semantic_id": len(gt_ids),
            "matched_gt_objects": matched_gt_count,
            "associated_track_count": associated_track_count,
            "per_gt_track_count": per_gt if available else None,
            "fragmented_gt_count": (
                sum(1 for tracks in gt_to_tracks.values() if len(tracks) > 1)
                if available
                else None
            ),
            "fragmentation_tracks_per_gt": fragmentation_tracks_per_gt,
            "tracks_per_gt": fragmentation_tracks_per_gt,
            "wrong_merge_track_count": (
                wrong_merge_count if available else None
            ),
            "wrong_merge_rate": (
                wrong_merge_count / associated_track_count
                if available and associated_track_count
                else None
            ),
            "wrong_merge_tracks": wrong_merges if available else None,
        }
        if not available:
            unavailable.append(
                {
                    "metric": f"track_association.{scope}",
                    "reason": (
                        "track metrics require at least one TP match and "
                        "complete semantic_id/online_track_id coverage for "
                        "all fixed-operating-point TP matches"
                    ),
                }
            )
    return report


def _xz_error_report(
    matches: Sequence[Mapping[str, Any]],
    unavailable: list[dict[str, str]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for scope in (*TARGET_CLASSES, "overall"):
        scope_matches = [
            item
            for item in matches
            if scope == "overall"
            or item["ground_truth"]["canonical_label"] == scope
        ]
        pair_rows: list[dict[str, Any]] = []
        for match in scope_matches:
            position = match["detection"].get("position_3d")
            center = match["ground_truth"].get("world_visible_center_xyz")
            missing_reasons = []
            if position is None:
                missing_reasons.append("missing_detection_position")
            if center is None:
                missing_reasons.append("missing_gt_visible_center")
            error = (
                None
                if missing_reasons
                else math.hypot(
                    float(position[0]) - float(center[0]),
                    float(position[2]) - float(center[2]),
                )
            )
            pair_rows.append(
                {
                    "frame_index": int(
                        match["ground_truth"]["frame_index"]
                    ),
                    "semantic_id": match["ground_truth"].get("semantic_id"),
                    "visible_depth_median": match["ground_truth"].get(
                        "visible_depth_median"
                    ),
                    "error_m": error,
                    "missing_reasons": missing_reasons,
                }
            )
        values = sorted(
            float(item["error_m"])
            for item in pair_rows
            if item["error_m"] is not None
        )
        missing_reason_counts = Counter(
            reason
            for item in pair_rows
            for reason in item["missing_reasons"]
        )
        stratification = _xz_stratification(pair_rows)
        missing_pairs = [
            {
                "frame_index": item["frame_index"],
                "semantic_id": item["semantic_id"],
                "visible_depth_median": item["visible_depth_median"],
                "reasons": item["missing_reasons"],
            }
            for item in pair_rows
            if item["error_m"] is None
        ]
        values.sort()
        if not values:
            report[scope] = {
                "available": False,
                "matched_tp_pairs": len(scope_matches),
                "usable_position_pairs": 0,
                "missing_position_pairs": len(scope_matches),
                "usable_position_pair_rate": 0.0,
                "missing_reason_counts": dict(
                    sorted(missing_reason_counts.items())
                ),
                "missing_pairs": missing_pairs,
                "stratification": stratification,
                "distribution": None,
            }
            unavailable.append(
                {
                    "metric": f"xz_error_m.{scope}",
                    "reason": (
                        "no fixed-operating-point TP match has both valid "
                        "position_3d and a GT visible-surface world center"
                    ),
                }
            )
            continue
        report[scope] = {
            "available": True,
            "matched_tp_pairs": len(scope_matches),
            "usable_position_pairs": len(values),
            "missing_position_pairs": len(scope_matches) - len(values),
            "usable_position_pair_rate": (
                len(values) / len(scope_matches) if scope_matches else None
            ),
            "missing_reason_counts": dict(
                sorted(missing_reason_counts.items())
            ),
            "missing_pairs": missing_pairs,
            "stratification": stratification,
            "distribution": _xz_distribution(values),
        }
    return report


def _xz_stratification(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_semantic_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in pair_rows:
        token = _id_token(item.get("semantic_id")) or "missing"
        by_semantic_id[token].append(item)

    depth_rows = []
    for lower, upper in zip(
        XZ_DEPTH_STRATUM_EDGES_M[:-1],
        XZ_DEPTH_STRATUM_EDGES_M[1:],
    ):
        rows = [
            item
            for item in pair_rows
            if isinstance(item.get("visible_depth_median"), (int, float))
            and not isinstance(item.get("visible_depth_median"), bool)
            and math.isfinite(float(item["visible_depth_median"]))
            and float(item["visible_depth_median"]) >= lower
            and (
                float(item["visible_depth_median"]) < upper
                or math.isinf(upper)
            )
        ]
        depth_rows.append(
            {
                "lower_inclusive_m": lower,
                "upper_exclusive_m": None if math.isinf(upper) else upper,
                **_xz_stratum_report(rows),
            }
        )
    unknown_depth = [
        item
        for item in pair_rows
        if not isinstance(item.get("visible_depth_median"), (int, float))
        or isinstance(item.get("visible_depth_median"), bool)
        or not math.isfinite(float(item["visible_depth_median"]))
    ]
    return {
        "by_semantic_id": [
            {
                "semantic_id_token": token,
                "semantic_id": rows[0].get("semantic_id"),
                **_xz_stratum_report(rows),
            }
            for token, rows in sorted(by_semantic_id.items())
        ],
        "by_visible_depth_m": depth_rows,
        "unknown_visible_depth": _xz_stratum_report(unknown_depth),
    }


def _xz_stratum_report(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = sorted(
        float(item["error_m"])
        for item in pair_rows
        if item.get("error_m") is not None
    )
    missing_reason_counts = Counter(
        reason
        for item in pair_rows
        for reason in item.get("missing_reasons", [])
    )
    return {
        "matched_tp_pairs": len(pair_rows),
        "usable_position_pairs": len(values),
        "missing_position_pairs": len(pair_rows) - len(values),
        "missing_reason_counts": dict(sorted(missing_reason_counts.items())),
        "distribution": _xz_distribution(values) if values else None,
    }


def _xz_distribution(values: Sequence[float]) -> dict[str, Any]:
    values = sorted(float(value) for value in values)
    return {
        "count": len(values),
        "values_m": values,
        "min": values[0],
        "mean": sum(values) / len(values),
        "median": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "max": values[-1],
        "quantile_method": "linear_interpolation",
        "histogram": _histogram(values),
    }


def _histogram(values: Sequence[float]) -> list[dict[str, Any]]:
    bins = []
    for lower, upper in zip(XZ_HISTOGRAM_EDGES_M[:-1], XZ_HISTOGRAM_EDGES_M[1:]):
        count = sum(
            1
            for value in values
            if value >= lower and (value < upper or math.isinf(upper))
        )
        bins.append(
            {
                "lower_inclusive_m": lower,
                "upper_exclusive_m": None if math.isinf(upper) else upper,
                "count": count,
            }
        )
    return bins


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - fraction)
        + float(sorted_values[upper]) * fraction
    )


def _normalize_detections(
    raw_detections: Sequence[Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    normalized = []
    issues: Counter[str] = Counter()
    for source_index, raw in enumerate(raw_detections):
        if not isinstance(raw, Mapping):
            issues["record_not_object"] += 1
            continue
        label = _canonical_label(raw)
        frame_index = _frame_index(raw.get("frame_index"))
        score = _finite_number(raw.get("score"))
        box = _valid_box(raw.get("box"))
        reasons = []
        if not label:
            reasons.append("missing_label")
        if frame_index is None:
            reasons.append("invalid_frame_index")
        if score is None:
            reasons.append("invalid_score")
        if box is None:
            reasons.append("invalid_box")
        if reasons:
            for reason in reasons:
                issues[reason] += 1
            continue
        normalized.append(
            {
                "_source_index": source_index,
                "frame_index": frame_index,
                "canonical_label": label,
                "score": score,
                "box": box,
                "position_3d": _valid_xyz(raw.get("position_3d")),
                "online_track_id": raw.get("online_track_id"),
            }
        )
    return normalized, issues


def _normalize_ground_truth(
    raw_frames: Sequence[Any],
) -> tuple[list[dict[str, Any]], Counter[str], list[Any], set[int]]:
    normalized = []
    issues: Counter[str] = Counter()
    raw_instances: list[Any] = []
    evaluated_frames: set[int] = set()
    source_index = 0
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, Mapping):
            issues["frame_not_object"] += 1
            continue
        frame_index = _frame_index(raw_frame.get("frame_index"))
        instances = raw_frame.get("instances")
        if not isinstance(instances, list):
            issues["instances_not_list"] += 1
            continue
        if frame_index is not None:
            evaluated_frames.add(frame_index)
        for raw in instances:
            current_index = source_index
            source_index += 1
            raw_instances.append(raw)
            if not isinstance(raw, Mapping):
                issues["record_not_object"] += 1
                continue
            label = _normalized_text(raw.get("canonical_label"))
            box = _valid_box(raw.get("box"))
            reasons = []
            if frame_index is None:
                reasons.append("invalid_frame_index")
            if box is None:
                reasons.append("invalid_box")
            if reasons:
                for reason in reasons:
                    issues[reason] += 1
                continue
            normalized.append(
                {
                    "_source_index": current_index,
                    "frame_index": frame_index,
                    "canonical_label": label,
                    "box": box,
                    "semantic_id": raw.get("semantic_id"),
                    "object_id": raw.get("object_id"),
                    "raw_category": raw.get("raw_category"),
                    "world_visible_center_xyz": _valid_xyz(
                        raw.get("world_visible_center_xyz")
                    ),
                    "world_center_xyz": _valid_xyz(
                        raw.get("world_center_xyz")
                    ),
                    "visible_depth_median": _finite_number(
                        raw.get("visible_depth_median")
                    ),
                    "visible_depth_valid_ratio": _finite_number(
                        raw.get("visible_depth_valid_ratio")
                    ),
                    "visible_projected_points": _frame_index(
                        raw.get("visible_projected_points")
                    ),
                    "area_px": _finite_number(raw.get("area_px")),
                }
            )
    return normalized, issues, raw_instances, evaluated_frames


def _coverage_report(
    *,
    raw_detections: Sequence[Any],
    raw_frames: Sequence[Any],
    raw_gt_instances: Sequence[Any],
    detections: Sequence[Mapping[str, Any]],
    evaluated_detections: Sequence[Mapping[str, Any]],
    target_detections: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    target_ground_truth: Sequence[Mapping[str, Any]],
    evaluated_frames: set[int],
    detection_issues: Counter[str],
    gt_issues: Counter[str],
) -> dict[str, Any]:
    detection_frames = sorted(
        {
            item["frame_index"]
            for item in detections
            if item["canonical_label"] in TARGET_CLASSES
        }
    )
    gt_frames = sorted(
        {
            item["frame_index"]
            for item in ground_truth
            if item["canonical_label"] in TARGET_CLASSES
        }
    )
    detection_fields = _field_coverage(
        [item for item in raw_detections if isinstance(item, Mapping)],
        (
            "frame_index",
            "label",
            "canonical_label",
            "score",
            "box",
            "position_3d",
            "online_track_id",
        ),
    )
    gt_fields = _field_coverage(
        [item for item in raw_gt_instances if isinstance(item, Mapping)],
        (
            "semantic_id",
            "object_id",
            "raw_category",
            "canonical_label",
            "box",
            "world_visible_center_xyz",
            "world_center_xyz",
            "visible_depth_median",
            "visible_depth_valid_ratio",
            "visible_projected_points",
            "area_px",
        ),
    )
    return {
        "detections": {
            "raw_records": len(raw_detections),
            "valid_records_all_frames": len(detections),
            "evaluable_records_all_classes": len(evaluated_detections),
            "evaluable_target_records": len(target_detections),
            "excluded_unannotated_frame_records": (
                len(detections) - len(evaluated_detections)
            ),
            "target_records_by_class": dict(
                sorted(
                    Counter(
                        item["canonical_label"] for item in target_detections
                    ).items()
                )
            ),
            "invalid_record_reasons": dict(sorted(detection_issues.items())),
            "field_non_null_counts": detection_fields,
        },
        "semantic_gt": {
            "raw_frames": len(raw_frames),
            "raw_instances": len(raw_gt_instances),
            "evaluable_instances_all_classes": len(ground_truth),
            "evaluable_target_instances": len(target_ground_truth),
            "target_instances_by_class": dict(
                sorted(
                    Counter(
                        item["canonical_label"]
                        for item in target_ground_truth
                    ).items()
                )
            ),
            "invalid_record_reasons": dict(sorted(gt_issues.items())),
            "field_non_null_counts": gt_fields,
        },
        "target_frame_coverage": {
            "evaluated_frame_count": len(evaluated_frames),
            "evaluated_frame_indices": sorted(evaluated_frames),
            "detection_frame_count": len(detection_frames),
            "semantic_gt_frame_count": len(gt_frames),
            "overlap_frame_count": len(
                set(detection_frames).intersection(gt_frames)
            ),
            "detection_only_frames": sorted(
                set(detection_frames).difference(gt_frames)
            ),
            "semantic_gt_only_frames": sorted(
                set(gt_frames).difference(detection_frames)
            ),
        },
    }


def _field_coverage(
    records: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> dict[str, int]:
    records = list(records)
    return {
        field: sum(
            1 for record in records if field in record and record[field] is not None
        )
        for field in fields
    }


def _canonical_label(record: Mapping[str, Any]) -> str:
    value = record.get("canonical_label")
    if not isinstance(value, str) or not value.strip():
        value = record.get("label")
    return _normalized_text(value)


def _normalized_text(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _hard_negative_category(raw_category: Any) -> str | None:
    normalized = _normalized_text(raw_category)
    normalized = " ".join(
        normalized.replace("_", " ").replace("-", " ").split()
    )
    words = set(normalized.split())
    if "window" in words:
        return "window"
    if "refrigerator" in words or "fridge" in words:
        return "refrigerator door"
    if "cabinet" in words or "cupboard" in words:
        return "cabinet door"
    if "mirror" in words:
        return "mirror"
    if {"wall", "panel"} <= words:
        return "wall panel"
    return None


def _frame_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    values = [_finite_number(item) for item in value]
    if any(item is None for item in values):
        return None
    box = [float(item) for item in values if item is not None]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _valid_xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    values = [_finite_number(item) for item in value]
    if any(item is None for item in values):
        return None
    return [float(item) for item in values if item is not None]


def _id_token(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if not isinstance(value, (str, int, float)):
        return None
    return f"{type(value).__name__}:{value}"


def _validate_threshold(
    name: str, value: float, *, lower: float, upper: float
) -> None:
    if not math.isfinite(value) or value < lower or value > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _required_sha256(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{context}.{key} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{context}.{key} must be a SHA-256 hex digest"
        ) from error
    return value.lower()


def _validate_semantic_gt_integrity(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    frames = payload.get("frames")
    source = payload.get("source")
    selection = payload.get("selection")
    if not isinstance(frames, list):
        raise ValueError("semantic GT frames must be a list")
    if not isinstance(source, Mapping):
        raise ValueError("semantic GT source must be an object")
    if not isinstance(selection, Mapping):
        raise ValueError("semantic GT selection must be an object")
    frame_count = len(frames)
    if selection.get("selected_num_frames") != frame_count:
        raise ValueError(
            "semantic GT selected_num_frames does not match frames length"
        )
    frame_indices = []
    input_frame_hashes = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("semantic GT frame must be an object")
        frame_index = _frame_index(frame.get("frame_index"))
        if frame_index is None:
            raise ValueError("semantic GT frame_index is invalid")
        frame_indices.append(frame_index)
        input_frame_hashes.append(
            {
                "frame_index": frame_index,
                "rgb_sha256": _required_sha256(
                    frame,
                    "source_rgb_sha256",
                    f"semantic GT frame {frame_index}",
                ),
                "depth_sha256": _required_sha256(
                    frame,
                    "source_depth_sha256",
                    f"semantic GT frame {frame_index}",
                ),
            }
        )
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError("semantic GT frame_index values must be unique")
    selected_indices_sha256 = _canonical_payload_sha256(
        {"frame_indices": frame_indices}
    )
    if (
        _required_sha256(
            selection,
            "selected_frame_indices_sha256",
            "semantic GT selection",
        )
        != selected_indices_sha256
    ):
        raise ValueError("semantic GT selected frame index hash mismatch")
    input_frame_hashes_sha256 = _canonical_payload_sha256(
        {"frames": input_frame_hashes}
    )
    if (
        _required_sha256(
            source,
            "input_frame_hashes_sha256",
            "semantic GT source",
        )
        != input_frame_hashes_sha256
    ):
        raise ValueError("semantic GT input frame hash manifest mismatch")

    integrity_contract: dict[str, Any] = {
        "contract": "full_frame_rgb_depth_replay_v1",
        "frame_count": frame_count,
    }
    for name in ("rgb", "depth"):
        checks = payload.get(f"{name}_replay_checks")
        if not isinstance(checks, list):
            raise ValueError(
                f"semantic GT {name}_replay_checks must be a list"
            )
        if len(checks) != frame_count:
            raise ValueError(
                f"semantic GT {name} replay checks are not full-frame"
            )
        checked_frame_indices = []
        for check in checks:
            if not isinstance(check, Mapping):
                raise ValueError(
                    f"semantic GT {name} replay check must be an object"
                )
            check_frame_index = _frame_index(check.get("frame_index"))
            if check_frame_index is None:
                raise ValueError(
                    f"semantic GT {name} replay check frame_index is invalid"
                )
            if check.get("available") is not True:
                raise ValueError(
                    f"semantic GT {name} replay check is unavailable for "
                    f"frame {check_frame_index}"
                )
            checked_frame_indices.append(check_frame_index)
        if checked_frame_indices != frame_indices:
            raise ValueError(
                f"semantic GT {name} replay checks do not match selected "
                "frame order"
            )
        report = payload.get(f"{name}_replay_integrity")
        if not isinstance(report, Mapping):
            raise ValueError(
                f"semantic GT {name}_replay_integrity must be an object"
            )
        if report.get("enabled") is not True:
            raise ValueError(
                f"semantic GT {name} replay integrity was not enabled"
            )
        if report.get("passed") is not True:
            raise ValueError(
                f"semantic GT {name} replay integrity did not pass"
            )
        if report.get("required_checks") != frame_count:
            raise ValueError(
                f"semantic GT {name} integrity is not full-frame"
            )
        if report.get("available_checks") != frame_count:
            raise ValueError(
                f"semantic GT {name} integrity has unavailable checks"
            )
        integrity_contract[f"{name}_integrity"] = dict(report)
    integrity_contract.update(
        {
            "generator_sha256": _required_sha256(
                source,
                "generator_sha256",
                "semantic GT source",
            ),
            "scene_id": str(source.get("scene_id", "")).strip(),
            "scene_sha256": _required_sha256(
                source,
                "scene_sha256",
                "semantic GT source",
            ),
            "frames_metadata_sha256": _required_sha256(
                source,
                "frames_metadata_sha256",
                "semantic GT source",
            ),
            "selected_frame_indices_sha256": selected_indices_sha256,
            "input_frame_hashes_sha256": input_frame_hashes_sha256,
        }
    )
    if not integrity_contract["scene_id"]:
        raise ValueError("semantic GT source.scene_id is required")
    return integrity_contract


def _detection_contract(
    detections_payload: Mapping[str, Any],
    *,
    external_contract_path: Path | None,
    semantic_integrity: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    embedded_contract = detections_payload.get("algorithm_contract")
    embedded_sha256 = detections_payload.get("algorithm_sha256")
    embedded_run_contract = detections_payload.get("run_contract")
    embedded_run_sha256 = detections_payload.get("run_contract_sha256")
    external_payload = None
    if external_contract_path is not None:
        external_payload = json.loads(
            external_contract_path.read_text(encoding="utf-8")
        )
        if not isinstance(external_payload, Mapping):
            raise ValueError(
                "detection algorithm contract JSON root must be an object"
            )

    if isinstance(embedded_contract, Mapping):
        computed_sha256 = _canonical_payload_sha256(embedded_contract)
        if not isinstance(embedded_sha256, str) or (
            embedded_sha256.lower() != computed_sha256
        ):
            raise ValueError(
                "detections embedded algorithm_sha256 does not match "
                "algorithm_contract"
            )
        contract = embedded_contract
        run_contract = embedded_run_contract
        if not isinstance(run_contract, Mapping):
            raise ValueError("detections must embed a run_contract object")
        computed_run_sha256 = _canonical_payload_sha256(run_contract)
        if not isinstance(embedded_run_sha256, str) or (
            embedded_run_sha256.lower() != computed_run_sha256
        ):
            raise ValueError(
                "detections embedded run_contract_sha256 does not match "
                "run_contract"
            )
        _validate_embedded_detection_output(
            detections_payload,
            algorithm_contract=contract,
            run_contract=run_contract,
        )
        if external_payload is not None:
            external_algorithm = external_payload.get("algorithm_contract")
            external_run = external_payload.get("run_contract")
            if not isinstance(external_algorithm, Mapping) or (
                _canonical_payload_sha256(external_algorithm)
                != computed_sha256
            ):
                raise ValueError(
                    "external detection algorithm contract does not match "
                    "embedded contract"
                )
            if not isinstance(external_run, Mapping) or (
                _canonical_payload_sha256(external_run)
                != computed_run_sha256
            ):
                raise ValueError(
                    "external detection run contract does not match embedded "
                    "contract"
                )
        source_type = "embedded"
    elif external_payload is not None:
        contract = external_payload.get("algorithm_contract")
        run_contract = external_payload.get("run_contract")
        if not isinstance(contract, Mapping):
            raise ValueError(
                "external detection contract must contain algorithm_contract"
            )
        if not isinstance(run_contract, Mapping):
            raise ValueError(
                "external detection contract must contain run_contract"
            )
        computed_sha256 = _canonical_payload_sha256(contract)
        computed_run_sha256 = _canonical_payload_sha256(run_contract)
        source_type = "external_legacy"
    else:
        raise ValueError(
            "detections must embed an algorithm contract or provide "
            "--detection-algorithm-contract-json"
        )

    paired_fields = (
        "frames_metadata_sha256",
        "selected_frame_indices_sha256",
        "input_frame_hashes_sha256",
    )
    for field in paired_fields:
        if (
            _required_sha256(
                run_contract,
                field,
                "detection run_contract",
            )
            != semantic_integrity[field]
        ):
            raise ValueError(
                f"detection run_contract.{field} does not match "
                "semantic GT"
            )
    if run_contract.get("selected_frames") != semantic_integrity[
        "frame_count"
    ]:
        raise ValueError(
            "detection replay selected_frames does not match semantic GT"
        )
    return contract, {
        "source": source_type,
        "canonical_sha256": computed_sha256,
        "run_contract": dict(run_contract),
        "run_contract_sha256": computed_run_sha256,
        "external_json": (
            str(external_contract_path)
            if external_contract_path is not None
            else None
        ),
        "external_file_sha256": (
            _sha256(external_contract_path)
            if external_contract_path is not None
            else None
        ),
        "formal_eligible": source_type == "embedded",
    }


def _validate_embedded_detection_output(
    payload: Mapping[str, Any],
    *,
    algorithm_contract: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> None:
    generator_sha256 = _required_sha256(
        algorithm_contract,
        "generator_sha256",
        "detection algorithm_contract",
    )
    if algorithm_contract.get("semantic_oracle_access") is not False:
        raise ValueError(
            "embedded detection algorithm must declare "
            "semantic_oracle_access=false"
        )
    labels = algorithm_contract.get("labels")
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(item, str) and item.strip() for item in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(
            "embedded detection algorithm labels must be unique non-empty "
            "strings"
        )
    _validate_model_artifact_manifest(
        algorithm_contract.get("model_artifacts")
    )
    implementation_hashes = _validate_implementation_artifact_manifest(
        algorithm_contract.get("implementation_artifacts")
    )
    if implementation_hashes["scripts/grounding_detector_replay.py"] != (
        generator_sha256
    ):
        raise ValueError(
            "embedded detection generator hash does not match its "
            "implementation artifact"
        )
    runtime = algorithm_contract.get("runtime")
    if not isinstance(runtime, Mapping) or not str(
        runtime.get("python", "")
    ).strip():
        raise ValueError(
            "embedded detection algorithm runtime.python is required"
        )
    packages = runtime.get("packages")
    required_packages = ("numpy", "Pillow", "torch", "transformers")
    if not isinstance(packages, Mapping) or not all(
        isinstance(packages.get(name), str) and packages[name].strip()
        for name in required_packages
    ):
        raise ValueError(
            "embedded detection algorithm runtime package versions are "
            "incomplete"
        )

    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("embedded detections must contain a source object")
    paired_sha_fields = (
        "frames_metadata_sha256",
        "selected_frame_indices_sha256",
        "input_frame_hashes_sha256",
    )
    for field in paired_sha_fields:
        source_value = _required_sha256(
            source,
            field,
            "detection source",
        )
        run_value = _required_sha256(
            run_contract,
            field,
            "detection run_contract",
        )
        if source_value != run_value:
            raise ValueError(
                f"detection source.{field} does not match run_contract"
            )
    for field in (
        "selected_frames",
        "frame_start",
        "frame_end",
        "frame_stride",
        "max_frames",
    ):
        if source.get(field) != run_contract.get(field):
            raise ValueError(
                f"detection source.{field} does not match run_contract"
            )

    selected_frames = run_contract.get("selected_frames")
    if (
        not isinstance(selected_frames, int)
        or isinstance(selected_frames, bool)
        or selected_frames <= 0
    ):
        raise ValueError(
            "detection run_contract.selected_frames must be a positive integer"
        )
    input_hashes = source.get("input_frame_hashes")
    if not isinstance(input_hashes, list) or (
        len(input_hashes) != selected_frames
    ):
        raise ValueError(
            "embedded detection input frame hashes are not full-frame"
        )
    canonical_records = []
    frame_indices = []
    for record in input_hashes:
        if not isinstance(record, Mapping):
            raise ValueError(
                "embedded detection input frame hash record must be an object"
            )
        frame_index = _frame_index(record.get("frame_index"))
        if frame_index is None:
            raise ValueError(
                "embedded detection input frame hash frame_index is invalid"
            )
        frame_indices.append(frame_index)
        canonical_records.append(
            {
                "frame_index": frame_index,
                "rgb_sha256": _required_sha256(
                    record,
                    "rgb_sha256",
                    f"detection input frame {frame_index}",
                ),
                "depth_sha256": _required_sha256(
                    record,
                    "depth_sha256",
                    f"detection input frame {frame_index}",
                ),
            }
        )
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError(
            "embedded detection input frame indices must be unique"
        )
    if _canonical_payload_sha256(
        {"frame_indices": frame_indices}
    ) != run_contract["selected_frame_indices_sha256"]:
        raise ValueError(
            "embedded detection selected frame index hash does not match "
            "input records"
        )
    if _canonical_payload_sha256(
        {"frames": canonical_records}
    ) != run_contract["input_frame_hashes_sha256"]:
        raise ValueError(
            "embedded detection input frame hash manifest does not match "
            "input records"
        )

    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise ValueError("embedded detections.detections must be a list")
    selected_index_set = set(frame_indices)
    for detection in detections:
        if not isinstance(detection, Mapping):
            raise ValueError("embedded detection record must be an object")
        frame_index = _frame_index(detection.get("frame_index"))
        if frame_index not in selected_index_set:
            raise ValueError(
                "embedded detection references a frame outside the frozen "
                "selection"
            )


def _validate_model_artifact_manifest(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(
            "embedded detection algorithm model_artifacts must be an object"
        )
    local_directory = value.get("local_directory")
    if not isinstance(local_directory, bool):
        raise ValueError(
            "embedded detection model_artifacts.local_directory must be "
            "boolean"
        )
    if not local_directory:
        if not str(value.get("identifier", "")).strip():
            raise ValueError(
                "embedded detection remote model identifier is required"
            )
        return
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(
            "embedded detection local model artifact list must be non-empty"
        )
    canonical_files = []
    for index, record in enumerate(files):
        if not isinstance(record, Mapping):
            raise ValueError(
                "embedded detection model artifact record must be an object"
            )
        path = str(record.get("path", "")).strip()
        size_bytes = record.get("size_bytes")
        if not path:
            raise ValueError(
                "embedded detection model artifact path is required"
            )
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ValueError(
                "embedded detection model artifact size_bytes is invalid"
            )
        canonical_files.append(
            {
                "path": path,
                "size_bytes": size_bytes,
                "sha256": _required_sha256(
                    record,
                    "sha256",
                    f"detection model artifact {index}",
                ),
            }
        )
    if _required_sha256(
        value,
        "manifest_sha256",
        "detection model_artifacts",
    ) != _canonical_payload_sha256({"files": canonical_files}):
        raise ValueError(
            "embedded detection model artifact manifest hash mismatch"
        )


def _validate_implementation_artifact_manifest(
    value: Any,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "embedded detection implementation_artifacts must be an object"
        )
    files = value.get("files")
    if not isinstance(files, list):
        raise ValueError(
            "embedded detection implementation artifact files must be a list"
        )
    required_paths = {
        "scripts/grounding_detector_replay.py",
        "scripts/m25_groundingdino_export.py",
        "src/semantic_task_profile.py",
    }
    canonical_files = []
    hashes: dict[str, str] = {}
    for index, record in enumerate(files):
        if not isinstance(record, Mapping):
            raise ValueError(
                "embedded detection implementation artifact must be an object"
            )
        path = str(record.get("path", "")).strip()
        size_bytes = record.get("size_bytes")
        if not path or path in hashes:
            raise ValueError(
                "embedded detection implementation artifact paths must be "
                "unique and non-empty"
            )
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ValueError(
                "embedded detection implementation artifact size is invalid"
            )
        sha256 = _required_sha256(
            record,
            "sha256",
            f"detection implementation artifact {index}",
        )
        hashes[path] = sha256
        canonical_files.append(
            {
                "path": path,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )
    if set(hashes) != required_paths:
        raise ValueError(
            "embedded detection implementation artifact set is incomplete"
        )
    if _required_sha256(
        value,
        "manifest_sha256",
        "detection implementation_artifacts",
    ) != _canonical_payload_sha256({"files": canonical_files}):
        raise ValueError(
            "embedded detection implementation artifact manifest hash "
            "mismatch"
        )
    return hashes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections-json", required=True)
    parser.add_argument("--semantic-gt-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--detection-algorithm-contract-json",
        help=(
            "Optional frozen detector/inference contract composed into every "
            "variant algorithm hash."
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Fixed operating point; detections with score >= threshold.",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="IoU used for fixed-operating-point TP/FP/FN and track metrics.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=1.0,
        help="Per-frame canonical-class NMS IoU; values >= 1 disable NMS.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    detections_path = Path(args.detections_json).expanduser().resolve()
    semantic_gt_path = Path(args.semantic_gt_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    detections_payload = json.loads(detections_path.read_text(encoding="utf-8"))
    semantic_gt_payload = json.loads(semantic_gt_path.read_text(encoding="utf-8"))
    semantic_integrity = _validate_semantic_gt_integrity(
        semantic_gt_payload
    )
    result = audit_payloads(
        detections_payload,
        semantic_gt_payload,
        score_threshold=args.score_threshold,
        match_iou=args.match_iou,
        nms_iou=args.nms_iou,
    )
    detector_contract_path = (
        Path(args.detection_algorithm_contract_json).expanduser().resolve()
        if args.detection_algorithm_contract_json
        else None
    )
    detector_contract, detector_contract_provenance = _detection_contract(
        detections_payload,
        external_contract_path=detector_contract_path,
        semantic_integrity=semantic_integrity,
    )
    for algorithm_record in result["variant_algorithms"].values():
        variant_contract = algorithm_record["contract"]
        composite_contract = {
            "detection_algorithm": detector_contract,
            "evaluation_variant": variant_contract,
        }
        algorithm_record["contract"] = composite_contract
        algorithm_record["sha256"] = _canonical_payload_sha256(
            composite_contract
        )
    result["inputs"] = {
        "detections_json": str(detections_path),
        "detections_sha256": _sha256(detections_path),
        "semantic_gt_json": str(semantic_gt_path),
        "semantic_gt_sha256": _sha256(semantic_gt_path),
        "semantic_gt_generator_sha256": (
            semantic_integrity["generator_sha256"]
        ),
        "scene_id": semantic_integrity["scene_id"],
        "scene_sha256": semantic_integrity["scene_sha256"],
        "frames_metadata_sha256": semantic_integrity[
            "frames_metadata_sha256"
        ],
        "selected_frame_indices_sha256": semantic_integrity[
            "selected_frame_indices_sha256"
        ],
        "input_frame_hashes_sha256": semantic_integrity[
            "input_frame_hashes_sha256"
        ],
        "semantic_gt_integrity_contract": semantic_integrity,
        "semantic_gt_integrity_sha256": _canonical_payload_sha256(
            semantic_integrity
        ),
        "detection_algorithm_contract_json": (
            str(detector_contract_path)
            if detector_contract_path is not None
            else None
        ),
        "detection_algorithm_contract": detector_contract_provenance,
        "detection_run_contract_sha256": (
            detector_contract_provenance["run_contract_sha256"]
        ),
        "detection_contract_source": (
            detector_contract_provenance["source"]
        ),
        "formal_detection_contract_eligible": (
            detector_contract_provenance["formal_eligible"]
        ),
        "evaluator_script": str(Path(__file__).resolve()),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
