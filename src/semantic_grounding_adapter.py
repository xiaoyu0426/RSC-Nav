from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import json
import math


@dataclass
class GroundingCandidate:
    id: str
    label: str
    position_3d: tuple[float, float, float]
    confidence: float
    context_id: str
    source: str
    semantic_id: int | None = None
    source_view_ids: list[str] = field(default_factory=list)
    bbox: dict[str, float] | None = None
    mask_ref: str | None = None
    status: str = "active"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["position_3d"] = [float(value) for value in self.position_3d]
        return data


def load_grounding_candidates(path: str | Path, context_id: str | None = None) -> list[GroundingCandidate]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_items = data.get("items", data.get("objects", data)) if isinstance(data, dict) else data
    candidates = []
    for index, raw in enumerate(raw_items):
        candidate = _candidate_from_raw(raw, index=index, context_id=context_id)
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.id)


def candidates_from_habitat_memory(
    path: str | Path,
    context_id: str,
    min_confidence: float = 0.05,
    include_statuses: Iterable[str] = ("active", "stale"),
) -> list[GroundingCandidate]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed_statuses = {str(value).lower() for value in include_statuses}
    candidates = []
    for index, item in enumerate(data.get("items", [])):
        status = str(item.get("status", "active")).lower()
        confidence = float(item.get("confidence", 0.0))
        if status not in allowed_statuses or confidence < float(min_confidence):
            continue
        centroid = item.get("centroid_xz") or item.get("bev_position")
        if centroid is None or len(centroid) < 2:
            continue
        label = str(item.get("category") or item.get("label") or "").strip().lower()
        if not label:
            continue
        visible_steps = [int(value) for value in item.get("visible_steps", [])]
        source_views = [f"step_{step:04d}" for step in visible_steps]
        raw_id = item.get("id") or item.get("object_id") or f"{label}_{index:03d}"
        candidates.append(
            GroundingCandidate(
                id=str(raw_id),
                label=label,
                position_3d=(float(centroid[0]), 0.0, float(centroid[1])),
                confidence=max(0.0, min(1.0, confidence)),
                context_id=context_id,
                source="habitat_semantic_oracle",
                semantic_id=_optional_int(item.get("semantic_id")),
                source_view_ids=source_views,
                status=status,
                raw=dict(item),
            )
        )
    return sorted(candidates, key=lambda item: item.id)


def write_grounding_candidates(path: str | Path, candidates: Iterable[GroundingCandidate], metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "source": "m25_open_vocab_grounding_adapter",
        "metadata": metadata or {},
        "items": [item.to_dict() for item in candidates],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compare_candidates(
    predicted: Iterable[GroundingCandidate],
    gold: Iterable[GroundingCandidate],
    distance_threshold_m: float = 0.75,
) -> dict[str, Any]:
    predicted_items = list(predicted)
    gold_items = list(gold)
    matched_predicted: set[int] = set()
    matches = []
    for gold_index, gold_item in enumerate(gold_items):
        best_index = None
        best_distance = math.inf
        for pred_index, pred_item in enumerate(predicted_items):
            if pred_index in matched_predicted or pred_item.label != gold_item.label:
                continue
            distance = _distance_xz(pred_item.position_3d, gold_item.position_3d)
            if distance < best_distance:
                best_distance = distance
                best_index = pred_index
        if best_index is None:
            continue
        if best_distance <= float(distance_threshold_m):
            matched_predicted.add(best_index)
            matches.append(
                {
                    "gold_id": gold_item.id,
                    "predicted_id": predicted_items[best_index].id,
                    "label": gold_item.label,
                    "centroid_error_m": best_distance,
                }
            )
    true_positive = len(matches)
    predicted_count = len(predicted_items)
    gold_count = len(gold_items)
    precision = _safe_div(true_positive, predicted_count)
    recall = _safe_div(true_positive, gold_count)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    by_label = {}
    for label in sorted({item.label for item in [*predicted_items, *gold_items]}):
        label_pred = [item for item in predicted_items if item.label == label]
        label_gold = [item for item in gold_items if item.label == label]
        label_matches = [item for item in matches if item["label"] == label]
        label_precision = _safe_div(len(label_matches), len(label_pred))
        label_recall = _safe_div(len(label_matches), len(label_gold))
        by_label[label] = {
            "predicted": len(label_pred),
            "gold": len(label_gold),
            "matched": len(label_matches),
            "precision": label_precision,
            "recall": label_recall,
        }
    return {
        "distance_threshold_m": float(distance_threshold_m),
        "predicted_count": predicted_count,
        "gold_count": gold_count,
        "true_positive": true_positive,
        "false_positive": predicted_count - true_positive,
        "false_negative": gold_count - true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_centroid_error_m": (
            sum(item["centroid_error_m"] for item in matches) / len(matches)
            if matches
            else None
        ),
        "by_label": by_label,
        "matches": matches,
    }


def _candidate_from_raw(raw: dict[str, Any], index: int, context_id: str | None) -> GroundingCandidate | None:
    label = str(raw.get("label") or raw.get("name") or raw.get("category") or "").strip().lower()
    if not label:
        return None
    position = raw.get("position_3d") or raw.get("position") or raw.get("centroid_3d")
    if position is None:
        centroid = raw.get("centroid_xz") or raw.get("bev_position")
        if centroid is None or len(centroid) < 2:
            return None
        position = [float(centroid[0]), 0.0, float(centroid[1])]
    if len(position) < 3:
        return None
    object_id = str(raw.get("id") or raw.get("object_id") or f"grounding_{label}_{index:03d}")
    return GroundingCandidate(
        id=object_id.replace(" ", "_"),
        label=label,
        position_3d=(float(position[0]), float(position[1]), float(position[2])),
        confidence=max(0.0, min(1.0, float(raw.get("confidence", raw.get("score", 0.75))))),
        context_id=str(raw.get("context_id") or context_id or "unknown_context"),
        source=str(raw.get("source") or "external_grounding"),
        semantic_id=_optional_int(raw.get("semantic_id")),
        source_view_ids=[str(value) for value in raw.get("source_view_ids", raw.get("view_ids", []))],
        bbox=raw.get("bbox", raw.get("bounding_box")),
        mask_ref=raw.get("mask_ref"),
        status=str(raw.get("status") or "active"),
        raw=dict(raw),
    )


def _distance_xz(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return float(math.hypot(float(left[0]) - float(right[0]), float(left[2]) - float(right[2])))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
