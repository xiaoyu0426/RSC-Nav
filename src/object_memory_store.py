from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, sqrt
from pathlib import Path
from typing import Iterable, Optional

import json


WorldXZ = tuple[float, float]


@dataclass
class ObjectMemoryItem:
    id: str
    semantic_id: int
    object_id: Optional[str]
    category: str
    centroid_xz: WorldXZ
    confidence: float
    freshness: float
    first_seen_step: int
    last_seen_step: int
    visible_steps: list[int] = field(default_factory=list)
    footprint_cells: int = 0
    fragmentation_count: int = 0
    missed_observation_count: int = 0
    negative_evidence_count: int = 0
    not_observable_count: int = 0
    missed_observation_weight: float = 0.0
    negative_evidence_weight: float = 0.0
    not_observable_weight: float = 0.0
    positive_evidence_weight: float = 0.0
    status: str = "active"
    source: str = "semantic_gt"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["centroid_xz"] = list(self.centroid_xz)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectMemoryItem":
        item = dict(data)
        item["centroid_xz"] = tuple(float(value) for value in item["centroid_xz"])
        return cls(**item)


class ObjectMemoryStore:
    """Persistent object memory built from Habitat semantic GT tracks."""

    def __init__(
        self,
        scene_id: str,
        freshness_tau_steps: float = 20.0,
        missing_confidence_threshold: float = 0.35,
        missing_missed_weight_threshold: float = 6.0,
        stale_missed_weight_threshold: float = 2.5,
    ) -> None:
        self.scene_id = scene_id
        self.freshness_tau_steps = max(1.0, float(freshness_tau_steps))
        self.missing_confidence_threshold = float(missing_confidence_threshold)
        self.missing_missed_weight_threshold = float(missing_missed_weight_threshold)
        self.stale_missed_weight_threshold = float(stale_missed_weight_threshold)
        self.items: dict[str, ObjectMemoryItem] = {}

    def update_from_tracks(
        self,
        tracks: Iterable[dict],
        current_step: Optional[int] = None,
        observability: Optional[dict] = None,
        evidence_weight: float = 1.0,
        negative_evidence_weight: Optional[float] = None,
    ) -> dict:
        tracks = list(tracks)
        if current_step is None:
            current_step = max((int(track.get("last_seen_step", 0)) for track in tracks), default=0)
        evidence_weight = _clip(float(evidence_weight))

        created = 0
        updated = 0
        positive_ids = set()
        for track in tracks:
            item_id = self._item_id(track)
            positive_ids.add(item_id)
            incoming = self._item_from_track(track, item_id=item_id)
            incoming.freshness = 1.0
            incoming.missed_observation_count = 0
            incoming.negative_evidence_count = 0
            incoming.status = "active"
            if item_id in self.items:
                self.items[item_id] = _positive_update(self.items[item_id], incoming, evidence_weight=evidence_weight)
                updated += 1
            else:
                incoming.positive_evidence_weight = evidence_weight
                self.items[item_id] = incoming
                created += 1

        evidence_updates = self.update_visibility_evidence(
            current_step=current_step,
            positive_ids=positive_ids,
            observability=observability or {},
            evidence_weight=evidence_weight,
            negative_evidence_weight=negative_evidence_weight,
        )
        return {
            "created": created,
            "updated": updated,
            **evidence_updates,
            "current_step": int(current_step),
            "num_items": len(self.items),
        }

    def decay(self, current_step: int) -> int:
        return self.update_visibility_evidence(
            current_step=current_step,
            positive_ids=set(),
            observability={},
        )["not_observable"]

    def update_visibility_evidence(
        self,
        current_step: int,
        positive_ids: set[str],
        observability: dict,
        evidence_weight: float = 1.0,
        negative_evidence_weight: Optional[float] = None,
    ) -> dict:
        evidence_weight = _clip(float(evidence_weight))
        negative_evidence_weight = evidence_weight if negative_evidence_weight is None else _clip(float(negative_evidence_weight))
        not_observable = 0
        expected_visible_miss = 0
        for item in self.items.values():
            if item.id in positive_ids:
                continue
            state = _observability_for_item(observability, item)
            if state == "expected_visible_miss" and negative_evidence_weight >= 0.15:
                _expected_visible_miss_update(item, evidence_weight=negative_evidence_weight)
                expected_visible_miss += 1
            else:
                _not_observable_update(item, evidence_weight=evidence_weight)
                not_observable += 1
            item.status = self._status_from_evidence(item)
        return {
            "not_observable": not_observable,
            "expected_visible_miss": expected_visible_miss,
        }

    def retrieve(
        self,
        category: str,
        current_xz: WorldXZ = (0.0, 0.0),
        top_k: int = 5,
    ) -> list[dict]:
        results = []
        for item in self.items.values():
            if item.category != category:
                continue
            distance = _distance(current_xz, item.centroid_xz)
            spatial = 1.0 / (1.0 + distance)
            score = 1.2 * item.confidence + 1.0 * item.freshness + 0.4 * spatial - _status_penalty(item.status)
            results.append(
                {
                    "id": item.id,
                    "category": item.category,
                    "semantic_id": item.semantic_id,
                    "score": round(score, 4),
                    "confidence": round(item.confidence, 4),
                    "freshness": round(item.freshness, 4),
                    "status": item.status,
                    "missed_observation_count": item.missed_observation_count,
                    "negative_evidence_count": item.negative_evidence_count,
                    "centroid_xz": list(item.centroid_xz),
                    "distance": round(distance, 4),
                }
            )
        results.sort(key=lambda row: row["score"], reverse=True)
        return results[:top_k]

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "freshness_tau_steps": self.freshness_tau_steps,
            "status_thresholds": {
                "missing_confidence_threshold": self.missing_confidence_threshold,
                "missing_missed_weight_threshold": self.missing_missed_weight_threshold,
                "stale_missed_weight_threshold": self.stale_missed_weight_threshold,
            },
            "items": [item.to_dict() for item in sorted(self.items.values(), key=lambda value: value.id)],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ObjectMemoryStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        thresholds = data.get("status_thresholds", {})
        store = cls(
            scene_id=data["scene_id"],
            freshness_tau_steps=data["freshness_tau_steps"],
            missing_confidence_threshold=float(thresholds.get("missing_confidence_threshold", 0.35)),
            missing_missed_weight_threshold=float(thresholds.get("missing_missed_weight_threshold", 6.0)),
            stale_missed_weight_threshold=float(thresholds.get("stale_missed_weight_threshold", 2.5)),
        )
        store.items = {
            item.id: item
            for item in (ObjectMemoryItem.from_dict(item_data) for item_data in data.get("items", []))
        }
        return store

    def summary(self) -> dict:
        per_class: dict[str, int] = {}
        by_source: dict[str, dict[str, float | int | dict[str, int]]] = {}
        for item in self.items.values():
            per_class[item.category] = per_class.get(item.category, 0) + 1
            source = item.source or "unknown"
            row = by_source.setdefault(
                source,
                {
                    "num_items": 0,
                    "active_items": 0,
                    "stale_items": 0,
                    "missing_items": 0,
                    "negative_evidence_total": 0,
                    "expected_visible_miss_items": 0,
                    "mean_confidence_sum": 0.0,
                    "mean_freshness_sum": 0.0,
                    "per_class": {},
                },
            )
            row["num_items"] = int(row["num_items"]) + 1
            row[f"{item.status}_items"] = int(row.get(f"{item.status}_items", 0)) + 1
            row["negative_evidence_total"] = int(row["negative_evidence_total"]) + item.negative_evidence_count
            row["expected_visible_miss_items"] = int(row["expected_visible_miss_items"]) + int(item.missed_observation_count > 0)
            row["mean_confidence_sum"] = float(row["mean_confidence_sum"]) + item.confidence
            row["mean_freshness_sum"] = float(row["mean_freshness_sum"]) + item.freshness
            source_per_class = row["per_class"]
            if isinstance(source_per_class, dict):
                source_per_class[item.category] = int(source_per_class.get(item.category, 0)) + 1
        freshness_values = [item.freshness for item in self.items.values()]
        confidence_values = [item.confidence for item in self.items.values()]
        for row in by_source.values():
            count = max(1, int(row["num_items"]))
            row["mean_confidence"] = float(row.pop("mean_confidence_sum")) / count
            row["mean_freshness"] = float(row.pop("mean_freshness_sum")) / count
        return {
            "scene_id": self.scene_id,
            "status_thresholds": {
                "missing_confidence_threshold": self.missing_confidence_threshold,
                "missing_missed_weight_threshold": self.missing_missed_weight_threshold,
                "stale_missed_weight_threshold": self.stale_missed_weight_threshold,
            },
            "num_items": len(self.items),
            "per_class": per_class,
            "by_source": by_source,
            "mean_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
            "mean_freshness": sum(freshness_values) / len(freshness_values) if freshness_values else 0.0,
            "active_items": sum(1 for item in self.items.values() if item.status == "active"),
            "stale_items": sum(1 for item in self.items.values() if item.status == "stale"),
            "missing_items": sum(1 for item in self.items.values() if item.status == "missing"),
            "expected_visible_miss_items": sum(1 for item in self.items.values() if item.missed_observation_count > 0),
            "negative_evidence_total": sum(item.negative_evidence_count for item in self.items.values()),
        }

    def _item_id(self, track: dict) -> str:
        return f"{track.get('category', 'object')}_{int(track['semantic_id'])}"

    def _item_from_track(self, track: dict, item_id: str) -> ObjectMemoryItem:
        centroid = track.get("centroid_xz") or [0.0, 0.0]
        return ObjectMemoryItem(
            id=item_id,
            semantic_id=int(track["semantic_id"]),
            object_id=track.get("object_id"),
            category=str(track["category"]),
            centroid_xz=(float(centroid[0]), float(centroid[1])),
            confidence=_clip(float(track.get("confidence", 0.0))),
            freshness=_clip(float(track.get("freshness", 1.0))),
            first_seen_step=int(track.get("first_seen_step", 0)),
            last_seen_step=int(track.get("last_seen_step", 0)),
            visible_steps=[int(step) for step in track.get("visible_steps", [])],
            footprint_cells=int(track.get("footprint_cells", 0)),
            fragmentation_count=int(track.get("fragmentation_count", 0)),
            source=str(track.get("source", "semantic_gt")),
        )

    def _freshness(self, last_seen_step: int, current_step: int) -> float:
        age = max(0, int(current_step) - int(last_seen_step))
        return _clip(exp(-age / self.freshness_tau_steps))

    def _status_from_evidence(self, item: ObjectMemoryItem) -> str:
        if (item.confidence < 0.2 and item.freshness < 0.35) or (
            item.confidence < self.missing_confidence_threshold
            and item.missed_observation_weight >= self.missing_missed_weight_threshold
        ):
            return "missing"
        if item.confidence < 0.45 or item.missed_observation_weight >= self.stale_missed_weight_threshold or item.freshness < 0.35:
            return "stale"
        return "active"


def build_store_from_semantic_report(
    semantic_report: dict,
    scene_id: str,
    freshness_tau_steps: float = 20.0,
) -> ObjectMemoryStore:
    store = ObjectMemoryStore(scene_id=scene_id, freshness_tau_steps=freshness_tau_steps)
    store.update_from_tracks(semantic_report.get("tracks", []))
    return store


def _positive_update(
    old: ObjectMemoryItem,
    incoming: ObjectMemoryItem,
    evidence_weight: float = 1.0,
) -> ObjectMemoryItem:
    evidence_weight = _clip(float(evidence_weight))
    total_conf = max(1e-6, old.confidence + incoming.confidence)
    centroid = (
        (old.centroid_xz[0] * old.confidence + incoming.centroid_xz[0] * incoming.confidence) / total_conf,
        (old.centroid_xz[1] * old.confidence + incoming.centroid_xz[1] * incoming.confidence) / total_conf,
    )
    visible_steps = sorted(set(old.visible_steps) | set(incoming.visible_steps))
    last_seen = max(old.last_seen_step, incoming.last_seen_step)
    target_confidence = _clip(0.75 * old.confidence + 0.25 * incoming.confidence + 0.05)
    confidence = _clip(old.confidence + evidence_weight * (target_confidence - old.confidence))
    freshness = _clip(old.freshness + evidence_weight * (1.0 - old.freshness))
    negative_evidence_count = old.negative_evidence_count
    if evidence_weight >= 0.5:
        negative_evidence_count = max(0, old.negative_evidence_count - 1)
    return ObjectMemoryItem(
        id=old.id,
        semantic_id=old.semantic_id,
        object_id=old.object_id or incoming.object_id,
        category=old.category,
        centroid_xz=centroid,
        confidence=confidence,
        freshness=freshness,
        first_seen_step=min(old.first_seen_step, incoming.first_seen_step),
        last_seen_step=last_seen,
        visible_steps=visible_steps,
        footprint_cells=max(old.footprint_cells, incoming.footprint_cells),
        fragmentation_count=max(_count_segments(visible_steps) - 1, old.fragmentation_count, incoming.fragmentation_count),
        missed_observation_count=0,
        negative_evidence_count=negative_evidence_count,
        not_observable_count=old.not_observable_count,
        missed_observation_weight=0.0,
        negative_evidence_weight=max(0.0, old.negative_evidence_weight - evidence_weight),
        not_observable_weight=old.not_observable_weight,
        positive_evidence_weight=old.positive_evidence_weight + evidence_weight,
        status="active",
        source=old.source,
    )


def _not_observable_update(item: ObjectMemoryItem, evidence_weight: float = 1.0) -> None:
    evidence_weight = _clip(float(evidence_weight))
    item.freshness = _clip(item.freshness * exp(-evidence_weight / 100.0))
    item.not_observable_count += 1
    item.not_observable_weight += evidence_weight


def _expected_visible_miss_update(item: ObjectMemoryItem, evidence_weight: float = 1.0) -> None:
    evidence_weight = _clip(float(evidence_weight))
    item.missed_observation_count += 1
    item.negative_evidence_count += 1
    item.missed_observation_weight += evidence_weight
    item.negative_evidence_weight += evidence_weight
    item.confidence = _clip(item.confidence - 0.12 * evidence_weight)
    item.freshness = _clip(item.freshness * exp(-evidence_weight / 25.0))


def _observability_for_item(observability: dict, item: ObjectMemoryItem) -> str:
    for key in (item.id, str(item.semantic_id), item.semantic_id):
        if key in observability:
            return str(observability[key])
    return "not_observable"


def _count_segments(steps: list[int]) -> int:
    if not steps:
        return 0
    segments = 1
    previous = steps[0]
    for step in steps[1:]:
        if step != previous + 1:
            segments += 1
        previous = step
    return segments


def _distance(a: WorldXZ, b: WorldXZ) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _status_penalty(status: str) -> float:
    return {"active": 0.0, "stale": 0.5, "missing": 1.2}.get(status, 0.5)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
