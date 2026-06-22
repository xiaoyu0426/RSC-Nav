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

    def __init__(self, scene_id: str, freshness_tau_steps: float = 20.0) -> None:
        self.scene_id = scene_id
        self.freshness_tau_steps = max(1.0, float(freshness_tau_steps))
        self.items: dict[str, ObjectMemoryItem] = {}

    def update_from_tracks(self, tracks: Iterable[dict], current_step: Optional[int] = None) -> dict:
        tracks = list(tracks)
        if current_step is None:
            current_step = max((int(track.get("last_seen_step", 0)) for track in tracks), default=0)

        created = 0
        updated = 0
        for track in tracks:
            semantic_id = int(track["semantic_id"])
            item_id = self._item_id(track)
            incoming = self._item_from_track(track, item_id=item_id)
            incoming.freshness = self._freshness(incoming.last_seen_step, current_step)
            incoming.status = _status_from_confidence_freshness(incoming.confidence, incoming.freshness)
            if item_id in self.items:
                self.items[item_id] = _merge_item(self.items[item_id], incoming, current_step, self.freshness_tau_steps)
                updated += 1
            else:
                self.items[item_id] = incoming
                created += 1

        decayed = self.decay(current_step=current_step)
        return {
            "created": created,
            "updated": updated,
            "decayed": decayed,
            "current_step": int(current_step),
            "num_items": len(self.items),
        }

    def decay(self, current_step: int) -> int:
        decayed = 0
        for item in self.items.values():
            old_freshness = item.freshness
            item.freshness = self._freshness(item.last_seen_step, current_step)
            item.status = _status_from_confidence_freshness(item.confidence, item.freshness)
            if item.freshness != old_freshness:
                decayed += 1
        return decayed

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
            "items": [item.to_dict() for item in sorted(self.items.values(), key=lambda value: value.id)],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ObjectMemoryStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(scene_id=data["scene_id"], freshness_tau_steps=data["freshness_tau_steps"])
        store.items = {
            item.id: item
            for item in (ObjectMemoryItem.from_dict(item_data) for item_data in data.get("items", []))
        }
        return store

    def summary(self) -> dict:
        per_class: dict[str, int] = {}
        for item in self.items.values():
            per_class[item.category] = per_class.get(item.category, 0) + 1
        freshness_values = [item.freshness for item in self.items.values()]
        confidence_values = [item.confidence for item in self.items.values()]
        return {
            "scene_id": self.scene_id,
            "num_items": len(self.items),
            "per_class": per_class,
            "mean_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
            "mean_freshness": sum(freshness_values) / len(freshness_values) if freshness_values else 0.0,
            "active_items": sum(1 for item in self.items.values() if item.status == "active"),
            "stale_items": sum(1 for item in self.items.values() if item.status == "stale"),
            "missing_items": sum(1 for item in self.items.values() if item.status == "missing"),
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
        )

    def _freshness(self, last_seen_step: int, current_step: int) -> float:
        age = max(0, int(current_step) - int(last_seen_step))
        return _clip(exp(-age / self.freshness_tau_steps))


def build_store_from_semantic_report(
    semantic_report: dict,
    scene_id: str,
    freshness_tau_steps: float = 20.0,
) -> ObjectMemoryStore:
    store = ObjectMemoryStore(scene_id=scene_id, freshness_tau_steps=freshness_tau_steps)
    store.update_from_tracks(semantic_report.get("tracks", []))
    return store


def _merge_item(
    old: ObjectMemoryItem,
    incoming: ObjectMemoryItem,
    current_step: int,
    freshness_tau_steps: float,
) -> ObjectMemoryItem:
    total_conf = max(1e-6, old.confidence + incoming.confidence)
    centroid = (
        (old.centroid_xz[0] * old.confidence + incoming.centroid_xz[0] * incoming.confidence) / total_conf,
        (old.centroid_xz[1] * old.confidence + incoming.centroid_xz[1] * incoming.confidence) / total_conf,
    )
    visible_steps = sorted(set(old.visible_steps) | set(incoming.visible_steps))
    last_seen = max(old.last_seen_step, incoming.last_seen_step)
    confidence = _clip(0.65 * old.confidence + 0.35 * incoming.confidence + 0.05)
    freshness = _clip(exp(-max(0, current_step - last_seen) / max(1.0, freshness_tau_steps)))
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
        status=_status_from_confidence_freshness(confidence, freshness),
        source=old.source,
    )


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


def _status_from_confidence_freshness(confidence: float, freshness: float) -> str:
    if confidence < 0.2 or freshness < 0.08:
        return "missing"
    if confidence < 0.45 or freshness < 0.35:
        return "stale"
    return "active"


def _distance(a: WorldXZ, b: WorldXZ) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _status_penalty(status: str) -> float:
    return {"active": 0.0, "stale": 0.5, "missing": 1.2}.get(status, 0.5)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
