from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, sqrt
from typing import Dict, Iterable, List, Optional, Tuple


Position = Tuple[int, int]


@dataclass
class MemoryItem:
    id: str
    semantic_label: str
    bev_position: Position
    confidence: float
    freshness: float
    last_seen_time: int
    visit_count: int = 1
    negative_evidence_count: int = 0
    status: str = "active"
    context_id: str = "scene_i"
    source_view_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bev_position"] = list(self.bev_position)
        return data


@dataclass
class RetrievalResult:
    item: MemoryItem
    score: float
    score_parts: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "item": self.item.to_dict(),
            "score": self.score,
            "score_parts": self.score_parts,
        }


class SemanticSpatialMemory:
    """Minimal long-term semantic-spatial memory for Phase 1 smoke tests."""

    def __init__(self, grid_size: Tuple[int, int] = (12, 12), scene_id: str = "scene_i") -> None:
        self.grid_size = grid_size
        self.scene_id = scene_id
        self.items: Dict[str, MemoryItem] = {}
        self._next_id = 1

    def observe(
        self,
        label: str,
        position: Position,
        confidence: float,
        time: int,
        source_view_id: str,
        merge_radius: float = 1.5,
    ) -> MemoryItem:
        match = self._nearest_active_or_stale(label, position, merge_radius)
        if match is None:
            item = MemoryItem(
                id=self._new_id(label),
                semantic_label=label,
                bev_position=position,
                confidence=_clip(confidence),
                freshness=1.0,
                last_seen_time=time,
                context_id=self.scene_id,
                source_view_ids=[source_view_id],
            )
            self.items[item.id] = item
            return item

        match.confidence = _clip(0.65 * match.confidence + 0.35 * confidence + 0.10)
        match.freshness = 1.0
        match.last_seen_time = time
        match.visit_count += 1
        match.negative_evidence_count = max(0, match.negative_evidence_count - 1)
        match.status = "active"
        if source_view_id not in match.source_view_ids:
            match.source_view_ids.append(source_view_id)
        return match

    def weaken_expected_visible(
        self,
        label: str,
        expected_position: Position,
        time: int,
        visible_radius: float = 1.5,
        penalty: float = 0.35,
    ) -> Optional[MemoryItem]:
        item = self._nearest_by_label(label, expected_position, visible_radius)
        if item is None:
            return None
        item.confidence = _clip(item.confidence - penalty)
        item.freshness = _clip(item.freshness - 0.45)
        item.negative_evidence_count += 1
        item.last_seen_time = max(item.last_seen_time, time)
        if item.confidence <= 0.2:
            item.status = "missing"
        elif item.confidence <= 0.55:
            item.status = "stale"
        return item

    def relocate(
        self,
        label: str,
        old_position: Position,
        new_position: Position,
        confidence: float,
        time: int,
        source_view_id: str,
    ) -> MemoryItem:
        old = self._nearest_by_label(label, old_position, radius=2.0)
        if old is not None:
            old.status = "relocated"
            old.confidence = _clip(old.confidence - 0.30)
            old.freshness = _clip(old.freshness - 0.50)
            old.negative_evidence_count += 1
        return self.observe(
            label=label,
            position=new_position,
            confidence=confidence,
            time=time,
            source_view_id=source_view_id,
            merge_radius=1.0,
        )

    def decay_freshness(self, current_time: int, half_life: float = 6.0) -> None:
        for item in self.items.values():
            age = max(0, current_time - item.last_seen_time)
            item.freshness = _clip(exp(-age / half_life))
            if item.status == "active" and item.freshness < 0.35:
                item.status = "stale"

    def retrieve(
        self,
        goal_label: str,
        current_position: Position,
        top_k: int = 3,
        use_status_penalty: bool = True,
        use_freshness: bool = True,
    ) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for item in self.items.values():
            semantic_match = 1.0 if item.semantic_label == goal_label else 0.0
            distance = _distance(current_position, item.bev_position)
            spatial_proximity = 1.0 / (1.0 + distance)
            confidence = item.confidence
            freshness = item.freshness if use_freshness else 1.0
            status_penalty = _status_penalty(item.status) if use_status_penalty else 0.0
            score = (
                2.5 * semantic_match
                + 0.8 * spatial_proximity
                + 1.2 * confidence
                + 0.8 * freshness
                - status_penalty
            )
            results.append(
                RetrievalResult(
                    item=item,
                    score=round(score, 4),
                    score_parts={
                        "semantic_match": semantic_match,
                        "spatial_proximity": round(spatial_proximity, 4),
                        "confidence": round(confidence, 4),
                        "freshness": round(freshness, 4),
                        "status_penalty": round(status_penalty, 4),
                    },
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def snapshot(self) -> List[dict]:
        return [item.to_dict() for item in sorted(self.items.values(), key=lambda x: x.id)]

    def _new_id(self, label: str) -> str:
        value = f"{label}_{self._next_id:03d}"
        self._next_id += 1
        return value

    def _nearest_active_or_stale(
        self, label: str, position: Position, radius: float
    ) -> Optional[MemoryItem]:
        candidates = [
            item
            for item in self.items.values()
            if item.semantic_label == label
            and item.status in {"active", "stale"}
            and _distance(item.bev_position, position) <= radius
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: _distance(item.bev_position, position))

    def _nearest_by_label(
        self, label: str, position: Position, radius: float
    ) -> Optional[MemoryItem]:
        candidates = [
            item
            for item in self.items.values()
            if item.semantic_label == label and _distance(item.bev_position, position) <= radius
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: _distance(item.bev_position, position))


def retrieval_hit_at_k(results: Iterable[RetrievalResult], label: str, position: Position, k: int) -> bool:
    for result in list(results)[:k]:
        if result.item.semantic_label == label and result.item.bev_position == position:
            return True
    return False


def _distance(a: Position, b: Position) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _status_penalty(status: str) -> float:
    return {
        "active": 0.0,
        "stale": 0.8,
        "missing": 1.5,
        "relocated": 1.2,
    }.get(status, 0.5)
