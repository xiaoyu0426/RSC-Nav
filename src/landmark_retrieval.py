from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt
from typing import Any, Iterable


WorldXZ = tuple[float, float]


DEFAULT_ALIAS_TABLE: dict[str, list[str]] = {
    "seat": ["chair", "sofa"],
    "seating": ["chair", "sofa"],
    "desk": ["table"],
    "mug": ["cup"],
    "entrance": ["door"],
    "exit": ["door"],
}

STATUS_SCORES = {
    "active": 1.0,
    "stale": 0.5,
    "relocated": 0.4,
    "missing": 0.0,
}


@dataclass
class LandmarkNode:
    id: str
    label: str
    aliases: list[str]
    bev_position: WorldXZ
    confidence: float
    freshness: float
    status: str
    context_id: str
    source_object_ids: list[str] = field(default_factory=list)
    visit_count: int = 1
    last_seen_step: int = 0
    source: str = "object_memory"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bev_position"] = [float(self.bev_position[0]), float(self.bev_position[1])]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LandmarkNode":
        item = dict(data)
        item["bev_position"] = _position_from_any(item.get("bev_position", item.get("centroid_xz", [0.0, 0.0])))
        item["aliases"] = [str(value) for value in item.get("aliases", [])]
        item["source_object_ids"] = [str(value) for value in item.get("source_object_ids", [])]
        return cls(**item)


@dataclass
class RetrievalResult:
    query: str
    landmark_id: str
    label: str
    bev_position: WorldXZ
    final_score: float
    score_breakdown: dict[str, float]
    status: str
    confidence: float
    freshness: float
    context_id: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bev_position"] = [float(self.bev_position[0]), float(self.bev_position[1])]
        return data


def build_landmark_nodes(
    objects: Iterable[Any],
    context_id: str = "scene_i",
    merge_radius_m: float = 0.75,
    alias_table: dict[str, list[str]] | None = None,
) -> list[LandmarkNode]:
    aliases = alias_table or DEFAULT_ALIAS_TABLE
    nodes: list[LandmarkNode] = []
    for obj in objects:
        normalized = _normalize_object(obj, default_context_id=context_id, alias_table=aliases)
        if normalized is None:
            continue
        match = _find_merge_candidate(nodes, normalized, merge_radius_m, aliases)
        if match is None:
            nodes.append(normalized)
        else:
            _merge_node(match, normalized)
    nodes.sort(key=lambda node: node.id)
    return nodes


def retrieve_landmarks(
    query: str,
    landmarks: Iterable[LandmarkNode],
    context_id: str = "scene_i",
    top_k: int = 5,
    alias_table: dict[str, list[str]] | None = None,
    weights: dict[str, float] | None = None,
    include_irrelevant_if_matched: bool = False,
) -> list[RetrievalResult]:
    aliases = alias_table or DEFAULT_ALIAS_TABLE
    weights = weights or {
        "semantic": 0.40,
        "confidence": 0.20,
        "freshness": 0.15,
        "status": 0.15,
        "context": 0.10,
        "source": 0.00,
    }
    results: list[RetrievalResult] = []
    for node in landmarks:
        semantic_match = _semantic_match(query, node, aliases)
        status_score = STATUS_SCORES.get(node.status, 0.4)
        context_match = _context_match(context_id, node.context_id)
        source_bonus = 1.0 if node.source == "imported" else 0.0
        final_score = (
            weights["semantic"] * semantic_match
            + weights["confidence"] * _clip(node.confidence)
            + weights["freshness"] * _clip(node.freshness)
            + weights["status"] * status_score
            + weights["context"] * context_match
            + weights["source"] * source_bonus
        )
        results.append(
            RetrievalResult(
                query=query,
                landmark_id=node.id,
                label=node.label,
                bev_position=node.bev_position,
                final_score=round(final_score, 4),
                score_breakdown={
                    "semantic_match": round(semantic_match, 4),
                    "confidence": round(_clip(node.confidence), 4),
                    "freshness": round(_clip(node.freshness), 4),
                    "status_score": round(status_score, 4),
                    "context_match": round(context_match, 4),
                    "source_bonus": round(source_bonus, 4),
                    "final_score": round(final_score, 4),
                },
                status=node.status,
                confidence=round(_clip(node.confidence), 4),
                freshness=round(_clip(node.freshness), 4),
                context_id=node.context_id,
            )
        )
    if not include_irrelevant_if_matched:
        results = [result for result in results if result.score_breakdown["semantic_match"] > 0.0]
    results.sort(key=lambda result: result.final_score, reverse=True)
    return results[: max(1, int(top_k))]


def nodes_to_json(nodes: Iterable[LandmarkNode]) -> list[dict[str, Any]]:
    return [node.to_dict() for node in nodes]


def results_to_json(results: Iterable[RetrievalResult]) -> list[dict[str, Any]]:
    return [result.to_dict() for result in results]


def _normalize_object(
    obj: Any,
    default_context_id: str,
    alias_table: dict[str, list[str]],
) -> LandmarkNode | None:
    data = obj.to_dict() if hasattr(obj, "to_dict") else dict(obj)
    label = str(data.get("label") or data.get("category") or data.get("semantic_label") or "").strip().lower()
    if not label:
        return None
    object_id = str(data.get("id") or data.get("object_id") or f"{label}_{abs(hash(str(data))) % 100000}")
    context_id = str(data.get("context_id") or default_context_id)
    position = _position_from_any(data.get("bev_position", data.get("centroid_xz", data.get("position", [0.0, 0.0]))))
    confidence = _clip(float(data.get("confidence", 0.5)))
    freshness = _clip(float(data.get("freshness", 1.0)))
    status = str(data.get("status", "active")).lower()
    source = str(data.get("source", "object_memory"))
    return LandmarkNode(
        id=f"lm_{context_id}_{label}_{object_id}".replace(" ", "_"),
        label=label,
        aliases=_aliases_for_label(label, alias_table),
        bev_position=position,
        confidence=confidence,
        freshness=freshness,
        status=status,
        context_id=context_id,
        source_object_ids=[object_id],
        visit_count=int(data.get("visit_count", 1)),
        last_seen_step=int(data.get("last_seen_step", data.get("last_seen_time", 0))),
        source=source,
    )


def _find_merge_candidate(
    nodes: list[LandmarkNode],
    incoming: LandmarkNode,
    merge_radius_m: float,
    alias_table: dict[str, list[str]],
) -> LandmarkNode | None:
    for node in nodes:
        if node.context_id != incoming.context_id:
            continue
        if not _labels_match_for_merge(node.label, incoming.label, alias_table):
            continue
        if _distance(node.bev_position, incoming.bev_position) <= merge_radius_m:
            return node
    return None


def _merge_node(target: LandmarkNode, incoming: LandmarkNode) -> None:
    total = max(1e-6, target.confidence + incoming.confidence)
    target.bev_position = (
        (target.bev_position[0] * target.confidence + incoming.bev_position[0] * incoming.confidence) / total,
        (target.bev_position[1] * target.confidence + incoming.bev_position[1] * incoming.confidence) / total,
    )
    target.confidence = max(target.confidence, incoming.confidence)
    target.freshness = max(target.freshness, incoming.freshness)
    target.status = _better_status(target.status, incoming.status)
    target.source_object_ids = sorted(set(target.source_object_ids + incoming.source_object_ids))
    target.visit_count += incoming.visit_count
    target.last_seen_step = max(target.last_seen_step, incoming.last_seen_step)
    target.source = "mixed" if target.source != incoming.source else target.source


def _semantic_match(query: str, node: LandmarkNode, alias_table: dict[str, list[str]]) -> float:
    normalized = _normalize_text(query)
    if normalized == node.label:
        return 1.0
    query_labels = set(alias_table.get(normalized, []))
    if node.label in query_labels:
        return 0.8
    if normalized in node.aliases:
        return 0.8
    if normalized and (normalized in node.label or node.label in normalized):
        return 0.6
    return 0.0


def _labels_match_for_merge(left: str, right: str, alias_table: dict[str, list[str]]) -> bool:
    return left == right


def _context_match(query_context: str, node_context: str) -> float:
    if not query_context or query_context == "unknown":
        return 0.5
    if query_context == node_context:
        return 1.0
    return 0.0


def _aliases_for_label(label: str, alias_table: dict[str, list[str]]) -> list[str]:
    values = [alias for alias, labels in alias_table.items() if label in labels]
    return sorted(set(values))


def _better_status(left: str, right: str) -> str:
    rank = {"active": 3, "stale": 2, "relocated": 1, "missing": 0}
    return left if rank.get(left, 1) >= rank.get(right, 1) else right


def _normalize_text(value: str) -> str:
    value = value.lower().strip()
    for prefix in ("find ", "where is the ", "where is ", "go to the ", "go to "):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    for token in (" in context_a", " in context_b"):
        value = value.replace(token, "")
    return value.strip()


def _position_from_any(value: Any) -> WorldXZ:
    if isinstance(value, dict):
        return (float(value.get("x", 0.0)), float(value.get("z", value.get("y", 0.0))))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return (0.0, 0.0)


def _distance(a: WorldXZ, b: WorldXZ) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
