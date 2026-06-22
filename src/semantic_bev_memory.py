from __future__ import annotations

from typing import Any

import numpy as np

from dense_bev_mapper import DenseBEVMapper, depth_to_world_samples


SEMANTIC_COLORS = {
    "wall": "#2f2f2f",
    "door": "#1f77b4",
    "table": "#f28e2b",
    "chair": "#59a14f",
}


class SemanticBEVAccumulator:
    """Accumulates Habitat semantic GT into allocentric BEV evidence."""

    def __init__(
        self,
        mapper: DenseBEVMapper,
        semantic_scene: Any,
        categories: list[str],
        confidence_saturation: float,
        freshness_tau_steps: float,
    ) -> None:
        self.mapper = mapper
        self.categories = categories
        self.category_to_index = {category: idx for idx, category in enumerate(categories)}
        self.semantic_id_to_class: dict[int, str] = {}
        self.semantic_id_to_gt: dict[int, dict] = {}
        self.evidence = np.zeros((len(categories), *mapper.config.grid_size), dtype=np.float32)
        self.instance_stats: dict[int, dict] = {}
        self.confidence_saturation = max(1.0, float(confidence_saturation))
        self.freshness_tau_steps = max(1.0, float(freshness_tau_steps))
        self.frame_seen: dict[int, set[int]] = {}
        self.latest_step = 0
        self._index_scene_objects(semantic_scene)

    def _index_scene_objects(self, semantic_scene: Any) -> None:
        for obj in getattr(semantic_scene, "objects", []):
            if obj is None or obj.category is None:
                continue
            category = _match_category(obj.category.name(), self.categories)
            if category is None:
                continue
            semantic_id = int(obj.semantic_id)
            self.semantic_id_to_class[semantic_id] = category
            center = _object_center(obj)
            self.semantic_id_to_gt[semantic_id] = {
                "object_id": str(obj.id),
                "semantic_id": semantic_id,
                "category": category,
                "raw_category": obj.category.name(),
                "gt_center_xz": center,
            }

    def update_from_observation(
        self,
        depth: np.ndarray,
        semantic: np.ndarray,
        sensor_position_xyz: np.ndarray,
        sensor_rotation,
        floor_y: float,
        hfov_deg: float,
        step: int,
    ) -> None:
        samples = depth_to_world_samples(
            depth=depth,
            sensor_position_xyz=sensor_position_xyz,
            sensor_rotation=sensor_rotation,
            hfov_deg=hfov_deg,
            stride=self.mapper.config.sample_stride,
            min_depth_m=self.mapper.config.min_depth_m,
            max_depth_m=self.mapper.config.max_depth_m,
        )
        points_world = samples["points_world"]
        if points_world.size == 0:
            return
        rows = np.clip(samples["rows"], 0, semantic.shape[0] - 1)
        cols = np.clip(samples["cols"], 0, semantic.shape[1] - 1)
        semantic_ids = semantic[rows, cols].astype(np.int64)
        rel_y = points_world[:, 1] - float(floor_y)
        height_mask = np.logical_and(rel_y >= -0.3, rel_y <= 2.5)
        seen_ids: set[int] = set()
        self.latest_step = max(self.latest_step, int(step))

        for semantic_id, point, keep in zip(semantic_ids, points_world, height_mask):
            if not keep:
                continue
            semantic_id = int(semantic_id)
            category = self.semantic_id_to_class.get(semantic_id)
            if category is None:
                continue
            cell = self.mapper.world_to_grid((float(point[0]), float(point[2])))
            if cell is None:
                continue
            seen_ids.add(semantic_id)
            class_index = self.category_to_index[category]
            self.evidence[class_index, cell[0], cell[1]] += 1.0
            stats = self.instance_stats.setdefault(
                semantic_id,
                {
                    "semantic_id": semantic_id,
                    "category": category,
                    "count": 0,
                    "sum_x": 0.0,
                    "sum_z": 0.0,
                    "first_seen_step": int(step),
                    "last_seen_step": int(step),
                    "cells": set(),
                },
            )
            stats["count"] += 1
            stats["sum_x"] += float(point[0])
            stats["sum_z"] += float(point[2])
            stats["last_seen_step"] = int(step)
            stats["cells"].add(cell)
        if seen_ids:
            self.frame_seen.setdefault(int(step), set()).update(seen_ids)

    def semantic_state(self) -> np.ndarray:
        state = np.full(self.mapper.config.grid_size, -1, dtype=np.int16)
        if not self.categories:
            return state
        max_evidence = self.evidence.max(axis=0)
        state[max_evidence > 0] = np.argmax(self.evidence, axis=0)[max_evidence > 0]
        return state

    def confidence(self) -> np.ndarray:
        if not self.categories:
            return np.zeros(self.mapper.config.grid_size, dtype=np.float32)
        return np.clip(self.evidence.max(axis=0) / self.confidence_saturation, 0.0, 1.0)

    def report(self) -> dict:
        tracks = []
        centroid_errors = []
        per_class_errors: dict[str, list[float]] = {category: [] for category in self.categories}
        for semantic_id, stats in sorted(self.instance_stats.items()):
            count = int(stats["count"])
            centroid = [stats["sum_x"] / count, stats["sum_z"] / count] if count else [None, None]
            gt = self.semantic_id_to_gt.get(semantic_id, {})
            gt_center = gt.get("gt_center_xz")
            error = None
            if gt_center is not None and centroid[0] is not None:
                error = float(np.linalg.norm(np.asarray(centroid) - np.asarray(gt_center)))
                centroid_errors.append(error)
                per_class_errors[stats["category"]].append(error)
            visible_steps = self._visible_steps(semantic_id)
            fragmentation_count = max(0, _count_segments(visible_steps) - 1)
            age_steps = max(0, int(self.latest_step) - int(stats["last_seen_step"]))
            freshness = float(np.exp(-age_steps / self.freshness_tau_steps))
            tracks.append(
                {
                    "semantic_id": semantic_id,
                    "object_id": gt.get("object_id"),
                    "category": stats["category"],
                    "count": count,
                    "centroid_xz": centroid,
                    "gt_center_xz": gt_center,
                    "centroid_error_m": error,
                    "footprint_cells": len(stats["cells"]),
                    "confidence": min(1.0, count / self.confidence_saturation),
                    "freshness": freshness,
                    "age_steps": age_steps,
                    "visible_steps": visible_steps,
                    "visibility_segments": _count_segments(visible_steps),
                    "fragmentation_count": fragmentation_count,
                    "first_seen_step": int(stats["first_seen_step"]),
                    "last_seen_step": int(stats["last_seen_step"]),
                }
            )

        state = self.semantic_state()
        per_class_cells = {
            category: int((state == idx).sum())
            for category, idx in self.category_to_index.items()
        }
        per_class_mean_error = {
            category: (float(np.mean(values)) if values else None)
            for category, values in per_class_errors.items()
        }
        return {
            "categories": self.categories,
            "indexed_target_instances": len(self.semantic_id_to_class),
            "observed_target_instances": len(tracks),
            "semantic_cells": int((state >= 0).sum()),
            "per_class_cells": per_class_cells,
            "mean_centroid_error_m": float(np.mean(centroid_errors)) if centroid_errors else None,
            "per_class_mean_centroid_error_m": per_class_mean_error,
            "mean_fragmentation_count": float(np.mean([track["fragmentation_count"] for track in tracks])) if tracks else 0.0,
            "id_switches_upper_bound": 0,
            "mean_freshness": float(np.mean([track["freshness"] for track in tracks])) if tracks else 0.0,
            "tracks": tracks,
        }

    def _visible_steps(self, semantic_id: int) -> list[int]:
        return sorted(
            step
            for step, ids in self.frame_seen.items()
            if semantic_id in ids
        )


def semantic_array(raw_semantic) -> np.ndarray:
    if raw_semantic is None:
        raise RuntimeError("missing semantic observation")
    semantic = np.asarray(raw_semantic)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[:, :, 0]
    return semantic.astype(np.int64)


def _match_category(raw_name: str, categories: list[str]) -> str | None:
    name = (raw_name or "").lower()
    for category in categories:
        if category in name:
            return category
    return None


def _count_segments(steps: list[int]) -> int:
    if not steps:
        return 0
    segments = 1
    prev = steps[0]
    for step in steps[1:]:
        if step != prev + 1:
            segments += 1
        prev = step
    return segments


def _object_center(obj) -> list[float] | None:
    aabb = getattr(obj, "aabb", None)
    center = getattr(aabb, "center", None)
    if callable(center):
        center = center()
    if center is None:
        return None
    arr = np.asarray(center, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[2])]
