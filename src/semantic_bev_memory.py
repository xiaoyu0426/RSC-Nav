from __future__ import annotations

from typing import Any

import numpy as np

from dense_bev_mapper import DenseBEVMapper, depth_to_world_samples, _bresenham


SEMANTIC_COLORS = {
    "wall": "#2f2f2f",
    "door": "#e15759",
    "table": "#f28e2b",
    "chair": "#59a14f",
    "bed": "#1f77b4",
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
        self.prior_evidence = np.zeros_like(self.evidence)
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
            bounds = _object_bounds(obj)
            self.semantic_id_to_gt[semantic_id] = {
                "object_id": str(obj.id),
                "semantic_id": semantic_id,
                "category": category,
                "raw_category": obj.category.name(),
                "gt_center_xz": bounds["center_xz"],
                "gt_center_xyz": bounds["center_xyz"],
                "height_range_y": bounds["height_range_y"],
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
        evidence_weight: float = 1.0,
        prior_decay_weight: float = 0.0,
    ) -> dict[str, float | int]:
        evidence_weight = float(np.clip(evidence_weight, 0.0, 1.0))
        prior_decay_weight = float(np.clip(prior_decay_weight, 0.0, 3.0))
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
            return {"prior_decayed_cells": 0, "prior_evidence_before": float(self.prior_evidence.sum()), "prior_evidence_after": float(self.prior_evidence.sum())}
        rows = np.clip(samples["rows"], 0, semantic.shape[0] - 1)
        cols = np.clip(samples["cols"], 0, semantic.shape[1] - 1)
        semantic_ids = semantic[rows, cols].astype(np.int64)
        rel_y = points_world[:, 1] - float(floor_y)
        height_mask = np.logical_and(rel_y >= -0.3, rel_y <= 2.5)
        seen_ids: set[int] = set()
        self.latest_step = max(self.latest_step, int(step))
        prior_before = float(self.prior_evidence.sum())
        prior_decayed_cells: set[tuple[int, int]] = set()

        for semantic_id, point, keep in zip(semantic_ids, points_world, height_mask):
            if not keep:
                continue
            semantic_id = int(semantic_id)
            category = self.semantic_id_to_class.get(semantic_id)
            cell = self.mapper.world_to_grid((float(point[0]), float(point[2])))
            if cell is None:
                continue
            if prior_decay_weight > 0.0 and cell not in prior_decayed_cells:
                self._decay_prior_at_cell(cell, prior_decay_weight, surface=True)
                prior_decayed_cells.add(cell)
            if category is None:
                continue
            seen_ids.add(semantic_id)
            class_index = self.category_to_index[category]
            self.evidence[class_index, cell[0], cell[1]] += evidence_weight
            stats = self.instance_stats.setdefault(
                semantic_id,
                {
                    "semantic_id": semantic_id,
                    "category": category,
                    "count": 0.0,
                    "sum_x": 0.0,
                    "sum_z": 0.0,
                    "first_seen_step": int(step),
                    "last_seen_step": int(step),
                    "cells": set(),
                },
            )
            stats["count"] += evidence_weight
            stats["sum_x"] += float(point[0]) * evidence_weight
            stats["sum_z"] += float(point[2]) * evidence_weight
            stats["last_seen_step"] = int(step)
            stats["cells"].add(cell)
        if prior_decay_weight > 0.0:
            sensor_cell = self.mapper.world_to_grid((float(sensor_position_xyz[0]), float(sensor_position_xyz[2])))
            if sensor_cell is not None:
                for point, keep in zip(points_world, height_mask):
                    if not keep:
                        continue
                    endpoint = self.mapper.world_to_grid((float(point[0]), float(point[2])))
                    if endpoint is None:
                        continue
                    for free_cell in _bresenham(sensor_cell, endpoint):
                        if free_cell == endpoint:
                            break
                        if not self.mapper.in_bounds(free_cell):
                            continue
                        if free_cell in prior_decayed_cells:
                            continue
                        self._decay_prior_at_cell(free_cell, prior_decay_weight, surface=False)
                        prior_decayed_cells.add(free_cell)
        if seen_ids:
            self.frame_seen.setdefault(int(step), set()).update(seen_ids)
        return {
            "prior_decayed_cells": len(prior_decayed_cells),
            "prior_evidence_before": prior_before,
            "prior_evidence_after": float(self.prior_evidence.sum()),
        }

    def _decay_prior_at_cell(self, cell: tuple[int, int], weight: float, surface: bool) -> None:
        if self.prior_evidence.size == 0:
            return
        rate = 1.15 if surface else 0.85
        factor = float(np.exp(-rate * float(weight)))
        self.prior_evidence[:, cell[0], cell[1]] *= factor
        cell_values = self.prior_evidence[:, cell[0], cell[1]]
        cell_values[cell_values < 0.05] = 0.0

    def combined_evidence(self) -> np.ndarray:
        if self.prior_evidence.shape != self.evidence.shape:
            self.prior_evidence = np.zeros_like(self.evidence)
        return self.evidence + self.prior_evidence

    def seen_ids_for_step(self, step: int) -> set[int]:
        return set(self.frame_seen.get(int(step), set()))

    def expected_visible_ids(
        self,
        depth: np.ndarray,
        sensor_position_xyz: np.ndarray,
        sensor_rotation,
        hfov_deg: float,
        occlusion_margin_m: float = 0.25,
        min_projected_points: int = 2,
        min_unoccluded_fraction: float = 0.55,
        patch_radius_px: int = 2,
        min_patch_valid_fraction: float = 0.5,
    ) -> set[int]:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            return set()
        height, width = depth.shape
        fx = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
        fy = fx
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        sensor_xyz = np.asarray(sensor_position_xyz, dtype=np.float32).reshape(3)
        axes = _camera_axes(sensor_rotation)
        expected = set()

        for semantic_id, gt in self.semantic_id_to_gt.items():
            center_xyz = gt.get("gt_center_xyz")
            height_range = gt.get("height_range_y")
            size_xyz = gt.get("sizes_xyz")
            if center_xyz is None or height_range is None:
                continue
            points = _object_visibility_points(center_xyz, height_range, size_xyz)
            projected = [
                _project_world_point(point, sensor_xyz, axes, fx, fy, cx, cy, width, height)
                for point in points
            ]
            projected = [item for item in projected if item is not None]
            if len(projected) < max(1, int(min_projected_points)):
                continue
            unoccluded = 0
            for row, col, distance in projected:
                observed_depth = _patch_depth(depth, row, col, patch_radius_px, min_patch_valid_fraction)
                if observed_depth is not None and observed_depth + occlusion_margin_m >= distance:
                    unoccluded += 1
            if unoccluded >= max(2, int(np.ceil(len(projected) * float(min_unoccluded_fraction)))):
                expected.add(int(semantic_id))
        return expected

    def semantic_state(self) -> np.ndarray:
        state = np.full(self.mapper.config.grid_size, -1, dtype=np.int16)
        if not self.categories:
            return state
        combined = self.combined_evidence()
        max_evidence = combined.max(axis=0)
        state[max_evidence > 0] = np.argmax(combined, axis=0)[max_evidence > 0]
        return state

    def confidence(self) -> np.ndarray:
        if not self.categories:
            return np.zeros(self.mapper.config.grid_size, dtype=np.float32)
        return np.clip(self.combined_evidence().max(axis=0) / self.confidence_saturation, 0.0, 1.0)

    def report(self) -> dict:
        tracks = []
        centroid_errors = []
        per_class_errors: dict[str, list[float]] = {category: [] for category in self.categories}
        for semantic_id, stats in sorted(self.instance_stats.items()):
            count = float(stats["count"])
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
                    "count": round(count, 4),
                    "centroid_xz": centroid,
                    "gt_center_xz": gt_center,
                    "gt_center_xyz": gt.get("gt_center_xyz"),
                    "height_range_y": gt.get("height_range_y"),
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
            "live_semantic_cells": int((self.evidence.max(axis=0) > 0).sum()) if self.evidence.size else 0,
            "prior_semantic_cells": int((self.prior_evidence.max(axis=0) > 0).sum()) if self.prior_evidence.size else 0,
            "prior_evidence_total": float(self.prior_evidence.sum()) if self.prior_evidence.size else 0.0,
            "live_evidence_total": float(self.evidence.sum()) if self.evidence.size else 0.0,
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


def _object_bounds(obj) -> dict[str, list[float] | None]:
    aabb = getattr(obj, "aabb", None)
    center = getattr(aabb, "center", None)
    if callable(center):
        center = center()
    if center is None:
        return {"center_xz": None, "center_xyz": None, "height_range_y": None}
    arr = np.asarray(center, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return {"center_xz": None, "center_xyz": None, "height_range_y": None}

    sizes = getattr(aabb, "sizes", None)
    if callable(sizes):
        sizes = sizes()
    size_arr = np.asarray(sizes, dtype=np.float32).reshape(-1) if sizes is not None else np.empty((0,), dtype=np.float32)
    half_height = float(size_arr[1] * 0.5) if size_arr.size >= 2 and np.isfinite(size_arr[1]) else 0.6
    center_y = float(arr[1])
    return {
        "center_xz": [float(arr[0]), float(arr[2])],
        "center_xyz": [float(arr[0]), center_y, float(arr[2])],
        "height_range_y": [center_y - half_height, center_y + half_height],
        "sizes_xyz": [float(value) for value in size_arr[:3]] if size_arr.size >= 3 else None,
    }


def _object_visibility_points(center_xyz, height_range_y, size_xyz=None) -> list[np.ndarray]:
    center = np.asarray(center_xyz, dtype=np.float32).reshape(3)
    low_y, high_y = [float(value) for value in height_range_y]
    mid_y = float(center[1])
    size = np.asarray(size_xyz, dtype=np.float32).reshape(-1) if size_xyz is not None else np.empty((0,), dtype=np.float32)
    x_radius = float(np.clip(size[0] * 0.25, 0.05, 0.45)) if size.size >= 1 and np.isfinite(size[0]) else 0.15
    z_radius = float(np.clip(size[2] * 0.25, 0.05, 0.45)) if size.size >= 3 and np.isfinite(size[2]) else 0.15
    offsets = [(0.0, 0.0), (-x_radius, 0.0), (x_radius, 0.0), (0.0, -z_radius), (0.0, z_radius)]
    points = []
    for y in (low_y, mid_y, high_y):
        for dx, dz in offsets:
            points.append(np.asarray([center[0] + dx, y, center[2] + dz], dtype=np.float32))
    return points


def _patch_depth(depth: np.ndarray, row: int, col: int, radius: int, min_valid_fraction: float) -> float | None:
    radius = max(0, int(radius))
    r0 = max(0, int(row) - radius)
    r1 = min(depth.shape[0], int(row) + radius + 1)
    c0 = max(0, int(col) - radius)
    c1 = min(depth.shape[1], int(col) + radius + 1)
    patch = np.asarray(depth[r0:r1, c0:c1], dtype=np.float32)
    if patch.size == 0:
        return None
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size / patch.size < float(min_valid_fraction):
        return None
    return float(np.median(valid))


def _camera_axes(rotation) -> dict[str, np.ndarray]:
    return {
        "right": _rotate_vector(rotation, [1.0, 0.0, 0.0]),
        "up": _rotate_vector(rotation, [0.0, 1.0, 0.0]),
        "forward": _rotate_vector(rotation, [0.0, 0.0, -1.0]),
    }


def _rotate_vector(rotation, vector) -> np.ndarray:
    if hasattr(rotation, "transform_vector"):
        return np.asarray(rotation.transform_vector(vector), dtype=np.float32)
    try:
        import quaternion as np_quaternion

        matrix = np.asarray(np_quaternion.as_rotation_matrix(rotation), dtype=np.float32)
        return (np.asarray(vector, dtype=np.float32).reshape(1, 3) @ matrix.T).reshape(3)
    except Exception as exc:
        raise TypeError(f"Unsupported sensor rotation type: {type(rotation)!r}") from exc


def _project_world_point(
    point: np.ndarray,
    sensor_xyz: np.ndarray,
    axes: dict[str, np.ndarray],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[int, int, float] | None:
    rel = np.asarray(point, dtype=np.float32).reshape(3) - sensor_xyz
    distance = float(np.dot(rel, axes["forward"]))
    if distance <= 0.05:
        return None
    x_cam = float(np.dot(rel, axes["right"]))
    y_cam = float(np.dot(rel, axes["up"]))
    col = int(round(cx + fx * x_cam / distance))
    row = int(round(cy - fy * y_cam / distance))
    if not (0 <= row < height and 0 <= col < width):
        return None
    return row, col, distance
