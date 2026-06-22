from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


GridCoord = Tuple[int, int]
WorldXZ = Tuple[float, float]


@dataclass
class DenseBEVConfig:
    grid_size: Tuple[int, int] = (240, 240)
    resolution: float = 0.05
    camera_height_m: float = 1.5
    max_depth_m: float = 6.0
    min_depth_m: float = 0.05
    obstacle_min_height_m: float = 0.12
    obstacle_max_height_m: float = 2.2
    sample_stride: int = 2
    free_logodds: float = -0.30
    occupied_logodds: float = 0.90
    min_logodds: float = -4.0
    max_logodds: float = 4.0


class DenseBEVMapper:
    """Dense RGB-D to allocentric BEV occupancy mapper.

    This mapper intentionally stays geometry-only. Semantic object memory should
    sit on top of this stable geometric base instead of being mixed into the
    first occupancy pass.
    """

    def __init__(
        self,
        origin_world_xz: WorldXZ,
        config: Optional[DenseBEVConfig] = None,
    ) -> None:
        self.config = config or DenseBEVConfig()
        self.origin_world_xz = origin_world_xz
        self.occupancy_logodds = np.zeros(self.config.grid_size, dtype=np.float32)
        self.explored = np.zeros(self.config.grid_size, dtype=bool)
        self.observation_count = np.zeros(self.config.grid_size, dtype=np.int32)
        self.trajectory: list[GridCoord] = []

    def update_from_depth(
        self,
        depth: np.ndarray,
        agent_position_xyz: Iterable[float],
        sensor_position_xyz: Iterable[float],
        sensor_rotation,
        hfov_deg: float = 90.0,
    ) -> dict:
        depth = _valid_depth(depth)
        agent_xyz = np.asarray(agent_position_xyz, dtype=np.float32).reshape(3)
        sensor_xyz = np.asarray(sensor_position_xyz, dtype=np.float32).reshape(3)

        agent_cell = self.world_to_grid((float(agent_xyz[0]), float(agent_xyz[2])))
        if agent_cell is not None:
            self.trajectory.append(agent_cell)
            self.explored[agent_cell] = True

        points_world = depth_to_world_points(
            depth=depth,
            sensor_position_xyz=sensor_xyz,
            sensor_rotation=sensor_rotation,
            hfov_deg=hfov_deg,
            stride=self.config.sample_stride,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
        )
        if points_world.size == 0:
            return self.snapshot()

        floor_y = float(agent_xyz[1])
        rel_y = points_world[:, 1] - floor_y
        obstacle_mask = np.logical_and(
            rel_y >= self.config.obstacle_min_height_m,
            rel_y <= self.config.obstacle_max_height_m,
        )

        endpoint_cells = [
            self.world_to_grid((float(point[0]), float(point[2])))
            for point in points_world
        ]
        for cell in endpoint_cells:
            if cell is not None:
                self.explored[cell] = True
                self.observation_count[cell] += 1

        if agent_cell is not None:
            for cell in endpoint_cells:
                if cell is None:
                    continue
                for free_cell in _bresenham(agent_cell, cell):
                    if free_cell == cell:
                        break
                    if self.in_bounds(free_cell):
                        self.explored[free_cell] = True
                        self.occupancy_logodds[free_cell] = max(
                            self.config.min_logodds,
                            self.occupancy_logodds[free_cell] + self.config.free_logodds,
                        )

        for cell, is_obstacle in zip(endpoint_cells, obstacle_mask):
            if cell is None:
                continue
            if is_obstacle:
                self.occupancy_logodds[cell] = min(
                    self.config.max_logodds,
                    self.occupancy_logodds[cell] + self.config.occupied_logodds,
                )
            else:
                self.occupancy_logodds[cell] = max(
                    self.config.min_logodds,
                    self.occupancy_logodds[cell] + self.config.free_logodds,
                )

        return self.snapshot()

    def occupancy_state(self) -> np.ndarray:
        state = np.full(self.config.grid_size, 0, dtype=np.int8)
        state[np.logical_and(self.explored, self.occupancy_logodds <= 0.2)] = 1
        state[self.occupancy_logodds > 0.2] = 2
        return state

    def free_mask(self) -> np.ndarray:
        return np.logical_and(self.explored, self.occupancy_logodds <= 0.2)

    def occupied_mask(self) -> np.ndarray:
        return self.occupancy_logodds > 0.2

    def confidence(self) -> np.ndarray:
        return np.clip(np.abs(self.occupancy_logodds) / self.config.max_logodds, 0.0, 1.0)

    def world_to_grid(self, coord: WorldXZ) -> Optional[GridCoord]:
        gx = int(round((coord[0] - self.origin_world_xz[0]) / self.config.resolution))
        gy = int(round((coord[1] - self.origin_world_xz[1]) / self.config.resolution))
        cell = (gx, gy)
        return cell if self.in_bounds(cell) else None

    def grid_to_world(self, cell: GridCoord) -> WorldXZ:
        return (
            self.origin_world_xz[0] + cell[0] * self.config.resolution,
            self.origin_world_xz[1] + cell[1] * self.config.resolution,
        )

    def in_bounds(self, cell: GridCoord) -> bool:
        return 0 <= cell[0] < self.config.grid_size[0] and 0 <= cell[1] < self.config.grid_size[1]

    def snapshot(self) -> dict:
        return {
            "grid_size": list(self.config.grid_size),
            "resolution": self.config.resolution,
            "origin_world_xz": list(self.origin_world_xz),
            "num_explored_cells": int(self.explored.sum()),
            "num_free_cells": int(self.free_mask().sum()),
            "num_occupied_cells": int(self.occupied_mask().sum()),
            "mean_confidence": float(self.confidence()[self.explored].mean()) if self.explored.any() else 0.0,
            "trajectory": [list(cell) for cell in self.trajectory],
        }


def depth_to_world_points(
    depth: np.ndarray,
    sensor_position_xyz: np.ndarray,
    sensor_rotation,
    hfov_deg: float,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    depth = _valid_depth(depth)
    height, width = depth.shape
    stride = max(1, int(stride))
    rows = np.arange(0, height, stride, dtype=np.int32)
    cols = np.arange(0, width, stride, dtype=np.int32)
    uu, vv = np.meshgrid(cols, rows)
    z = depth[vv, uu].astype(np.float32)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32)

    fx = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    x_cam = (uu.astype(np.float32) - cx) / fx * z
    y_cam = -(vv.astype(np.float32) - cy) / fy * z
    z_cam = -z
    camera_points = np.stack([x_cam[valid], y_cam[valid], z_cam[valid]], axis=1)

    rotated_points = _rotate_vectors(sensor_rotation, camera_points)
    return (sensor_position_xyz.reshape(1, 3) + rotated_points).astype(np.float32)


def oracle_navmesh_mask(pathfinder, origin_world_xz: WorldXZ, grid_size: Tuple[int, int], resolution: float, height: float) -> np.ndarray:
    mask = np.zeros(grid_size, dtype=bool)
    for gx in range(grid_size[0]):
        x = origin_world_xz[0] + gx * resolution
        for gy in range(grid_size[1]):
            z = origin_world_xz[1] + gy * resolution
            try:
                mask[gx, gy] = bool(pathfinder.is_navigable(np.array([x, height, z], dtype=np.float32)))
            except TypeError:
                mask[gx, gy] = bool(pathfinder.is_navigable([x, height, z]))
    return mask


def mapping_metrics(pred_free: np.ndarray, pred_occupied: np.ndarray, explored: np.ndarray, oracle_free: np.ndarray, resolution: float) -> dict:
    observed = explored.astype(bool)
    oracle_free = oracle_free.astype(bool)
    pred_free = pred_free.astype(bool)
    pred_occupied = pred_occupied.astype(bool)

    observed_oracle = np.logical_and(observed, oracle_free)
    free_intersection = np.logical_and(pred_free, oracle_free).sum()
    free_union = np.logical_or(pred_free, observed_oracle).sum()
    free_iou = _safe_div(float(free_intersection), float(free_union))
    free_precision = _safe_div(float(free_intersection), float(pred_free.sum()))
    free_recall_observed = _safe_div(float(free_intersection), float(observed_oracle.sum()))

    oracle_obstacle_observed = np.logical_and(observed, ~oracle_free)
    occupied_intersection = np.logical_and(pred_occupied, oracle_obstacle_observed).sum()
    occupied_precision = _safe_div(float(occupied_intersection), float(pred_occupied.sum()))
    occupied_recall_observed = _safe_div(float(occupied_intersection), float(oracle_obstacle_observed.sum()))

    boundary_chamfer = chamfer_distance_cells(
        boundary_cells(pred_occupied),
        boundary_cells(oracle_obstacle_observed),
        resolution,
    )

    return {
        "free_iou_observed": free_iou,
        "free_precision": free_precision,
        "free_recall_observed": free_recall_observed,
        "occupied_precision_observed": occupied_precision,
        "occupied_recall_observed": occupied_recall_observed,
        "occupied_boundary_chamfer_m": boundary_chamfer,
        "pred_free_cells": int(pred_free.sum()),
        "pred_occupied_cells": int(pred_occupied.sum()),
        "observed_cells": int(observed.sum()),
        "oracle_free_observed_cells": int(observed_oracle.sum()),
        "oracle_obstacle_observed_cells": int(oracle_obstacle_observed.sum()),
    }


def boundary_cells(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.empty((0, 2), dtype=np.int32)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = np.logical_and(mask, ~neighbors)
    return np.argwhere(boundary)


def chamfer_distance_cells(a: np.ndarray, b: np.ndarray, resolution: float, max_points: int = 2000) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    a = _subsample_points(a.astype(np.float32), max_points)
    b = _subsample_points(b.astype(np.float32), max_points)
    distances = _min_distances(a, b)
    distances_rev = _min_distances(b, a)
    return float((distances.mean() + distances_rev.mean()) * 0.5 * resolution)


def _min_distances(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    out = np.empty((a.shape[0],), dtype=np.float32)
    for start in range(0, a.shape[0], chunk):
        aa = a[start : start + chunk]
        diff = aa[:, None, :] - b[None, :, :]
        out[start : start + chunk] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return out


def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int32)
    return points[idx]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _valid_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H, W), got {depth.shape}")
    return depth


def _rotate_vectors(rotation, points: np.ndarray) -> np.ndarray:
    if hasattr(rotation, "transform_vector"):
        return np.asarray([rotation.transform_vector(point) for point in points], dtype=np.float32)

    try:
        import quaternion as np_quaternion

        matrix = np.asarray(np_quaternion.as_rotation_matrix(rotation), dtype=np.float32)
        return (points @ matrix.T).astype(np.float32)
    except Exception as exc:
        raise TypeError(f"Unsupported sensor rotation type: {type(rotation)!r}") from exc


def _bresenham(start: GridCoord, end: GridCoord):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield (x, y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
