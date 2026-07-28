from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from math import exp
from typing import Iterable

import numpy as np


GridCoord = tuple[int, int]


@dataclass(frozen=True)
class InterestConfig:
    gain_radius_cells: int = 12
    min_unknown_gain: float = 0.08
    information_weight: float = 2.4
    distance_weight: float = 0.35
    revisit_weight: float = 0.55
    cluster_size_weight: float = 0.18
    obstacle_risk_weight: float = 0.35
    candidate_stride: int = 3
    max_candidates: int = 800
    min_frontier_distance_m: float = 0.8
    min_frontier_cluster_cells: int = 4
    viewpoints_per_cluster: int = 5
    ray_count: int = 61
    sensor_hfov_deg: float = 90.0
    sensor_range_m: float = 6.0


def frontier_mask(explored: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Free cells adjacent to at least one currently unknown cell."""
    explored = np.asarray(explored, dtype=bool)
    free = np.asarray(free, dtype=bool)
    if explored.shape != free.shape:
        raise ValueError("explored and free must have the same shape")
    unknown = ~explored
    adjacent_unknown = np.zeros_like(unknown)
    adjacent_unknown[1:, :] |= unknown[:-1, :]
    adjacent_unknown[:-1, :] |= unknown[1:, :]
    adjacent_unknown[:, 1:] |= unknown[:, :-1]
    adjacent_unknown[:, :-1] |= unknown[:, 1:]
    return free & adjacent_unknown


def rank_frontiers(
    explored: np.ndarray,
    free: np.ndarray,
    observation_count: np.ndarray,
    current_cell: GridCoord,
    resolution: float,
    config: InterestConfig | None = None,
) -> list[dict]:
    """Rank frontier cells using unknown-space gain, travel cost, and revisit cost."""
    config = config or InterestConfig()
    explored = np.asarray(explored, dtype=bool)
    free = np.asarray(free, dtype=bool)
    observation_count = np.asarray(observation_count, dtype=np.float32)
    if explored.shape != free.shape or explored.shape != observation_count.shape:
        raise ValueError("map tensors must have the same shape")

    reachable = reachable_free_mask(free, current_cell)
    cells = np.argwhere(frontier_mask(explored, free) & reachable)
    stride = max(1, int(config.candidate_stride))
    cells = cells[::stride][: max(1, int(config.max_candidates))]
    if cells.size == 0:
        return []

    unknown = (~explored).astype(np.int32)
    integral = np.pad(unknown, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    radius = max(1, int(config.gain_radius_cells))
    area = float((2 * radius + 1) ** 2)
    current = np.asarray(current_cell, dtype=np.float32)
    ranked: list[dict] = []
    for raw_cell in cells:
        gx, gy = int(raw_cell[0]), int(raw_cell[1])
        x0, x1 = max(0, gx - radius), min(explored.shape[0], gx + radius + 1)
        y0, y1 = max(0, gy - radius), min(explored.shape[1], gy + radius + 1)
        unknown_cells = _box_sum(integral, x0, x1, y0, y1)
        unknown_gain = float(unknown_cells) / area
        if unknown_gain < float(config.min_unknown_gain):
            continue
        distance_m = float(np.linalg.norm(raw_cell.astype(np.float32) - current) * resolution)
        if distance_m < float(config.min_frontier_distance_m):
            continue
        revisit = float(observation_count[gx, gy])
        score = (
            float(config.information_weight) * unknown_gain
            - float(config.distance_weight) * distance_m
            - float(config.revisit_weight) * np.log1p(revisit)
        )
        ranked.append(
            {
                "cell": (gx, gy),
                "score": score,
                "unknown_gain": unknown_gain,
                "distance_m": distance_m,
                "revisit_count": revisit,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def cluster_frontiers(mask: np.ndarray, min_cells: int = 4) -> list[np.ndarray]:
    """Group adjacent frontier cells into stable exploration regions."""
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask)
    clusters: list[np.ndarray] = []
    for raw_seed in np.argwhere(mask):
        seed = int(raw_seed[0]), int(raw_seed[1])
        if visited[seed]:
            continue
        queue: deque[GridCoord] = deque([seed])
        visited[seed] = True
        cells: list[GridCoord] = []
        while queue:
            cell = queue.popleft()
            cells.append(cell)
            for neighbor, _ in _neighbors(cell, mask.shape):
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if len(cells) >= max(1, int(min_cells)):
            clusters.append(np.asarray(cells, dtype=np.int32))
    clusters.sort(key=len, reverse=True)
    return clusters


def visible_unknown_gain(
    explored: np.ndarray,
    occupied: np.ndarray,
    viewpoint: GridCoord,
    yaw_rad: float,
    resolution: float,
    hfov_deg: float = 90.0,
    max_range_m: float = 6.0,
    ray_count: int = 61,
) -> tuple[int, int]:
    """Ray-cast a horizontal sensor FOV and count newly visible unknown cells."""
    explored = np.asarray(explored, dtype=bool)
    occupied = np.asarray(occupied, dtype=bool)
    if explored.shape != occupied.shape:
        raise ValueError("explored and occupied must have the same shape")
    max_cells = max(1, int(round(float(max_range_m) / max(float(resolution), 1e-6))))
    angles = np.linspace(
        float(yaw_rad) - np.deg2rad(float(hfov_deg)) / 2.0,
        float(yaw_rad) + np.deg2rad(float(hfov_deg)) / 2.0,
        max(3, int(ray_count)),
    )
    visible: set[GridCoord] = set()
    unknown: set[GridCoord] = set()
    origin = np.asarray(viewpoint, dtype=np.float32)
    for angle in angles:
        direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        previous: GridCoord | None = None
        for distance in range(1, max_cells + 1):
            raw = np.rint(origin + direction * distance).astype(np.int32)
            cell = int(raw[0]), int(raw[1])
            if cell == previous:
                continue
            previous = cell
            if not (0 <= cell[0] < explored.shape[0] and 0 <= cell[1] < explored.shape[1]):
                break
            visible.add(cell)
            if not explored[cell]:
                unknown.add(cell)
            if occupied[cell]:
                break
    return len(unknown), len(visible)


def rank_frontier_clusters(
    explored: np.ndarray,
    free: np.ndarray,
    occupied: np.ndarray,
    observation_count: np.ndarray,
    current_cell: GridCoord,
    resolution: float,
    config: InterestConfig | None = None,
) -> list[dict]:
    """Rank frontier regions by sensor-visible gain and executable path cost."""
    config = config or InterestConfig()
    explored = np.asarray(explored, dtype=bool)
    free = np.asarray(free, dtype=bool)
    occupied = np.asarray(occupied, dtype=bool)
    observation_count = np.asarray(observation_count, dtype=np.float32)
    if not (
        explored.shape
        == free.shape
        == occupied.shape
        == observation_count.shape
    ):
        raise ValueError("map tensors must have the same shape")

    distance_map = _free_distance_map(free, current_cell)
    reachable = np.isfinite(distance_map)
    clusters = cluster_frontiers(
        frontier_mask(explored, free) & reachable,
        min_cells=config.min_frontier_cluster_cells,
    )
    ranked: list[dict] = []
    for cluster_id, cells in enumerate(clusters):
        center = cells.astype(np.float32).mean(axis=0)
        unknown_neighbors: list[GridCoord] = []
        for raw_cell in cells:
            cell = int(raw_cell[0]), int(raw_cell[1])
            for neighbor, _ in _neighbors(cell, explored.shape):
                if not explored[neighbor]:
                    unknown_neighbors.append(neighbor)
        if unknown_neighbors:
            unknown_center = np.asarray(unknown_neighbors, dtype=np.float32).mean(axis=0)
        else:
            unknown_center = center
        cluster_size_gain = float(np.log1p(len(cells)))
        candidate_indices = {
            int(np.argmin(np.linalg.norm(cells.astype(np.float32) - center, axis=1))),
            int(
                np.argmin(
                    np.linalg.norm(
                        cells.astype(np.float32)
                        - np.asarray(current_cell, dtype=np.float32),
                        axis=1,
                    )
                )
            ),
        }
        count = min(max(1, int(config.viewpoints_per_cluster)), len(cells))
        candidate_indices.update(
            int(value)
            for value in np.linspace(0, len(cells) - 1, count, dtype=np.int32)
        )
        best: dict | None = None
        for candidate_index in sorted(candidate_indices):
            representative = cells[candidate_index]
            goal = int(representative[0]), int(representative[1])
            path_distance_cells = float(distance_map[goal])
            if not np.isfinite(path_distance_cells):
                continue
            path_distance_m = path_distance_cells * float(resolution)
            if path_distance_m < float(config.min_frontier_distance_m):
                continue
            direction = unknown_center - representative.astype(np.float32)
            yaw_rad = float(np.arctan2(direction[1], direction[0]))
            unknown_cells, visible_cells = visible_unknown_gain(
                explored,
                occupied,
                goal,
                yaw_rad=yaw_rad,
                resolution=resolution,
                hfov_deg=config.sensor_hfov_deg,
                max_range_m=config.sensor_range_m,
                ray_count=config.ray_count,
            )
            unknown_gain = float(unknown_cells) / max(1.0, float(visible_cells))
            if unknown_gain < float(config.min_unknown_gain):
                continue

            radius = max(1, int(config.gain_radius_cells // 2))
            x0 = max(0, goal[0] - radius)
            x1 = min(explored.shape[0], goal[0] + radius + 1)
            y0 = max(0, goal[1] - radius)
            y1 = min(explored.shape[1], goal[1] + radius + 1)
            revisit = float(np.mean(observation_count[x0:x1, y0:y1]))
            occupied_nearby = occupied[x0:x1, y0:y1]
            obstacle_risk = (
                float(np.mean(occupied_nearby)) if occupied_nearby.size else 0.0
            )
            score = (
                float(config.information_weight) * unknown_gain
                + float(config.cluster_size_weight) * cluster_size_gain
                - float(config.distance_weight) * path_distance_m
                - float(config.revisit_weight) * np.log1p(revisit)
                - float(config.obstacle_risk_weight) * obstacle_risk
            )
            item = {
                "cell": goal,
                "score": score,
                "unknown_gain": unknown_gain,
                "visible_unknown_cells": int(unknown_cells),
                "visible_cells": int(visible_cells),
                "distance_m": path_distance_m,
                "path_distance_m": path_distance_m,
                "revisit_count": revisit,
                "obstacle_risk": obstacle_risk,
                "cluster_id": cluster_id,
                "cluster_size": int(len(cells)),
                "view_yaw_rad": yaw_rad,
                "view_yaw_deg": float(
                    np.degrees(np.arctan2(-direction[0], -direction[1]))
                ),
                "sampled_viewpoints": len(candidate_indices),
            }
            if best is None or float(item["score"]) > float(best["score"]):
                best = item
        if best is not None:
            ranked.append(best)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def planning_free_mask(
    free: np.ndarray,
    occupied: np.ndarray,
    inflation_radius_cells: int = 3,
) -> np.ndarray:
    """Return observed free space with a robot-sized margin around obstacles."""
    free = np.asarray(free, dtype=bool)
    occupied = np.asarray(occupied, dtype=bool)
    if free.shape != occupied.shape:
        raise ValueError("free and occupied must have the same shape")
    radius = max(0, int(inflation_radius_cells))
    if radius == 0 or not occupied.any():
        return free & ~occupied
    inflated = occupied.copy()
    offsets = [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]
    for dx, dy in offsets:
        src_x0 = max(0, -dx)
        src_x1 = occupied.shape[0] - max(0, dx)
        src_y0 = max(0, -dy)
        src_y1 = occupied.shape[1] - max(0, dy)
        dst_x0 = max(0, dx)
        dst_x1 = occupied.shape[0] - max(0, -dx)
        dst_y0 = max(0, dy)
        dst_y1 = occupied.shape[1] - max(0, -dy)
        inflated[dst_x0:dst_x1, dst_y0:dst_y1] |= occupied[src_x0:src_x1, src_y0:src_y1]
    return free & ~inflated


def reachable_free_mask(free: np.ndarray, start: GridCoord) -> np.ndarray:
    """Flood-fill the observed free component containing the robot."""
    free = np.asarray(free, dtype=bool)
    reachable = np.zeros_like(free)
    seed = nearest_free_cell(free, start, max_radius=8)
    if seed is None:
        return reachable
    queue: deque[GridCoord] = deque([seed])
    reachable[seed] = True
    while queue:
        cell = queue.popleft()
        for neighbor, _ in _neighbors(cell, free.shape):
            if free[neighbor] and not reachable[neighbor]:
                reachable[neighbor] = True
                queue.append(neighbor)
    return reachable


def nearest_free_cell(
    free: np.ndarray,
    cell: GridCoord,
    max_radius: int = 8,
) -> GridCoord | None:
    free = np.asarray(free, dtype=bool)
    x, y = int(cell[0]), int(cell[1])
    if 0 <= x < free.shape[0] and 0 <= y < free.shape[1] and free[x, y]:
        return x, y
    for radius in range(1, max(1, int(max_radius)) + 1):
        candidates = []
        for dx in range(-radius, radius + 1):
            candidates.extend(((x + dx, y - radius), (x + dx, y + radius)))
        for dy in range(-radius + 1, radius):
            candidates.extend(((x - radius, y + dy), (x + radius, y + dy)))
        for candidate in candidates:
            if (
                0 <= candidate[0] < free.shape[0]
                and 0 <= candidate[1] < free.shape[1]
                and free[candidate]
            ):
                return candidate
    return None


def plan_observed_path(
    free: np.ndarray,
    start: GridCoord,
    goal: GridCoord,
    max_expansions: int = 120000,
) -> list[GridCoord]:
    """A* over observed free cells. Returns an empty list when unreachable."""
    free = np.asarray(free, dtype=bool)
    start_cell = nearest_free_cell(free, start, max_radius=8)
    goal_cell = nearest_free_cell(free, goal, max_radius=8)
    if start_cell is None or goal_cell is None:
        return []
    if start_cell == goal_cell:
        return [start_cell]

    candidates: list[tuple[float, float, GridCoord]] = [(0.0, 0.0, start_cell)]
    came_from: dict[GridCoord, GridCoord] = {}
    g_score: dict[GridCoord, float] = {start_cell: 0.0}
    closed: set[GridCoord] = set()
    expansions = 0
    while candidates and expansions < int(max_expansions):
        _, current_cost, current = heapq.heappop(candidates)
        if current in closed:
            continue
        if current == goal_cell:
            return _reconstruct_path(came_from, current)
        closed.add(current)
        expansions += 1
        for neighbor, move_cost in _neighbors(current, free.shape):
            if not free[neighbor] or neighbor in closed:
                continue
            dx = neighbor[0] - current[0]
            dy = neighbor[1] - current[1]
            if dx != 0 and dy != 0:
                if not free[current[0] + dx, current[1]] or not free[current[0], current[1] + dy]:
                    continue
            tentative = current_cost + move_cost
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heuristic = float(np.hypot(neighbor[0] - goal_cell[0], neighbor[1] - goal_cell[1]))
            heapq.heappush(candidates, (tentative + heuristic, tentative, neighbor))
    return []


def approach_cell_for_target(
    reachable_free: np.ndarray,
    target: GridCoord,
    desired_radius_cells: int,
    current: GridCoord | None = None,
) -> GridCoord | None:
    """Choose a reachable observation cell around a semantic target."""
    cells = np.argwhere(np.asarray(reachable_free, dtype=bool))
    if cells.size == 0:
        return None
    target_array = np.asarray(target, dtype=np.float32)
    distances = np.linalg.norm(cells.astype(np.float32) - target_array, axis=1)
    desired = max(1.0, float(desired_radius_cells))
    score = np.abs(distances - desired) + 0.12 * distances
    if current is not None:
        score += 0.08 * np.linalg.norm(
            cells.astype(np.float32) - np.asarray(current, dtype=np.float32),
            axis=1,
        )
    best = cells[int(np.argmin(score))]
    return int(best[0]), int(best[1])


def semantic_interest_score(
    label: str,
    confidence: float,
    views: int,
    distance_m: float,
    already_scanned: bool,
) -> float:
    """Task-neutral novelty score for inspectable indoor support surfaces."""
    if already_scanned:
        return -1e6
    priors = {"table": 1.0, "counter": 0.95, "sink": 0.85}
    prior = priors.get(str(label).lower(), 0.0)
    if prior <= 0.0:
        return -1e6
    stability = min(1.0, max(0.0, float(confidence))) * min(1.0, max(0, int(views)) / 5.0)
    return 1.8 * prior + 1.4 * stability - 0.18 * max(0.0, float(distance_m))


def smooth_motion_evidence_weight(
    translation_m: float,
    rotation_deg: float,
    nominal_translation_m: float = 0.25,
    nominal_rotation_deg: float = 15.0,
) -> float:
    """Smoothly weight evidence without rewarding stationary or extreme motion."""
    translation_units = max(0.0, float(translation_m)) / max(1e-6, float(nominal_translation_m))
    rotation_units = max(0.0, float(rotation_deg)) / max(1e-6, float(nominal_rotation_deg))
    motion = float(np.hypot(translation_units, 0.55 * rotation_units))
    rise = 1.0 - exp(-1.25 * motion)
    overspeed = max(0.0, motion - 2.25)
    damping = exp(-(overspeed * overspeed) / 6.0)
    return float(np.clip(0.08 + 0.92 * rise * damping, 0.08, 1.0))


def deep_familiarization_status(
    step: int,
    explored_cell_history: Iterable[int],
    free: np.ndarray,
    observation_count: np.ndarray,
    min_steps: int,
    max_steps: int,
    saturation_window: int,
    max_new_cells: int,
    min_reobserve_ratio: float,
) -> tuple[bool, dict]:
    """Evaluate a causal map-saturation and repeat-observation stopping rule."""
    history = [int(value) for value in explored_cell_history]
    free = np.asarray(free, dtype=bool)
    observation_count = np.asarray(observation_count)
    if free.shape != observation_count.shape:
        raise ValueError("free and observation_count must have the same shape")
    free_cells = int(free.sum())
    reobserved_cells = int(np.count_nonzero(free & (observation_count >= 2)))
    reobserve_ratio = reobserved_cells / max(1, free_cells)
    window = max(1, int(saturation_window))
    window_ready = len(history) > window
    window_gain = history[-1] - history[-1 - window] if window_ready else None
    saturated = bool(
        window_ready
        and window_gain is not None
        and window_gain <= int(max_new_cells)
        and reobserve_ratio >= float(min_reobserve_ratio)
    )
    max_budget_reached = int(step) >= int(max_steps)
    ready = bool(
        int(step) >= int(min_steps)
        and (saturated or max_budget_reached)
    )
    reason = (
        "max_familiarization_steps"
        if ready and max_budget_reached and not saturated
        else "saturated_and_reobserved"
        if ready
        else None
    )
    return ready, {
        "step": int(step),
        "free_cells": free_cells,
        "reobserved_cells": reobserved_cells,
        "reobserve_ratio": float(reobserve_ratio),
        "saturation_window": window,
        "window_gain_cells": int(window_gain) if window_gain is not None else None,
        "max_new_cells": int(max_new_cells),
        "min_reobserve_ratio": float(min_reobserve_ratio),
        "min_steps": int(min_steps),
        "max_steps": int(max_steps),
        "ready": ready,
        "reason": reason,
    }


def choose_semantic_target(
    tracks: Iterable[dict],
    current_xz: tuple[float, float],
    scanned_ids: set[int],
    min_views: int = 3,
    min_confidence: float = 0.24,
) -> dict | None:
    ranked = []
    current = np.asarray(current_xz, dtype=np.float32)
    for track in tracks:
        if int(track.get("views", 0)) < int(min_views):
            continue
        if float(track.get("confidence", 0.0)) < float(min_confidence):
            continue
        position = np.asarray(track.get("position_3d", []), dtype=np.float32)
        if position.shape != (3,) or not np.isfinite(position).all():
            continue
        track_id = int(track["track_id"])
        distance_m = float(np.linalg.norm(position[[0, 2]] - current))
        score = semantic_interest_score(
            label=str(track.get("label", "")),
            confidence=float(track.get("confidence", 0.0)),
            views=int(track.get("views", 0)),
            distance_m=distance_m,
            already_scanned=track_id in scanned_ids,
        )
        if score <= -1e5:
            continue
        ranked.append((score, distance_m, track))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1], int(item[2]["track_id"])))
    return ranked[0][2]


def _box_sum(integral: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> int:
    return int(integral[x1, y1] - integral[x0, y1] - integral[x1, y0] + integral[x0, y0])


def _neighbors(cell: GridCoord, shape: tuple[int, int]) -> Iterable[tuple[GridCoord, float]]:
    x, y = cell
    for dx, dy, cost in (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.4142),
        (-1, 1, 1.4142),
        (1, -1, 1.4142),
        (1, 1, 1.4142),
    ):
        neighbor = x + dx, y + dy
        if 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]:
            yield neighbor, cost


def _reconstruct_path(came_from: dict[GridCoord, GridCoord], current: GridCoord) -> list[GridCoord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _path_length(path: list[GridCoord]) -> float:
    return float(
        sum(
            np.hypot(float(end[0] - start[0]), float(end[1] - start[1]))
            for start, end in zip(path[:-1], path[1:])
        )
    )


def _free_distance_map(free: np.ndarray, start: GridCoord) -> np.ndarray:
    """Compute a fast 4-connected geodesic distance field with one BFS pass."""
    free = np.asarray(free, dtype=bool)
    distance = np.full(free.shape, -1, dtype=np.int32)
    seed = nearest_free_cell(free, start, max_radius=8)
    if seed is None:
        return distance.astype(np.float32)
    distance[seed] = 0
    queue: deque[GridCoord] = deque([seed])
    while queue:
        current = queue.popleft()
        next_distance = int(distance[current]) + 1
        for neighbor in (
            (current[0] - 1, current[1]),
            (current[0] + 1, current[1]),
            (current[0], current[1] - 1),
            (current[0], current[1] + 1),
        ):
            if not (
                0 <= neighbor[0] < free.shape[0]
                and 0 <= neighbor[1] < free.shape[1]
            ):
                continue
            if not free[neighbor] or distance[neighbor] >= 0:
                continue
            distance[neighbor] = next_distance
            queue.append(neighbor)
    result = distance.astype(np.float32)
    result[result < 0] = np.inf
    return result
