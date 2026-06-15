from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, radians, sin
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


GridCoord = Tuple[int, int]
WorldCoord = Tuple[float, float]


@dataclass
class AgentPose:
    x: float
    y: float
    heading_deg: float


@dataclass
class SyntheticRay:
    angle_deg: float
    distance: float
    hit_type: str = "free"  # free, obstacle, object
    semantic_label: Optional[str] = None
    semantic_confidence: float = 0.0


@dataclass
class SyntheticObservation:
    view_id: str
    time: int
    pose: AgentPose
    rays: List[SyntheticRay]


@dataclass
class ProjectedSemanticEvidence:
    label: str
    grid_coord: GridCoord
    confidence: float
    time: int
    source_view_id: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["grid_coord"] = list(self.grid_coord)
        return data


class BEVMemory:
    """Minimal allocentric BEV memory for Phase 2 synthetic projection tests."""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (16, 16),
        resolution: float = 1.0,
        origin_world_xy: WorldCoord = (0.0, 0.0),
    ) -> None:
        self.grid_size = grid_size
        self.resolution = resolution
        self.origin_world_xy = origin_world_xy
        self.occupancy_logodds = np.zeros(grid_size, dtype=np.float32)
        self.explored = np.zeros(grid_size, dtype=bool)
        self.semantic_label = np.full(grid_size, "", dtype=object)
        self.semantic_confidence = np.zeros(grid_size, dtype=np.float32)
        self.last_seen_time = np.full(grid_size, -1, dtype=np.int32)
        self.negative_evidence = np.zeros(grid_size, dtype=np.int32)
        self.trajectory: List[GridCoord] = []
        self.projected_semantic_evidence: List[ProjectedSemanticEvidence] = []

    def update_from_observation(self, observation: SyntheticObservation) -> List[ProjectedSemanticEvidence]:
        agent_cell = self.world_to_grid((observation.pose.x, observation.pose.y))
        if agent_cell is not None:
            self.trajectory.append(agent_cell)
            self.explored[agent_cell] = True

        new_semantic: List[ProjectedSemanticEvidence] = []
        for ray in observation.rays:
            endpoint = self._ray_endpoint(observation.pose, ray)
            endpoint_cell = self.world_to_grid(endpoint)
            if endpoint_cell is None or agent_cell is None:
                continue

            traversed = list(_bresenham(agent_cell, endpoint_cell))
            free_cells = traversed[:-1] if ray.hit_type in {"obstacle", "object"} else traversed
            for cell in free_cells:
                if self.in_bounds(cell):
                    self.explored[cell] = True
                    self.occupancy_logodds[cell] = max(-4.0, self.occupancy_logodds[cell] - 0.45)

            if self.in_bounds(endpoint_cell):
                self.explored[endpoint_cell] = True
                if ray.hit_type in {"obstacle", "object"}:
                    self.occupancy_logodds[endpoint_cell] = min(
                        4.0, self.occupancy_logodds[endpoint_cell] + 0.85
                    )

                if ray.hit_type == "object" and ray.semantic_label:
                    evidence = ProjectedSemanticEvidence(
                        label=ray.semantic_label,
                        grid_coord=endpoint_cell,
                        confidence=ray.semantic_confidence,
                        time=observation.time,
                        source_view_id=observation.view_id,
                    )
                    self._update_semantic_cell(evidence)
                    new_semantic.append(evidence)
                    self.projected_semantic_evidence.append(evidence)

        return new_semantic

    def world_to_grid(self, coord: WorldCoord) -> Optional[GridCoord]:
        gx = int(round((coord[0] - self.origin_world_xy[0]) / self.resolution))
        gy = int(round((coord[1] - self.origin_world_xy[1]) / self.resolution))
        cell = (gx, gy)
        return cell if self.in_bounds(cell) else None

    def grid_to_world(self, cell: GridCoord) -> WorldCoord:
        return (
            self.origin_world_xy[0] + cell[0] * self.resolution,
            self.origin_world_xy[1] + cell[1] * self.resolution,
        )

    def in_bounds(self, cell: GridCoord) -> bool:
        return 0 <= cell[0] < self.grid_size[0] and 0 <= cell[1] < self.grid_size[1]

    def occupancy_state(self) -> np.ndarray:
        state = np.full(self.grid_size, 0, dtype=np.int8)  # 0 unknown, 1 free, 2 occupied
        state[np.logical_and(self.explored, self.occupancy_logodds <= 0.2)] = 1
        state[self.occupancy_logodds > 0.2] = 2
        return state

    def snapshot(self) -> dict:
        return {
            "grid_size": list(self.grid_size),
            "resolution": self.resolution,
            "origin_world_xy": list(self.origin_world_xy),
            "trajectory": [list(cell) for cell in self.trajectory],
            "semantic_evidence": [evidence.to_dict() for evidence in self.projected_semantic_evidence],
            "num_explored_cells": int(self.explored.sum()),
            "num_occupied_cells": int((self.occupancy_logodds > 0.2).sum()),
            "num_semantic_cells": int((self.semantic_confidence > 0).sum()),
        }

    def _ray_endpoint(self, pose: AgentPose, ray: SyntheticRay) -> WorldCoord:
        theta = radians(pose.heading_deg + ray.angle_deg)
        return (pose.x + ray.distance * cos(theta), pose.y + ray.distance * sin(theta))

    def _update_semantic_cell(self, evidence: ProjectedSemanticEvidence) -> None:
        cell = evidence.grid_coord
        old_conf = float(self.semantic_confidence[cell])
        if evidence.confidence >= old_conf:
            self.semantic_label[cell] = evidence.label
            self.semantic_confidence[cell] = min(1.0, 0.35 * old_conf + 0.75 * evidence.confidence)
        else:
            self.semantic_confidence[cell] = min(1.0, old_conf + 0.10 * evidence.confidence)
        self.last_seen_time[cell] = evidence.time


def make_phase02_synthetic_observations() -> List[SyntheticObservation]:
    return [
        SyntheticObservation(
            view_id="bev_view_001",
            time=1,
            pose=AgentPose(x=2.0, y=2.0, heading_deg=0.0),
            rays=[
                SyntheticRay(angle_deg=-35, distance=4.0, hit_type="obstacle"),
                SyntheticRay(angle_deg=0, distance=5.0, hit_type="object", semantic_label="sofa", semantic_confidence=0.88),
                SyntheticRay(angle_deg=30, distance=4.5, hit_type="free"),
            ],
        ),
        SyntheticObservation(
            view_id="bev_view_002",
            time=2,
            pose=AgentPose(x=5.0, y=3.0, heading_deg=45.0),
            rays=[
                SyntheticRay(angle_deg=-30, distance=3.0, hit_type="object", semantic_label="table", semantic_confidence=0.82),
                SyntheticRay(angle_deg=10, distance=4.5, hit_type="obstacle"),
                SyntheticRay(angle_deg=45, distance=3.5, hit_type="free"),
            ],
        ),
        SyntheticObservation(
            view_id="bev_view_003",
            time=3,
            pose=AgentPose(x=7.0, y=6.0, heading_deg=70.0),
            rays=[
                SyntheticRay(angle_deg=-20, distance=3.0, hit_type="object", semantic_label="lamp", semantic_confidence=0.76),
                SyntheticRay(angle_deg=15, distance=3.5, hit_type="free"),
                SyntheticRay(angle_deg=40, distance=4.0, hit_type="obstacle"),
            ],
        ),
        SyntheticObservation(
            view_id="bev_view_004",
            time=4,
            pose=AgentPose(x=9.0, y=8.0, heading_deg=180.0),
            rays=[
                SyntheticRay(angle_deg=-15, distance=2.0, hit_type="object", semantic_label="sofa", semantic_confidence=0.94),
                SyntheticRay(angle_deg=25, distance=3.0, hit_type="obstacle"),
                SyntheticRay(angle_deg=55, distance=4.0, hit_type="free"),
            ],
        ),
    ]


def _bresenham(start: GridCoord, end: GridCoord) -> Iterable[GridCoord]:
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
