from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


WorldCoord = Tuple[float, float]


@dataclass
class AgentPose:
    x: float
    y: float
    heading_deg: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ObservationRay:
    angle_deg: float
    distance: float
    hit_type: str = "free"  # free, obstacle, object
    semantic_label: Optional[str] = None
    semantic_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ObservationFrame:
    frame_id: str
    time: int
    pose: AgentPose
    rays: List[ObservationRay]
    scene_id: str
    episode_id: str
    camera_intrinsics: Optional[CameraIntrinsics] = None
    rgb_shape: Optional[Tuple[int, ...]] = None
    depth_shape: Optional[Tuple[int, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "time": self.time,
            "pose": self.pose.to_dict(),
            "rays": [ray.to_dict() for ray in self.rays],
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "camera_intrinsics": (
                self.camera_intrinsics.to_dict() if self.camera_intrinsics else None
            ),
            "rgb_shape": list(self.rgb_shape) if self.rgb_shape else None,
            "depth_shape": list(self.depth_shape) if self.depth_shape else None,
            "metadata": dict(self.metadata),
        }


@dataclass
class SyntheticObservation:
    view_id: str
    time: int
    pose: AgentPose
    rays: List[ObservationRay]

    def to_dict(self) -> dict:
        return {
            "view_id": self.view_id,
            "time": self.time,
            "pose": self.pose.to_dict(),
            "rays": [ray.to_dict() for ray in self.rays],
        }
