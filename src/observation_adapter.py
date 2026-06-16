from __future__ import annotations

from math import degrees, radians, tan
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from observation_types import (
    AgentPose,
    CameraIntrinsics,
    ObservationFrame,
    ObservationRay,
    SyntheticObservation,
)


class SyntheticObservationAdapter:
    def __init__(
        self,
        scene_id: str = "synthetic_scene",
        episode_id: str = "synthetic_episode",
    ) -> None:
        self.scene_id = scene_id
        self.episode_id = episode_id

    def to_frame(self, observation: SyntheticObservation) -> ObservationFrame:
        return ObservationFrame(
            frame_id=observation.view_id,
            time=observation.time,
            pose=observation.pose,
            rays=list(observation.rays),
            scene_id=self.scene_id,
            episode_id=self.episode_id,
            metadata={"source": "synthetic"},
        )


class MockHabitatObservationAdapter:
    """Converts a Habitat-like dict into the project ObservationFrame contract."""

    def __init__(
        self,
        scene_id: str = "mock_habitat_scene",
        episode_id: str = "mock_habitat_episode",
    ) -> None:
        self.scene_id = scene_id
        self.episode_id = episode_id

    def to_frame(self, raw_observation: Dict[str, Any]) -> ObservationFrame:
        pose = _coerce_pose(raw_observation["pose"])
        intrinsics = _coerce_intrinsics(raw_observation.get("camera_intrinsics"))
        rays = _coerce_rays(
            raw_observation.get("rays", []),
            raw_observation.get("semantic_detections", []),
        )
        return ObservationFrame(
            frame_id=str(raw_observation.get("frame_id", raw_observation.get("view_id", "habitat_frame"))),
            time=int(raw_observation.get("time", 0)),
            pose=pose,
            rays=rays,
            scene_id=str(raw_observation.get("scene_id", self.scene_id)),
            episode_id=str(raw_observation.get("episode_id", self.episode_id)),
            camera_intrinsics=intrinsics,
            rgb_shape=_shape_or_none(raw_observation.get("rgb")),
            depth_shape=_shape_or_none(raw_observation.get("depth")),
            metadata={"source": "mock_habitat"},
        )


class HabitatObservationAdapter:
    """Converts RGB-D/semantic Habitat observations into ObservationFrame rays."""

    def __init__(
        self,
        scene_id: str = "habitat_scene",
        episode_id: str = "habitat_episode",
        hfov_deg: float = 90.0,
        max_rays: int = 9,
        max_depth: float = 10.0,
        semantic_id_to_label: Optional[Dict[int, str]] = None,
        ignore_semantic_ids: Iterable[int] = (-1, 0),
    ) -> None:
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.hfov_deg = hfov_deg
        self.max_rays = max_rays
        self.max_depth = max_depth
        self.semantic_id_to_label = semantic_id_to_label or {}
        self.ignore_semantic_ids = set(ignore_semantic_ids)

    def to_frame(self, raw_observation: Dict[str, Any]) -> ObservationFrame:
        pose = _coerce_habitat_pose(raw_observation)
        rgb = raw_observation.get("rgb")
        depth = _depth_array_or_none(raw_observation.get("depth"))
        semantic = _semantic_array_or_none(raw_observation.get("semantic"))
        intrinsics = _coerce_intrinsics(raw_observation.get("camera_intrinsics"))
        if intrinsics is None:
            intrinsics = _intrinsics_from_arrays(rgb, depth, self.hfov_deg)

        rays = _coerce_rays(
            raw_observation.get("rays", []),
            raw_observation.get("semantic_detections", []),
        )
        if depth is not None:
            rays.extend(
                _rays_from_depth_semantic(
                    depth=depth,
                    semantic=semantic,
                    hfov_deg=self.hfov_deg,
                    max_rays=self.max_rays,
                    max_depth=self.max_depth,
                    semantic_id_to_label=self.semantic_id_to_label,
                    ignore_semantic_ids=self.ignore_semantic_ids,
                )
            )
        if not rays:
            raise ValueError("HabitatObservationAdapter needs depth or explicit rays")

        return ObservationFrame(
            frame_id=str(raw_observation.get("frame_id", raw_observation.get("view_id", "habitat_frame"))),
            time=int(raw_observation.get("time", 0)),
            pose=pose,
            rays=rays,
            scene_id=str(raw_observation.get("scene_id", self.scene_id)),
            episode_id=str(raw_observation.get("episode_id", self.episode_id)),
            camera_intrinsics=intrinsics,
            rgb_shape=_shape_or_none(rgb),
            depth_shape=_shape_or_none(depth),
            metadata={
                "source": "habitat",
                "hfov_deg": self.hfov_deg,
                "ray_source": "depth_semantic" if depth is not None else "explicit_rays",
            },
        )


def _coerce_pose(raw_pose: Any) -> AgentPose:
    if isinstance(raw_pose, AgentPose):
        return raw_pose
    if isinstance(raw_pose, dict):
        return AgentPose(
            x=float(raw_pose["x"]),
            y=float(raw_pose["y"]),
            heading_deg=float(raw_pose.get("heading_deg", raw_pose.get("yaw_deg", 0.0))),
        )
    x, y, heading = raw_pose
    return AgentPose(x=float(x), y=float(y), heading_deg=float(heading))


def _coerce_habitat_pose(raw_observation: Dict[str, Any]) -> AgentPose:
    if "pose" in raw_observation:
        return _coerce_pose(raw_observation["pose"])
    if "gps" in raw_observation:
        gps = np.asarray(raw_observation["gps"]).reshape(-1)
        heading_deg = 0.0
        if "compass" in raw_observation:
            compass = float(np.asarray(raw_observation["compass"]).reshape(-1)[0])
            heading_deg = degrees(compass)
        return AgentPose(x=float(gps[0]), y=float(gps[-1]), heading_deg=heading_deg)
    if "agent_state" in raw_observation:
        state = raw_observation["agent_state"]
        position = getattr(state, "position", None)
        if position is None and isinstance(state, dict):
            position = state.get("position")
        if position is not None:
            values = np.asarray(position).reshape(-1)
            heading_deg = float(raw_observation.get("heading_deg", 0.0))
            return AgentPose(x=float(values[0]), y=float(values[-1]), heading_deg=heading_deg)
    raise ValueError("Habitat observation must include pose, gps/compass, or agent_state")


def _coerce_intrinsics(raw_intrinsics: Optional[Dict[str, Any]]) -> Optional[CameraIntrinsics]:
    if raw_intrinsics is None:
        return None
    if isinstance(raw_intrinsics, CameraIntrinsics):
        return raw_intrinsics
    return CameraIntrinsics(
        width=int(raw_intrinsics["width"]),
        height=int(raw_intrinsics["height"]),
        fx=float(raw_intrinsics["fx"]),
        fy=float(raw_intrinsics["fy"]),
        cx=float(raw_intrinsics["cx"]),
        cy=float(raw_intrinsics["cy"]),
    )


def _intrinsics_from_arrays(
    rgb: Any,
    depth: Optional[np.ndarray],
    hfov_deg: float,
) -> Optional[CameraIntrinsics]:
    shape = _shape_or_none(depth) or _shape_or_none(rgb)
    if shape is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    fx = width / (2.0 * tan(radians(hfov_deg) / 2.0))
    fy = fx
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
    )


def _coerce_rays(raw_rays: Iterable[Dict[str, Any]], raw_detections: Iterable[Dict[str, Any]]) -> List[ObservationRay]:
    rays = [
        ObservationRay(
            angle_deg=float(ray["angle_deg"]),
            distance=float(ray["distance"]),
            hit_type=str(ray.get("hit_type", "free")),
            semantic_label=ray.get("semantic_label"),
            semantic_confidence=float(ray.get("semantic_confidence", 0.0)),
        )
        for ray in raw_rays
    ]
    for detection in raw_detections:
        rays.append(
            ObservationRay(
                angle_deg=float(detection["angle_deg"]),
                distance=float(detection["distance"]),
                hit_type="object",
                semantic_label=str(detection["label"]),
                semantic_confidence=float(detection.get("confidence", 1.0)),
            )
        )
    return rays


def _depth_array_or_none(raw_depth: Any) -> Optional[np.ndarray]:
    if raw_depth is None:
        return None
    depth = np.asarray(raw_depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H, W) or (H, W, 1), got {depth.shape}")
    return depth


def _semantic_array_or_none(raw_semantic: Any) -> Optional[np.ndarray]:
    if raw_semantic is None:
        return None
    semantic = np.asarray(raw_semantic)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[:, :, 0]
    if semantic.ndim != 2:
        raise ValueError(f"Expected semantic shape (H, W) or (H, W, 1), got {semantic.shape}")
    return semantic


def _rays_from_depth_semantic(
    depth: np.ndarray,
    semantic: Optional[np.ndarray],
    hfov_deg: float,
    max_rays: int,
    max_depth: float,
    semantic_id_to_label: Dict[int, str],
    ignore_semantic_ids: set[int],
) -> List[ObservationRay]:
    height, width = depth.shape
    ray_count = max(1, min(max_rays, width))
    columns = np.linspace(0, width - 1, ray_count, dtype=np.int32)
    rays: List[ObservationRay] = []
    for column in columns:
        column_depth = depth[:, int(column)]
        valid_depth = column_depth[np.isfinite(column_depth) & (column_depth > 0)]
        if valid_depth.size == 0:
            continue
        distance = min(float(np.median(valid_depth)), max_depth)
        angle_deg = _column_to_angle_deg(int(column), width, hfov_deg)
        label = None
        if semantic is not None:
            label = _semantic_label_for_column(
                semantic[:, int(column)],
                semantic_id_to_label,
                ignore_semantic_ids,
            )
        rays.append(
            ObservationRay(
                angle_deg=angle_deg,
                distance=distance,
                hit_type="object" if label else "obstacle",
                semantic_label=label,
                semantic_confidence=1.0 if label else 0.0,
            )
        )
    return rays


def _column_to_angle_deg(column: int, width: int, hfov_deg: float) -> float:
    if width <= 1:
        return 0.0
    return (column / (width - 1) - 0.5) * hfov_deg


def _semantic_label_for_column(
    values: np.ndarray,
    semantic_id_to_label: Dict[int, str],
    ignore_semantic_ids: set[int],
) -> Optional[str]:
    valid = values[np.isfinite(values)].astype(np.int64)
    valid = valid[~np.isin(valid, list(ignore_semantic_ids))]
    if valid.size == 0:
        return None
    ids, counts = np.unique(valid, return_counts=True)
    semantic_id = int(ids[int(np.argmax(counts))])
    return semantic_id_to_label.get(semantic_id, f"semantic_{semantic_id}")


def _shape_or_none(array_like: Any) -> Optional[tuple[int, ...]]:
    shape = getattr(array_like, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)
