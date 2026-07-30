from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticTaskProfile:
    name: str
    task_text: str
    target_label: str
    target_aliases: tuple[str, ...]
    detector_labels: tuple[str, ...]
    support_labels: tuple[str, ...]
    verifier_labels: tuple[str, ...]
    verifier_positive_labels: tuple[str, ...]
    target_min_views: int
    target_min_confidence: float
    confirmation_min_task_views: int
    confirmation_min_visual_passes: int
    confirmation_max_attempts: int
    confirmation_min_depth_relief_m: float
    confirmation_min_depth_relief_passes: int
    confirmation_max_position_spread_m: float
    track_merge_radius_m: float
    dynamic_target_merge_radius_m: float

    def canonical_label(self, label: str) -> str:
        normalized = normalize_label(label)
        aliases = {normalize_label(value) for value in self.target_aliases}
        return self.target_label if normalized in aliases else normalized


DOOR_TASK_PROFILE = SemanticTaskProfile(
    name="door",
    task_text="Find and report all doors in the room.",
    target_label="door",
    target_aliases=("door", "doorway", "open door", "closed door"),
    detector_labels=(
        "door",
        "doorway",
        "open door",
        "closed door",
        "window",
        "cabinet door",
        "refrigerator",
        "table",
        "chair",
        "sofa",
        "bed",
        "sink",
        "counter",
        "toilet",
    ),
    support_labels=(),
    verifier_labels=(
        "door",
        "doorway",
        "open door",
        "closed door",
        "window",
        "cabinet door",
        "refrigerator door",
        "wall panel",
        "mirror",
    ),
    verifier_positive_labels=("door", "doorway", "open door", "closed door"),
    target_min_views=3,
    target_min_confidence=0.28,
    confirmation_min_task_views=2,
    confirmation_min_visual_passes=2,
    confirmation_max_attempts=3,
    confirmation_min_depth_relief_m=0.0,
    confirmation_min_depth_relief_passes=0,
    confirmation_max_position_spread_m=0.75,
    track_merge_radius_m=0.55,
    dynamic_target_merge_radius_m=0.85,
)


CUP_TASK_PROFILE = SemanticTaskProfile(
    name="cup",
    task_text="Find and report all cups in the room.",
    target_label="cup",
    target_aliases=(
        "cup",
        "mug",
        "glass",
        "drinking glass",
        "wine glass",
    ),
    detector_labels=("cup", "mug", "bottle", "table", "counter", "sink"),
    support_labels=("table", "counter", "sink"),
    verifier_labels=(
        "real drinking cup",
        "cup",
        "mug",
        "drinking glass",
        "printed picture",
        "poster",
        "wall outlet",
        "light switch",
        "wall decoration",
        "cabinet handle",
        "bottle",
    ),
    verifier_positive_labels=(
        "real drinking cup",
        "cup",
        "mug",
        "drinking glass",
    ),
    target_min_views=5,
    target_min_confidence=0.28,
    confirmation_min_task_views=2,
    confirmation_min_visual_passes=2,
    confirmation_max_attempts=3,
    confirmation_min_depth_relief_m=0.025,
    confirmation_min_depth_relief_passes=2,
    confirmation_max_position_spread_m=0.30,
    track_merge_radius_m=0.30,
    dynamic_target_merge_radius_m=0.75,
)


TASK_PROFILES = {
    DOOR_TASK_PROFILE.name: DOOR_TASK_PROFILE,
    CUP_TASK_PROFILE.name: CUP_TASK_PROFILE,
}


def get_task_profile(name: str) -> SemanticTaskProfile:
    normalized = normalize_label(name)
    try:
        return TASK_PROFILES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_PROFILES))
        raise ValueError(
            f"Unknown semantic task profile {name!r}; expected one of: {choices}"
        ) from exc


def normalize_label(label: str) -> str:
    return " ".join(
        str(label).strip().lower().replace("-", " ").replace("_", " ").split()
    )
