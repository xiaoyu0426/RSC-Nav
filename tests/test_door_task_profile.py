from __future__ import annotations

import unittest

from src.agent_caption import build_agent_caption
from src.cup_confirmation import (
    CupConfirmationConfig,
    evaluate_cup_confirmation,
)
from src.online_semantic_task_planner import build_online_planner_request
from src.semantic_task_profile import DOOR_TASK_PROFILE


class DoorTaskProfileTests(unittest.TestCase):
    def test_profile_excludes_cabinet_doors_from_target_aliases(self) -> None:
        self.assertEqual(
            DOOR_TASK_PROFILE.canonical_label("open-door"),
            "door",
        )
        self.assertEqual(
            DOOR_TASK_PROFILE.canonical_label("cabinet door"),
            "cabinet door",
        )
        self.assertNotIn(
            "cabinet door",
            DOOR_TASK_PROFILE.detector_labels,
        )
        self.assertIn(
            "cabinet door",
            DOOR_TASK_PROFILE.verifier_labels,
        )

    def test_planner_uses_only_grounded_door_candidates(self) -> None:
        tracks = [
            {
                "track_id": 1,
                "label": "door",
                "position_3d": [1.0, 1.0, 2.0],
                "confidence": 0.8,
                "views": 4,
            },
            {
                "track_id": 2,
                "label": "doorway",
                "position_3d": [4.0, 1.0, 2.0],
                "confidence": 0.7,
                "views": 3,
            },
            {
                "track_id": 5,
                "label": "door",
                "position_3d": [1.4, 1.0, 2.2],
                "confidence": 0.6,
                "views": 3,
            },
            {
                "track_id": 3,
                "label": "window",
                "position_3d": [2.0, 1.0, 3.0],
                "confidence": 0.9,
                "views": 5,
            },
            {
                "track_id": 4,
                "label": "cabinet door",
                "position_3d": [2.5, 1.0, 3.0],
                "confidence": 0.9,
                "views": 5,
            },
        ]
        request = build_online_planner_request(
            DOOR_TASK_PROFILE.task_text,
            current_xz=(0.0, 0.0),
            tracks=tracks,
            memory_items=[],
            target_labels=DOOR_TASK_PROFILE.target_aliases,
            support_labels=DOOR_TASK_PROFILE.support_labels,
            target_label=DOOR_TASK_PROFILE.target_label,
            target_merge_radius_m=DOOR_TASK_PROFILE.dynamic_target_merge_radius_m,
        )

        self.assertEqual(
            {item["id"] for item in request["candidate_landmarks"]},
            {"track_1", "track_2"},
        )
        self.assertTrue(
            all(
                item["label"] == "door"
                and item["kind"] == "target_object"
                for item in request["candidate_landmarks"]
            )
        )

    def test_planar_door_can_pass_when_depth_relief_gate_is_disabled(self) -> None:
        config = CupConfirmationConfig(
            min_task_views=2,
            min_visual_passes=2,
            min_visual_negatives=2,
            min_depth_relief_passes=0,
            min_depth_relief_m=0.0,
            max_position_spread_m=0.75,
        )
        observations = [
            {
                "step": 10,
                "camera_xzyaw": [0.0, 0.0, 0.0],
                "position_3d": [1.0, 1.0, 2.0],
                "crop_verifier_status": "pass",
                "crop_verifier_pass": True,
                "crop_positive_score": 0.8,
                "crop_negative_score": 0.1,
                "depth_surface_relief_m": 0.0,
            },
            {
                "step": 20,
                "camera_xzyaw": [0.5, 0.0, 15.0],
                "position_3d": [1.3, 1.0, 2.1],
                "crop_verifier_status": "pass",
                "crop_verifier_pass": True,
                "crop_positive_score": 0.75,
                "crop_negative_score": 0.1,
                "depth_surface_relief_m": 0.0,
            },
        ]

        result = evaluate_cup_confirmation(observations, config)

        self.assertTrue(result["verified"])
        self.assertEqual(result["depth_relief_passes"], 2)

    def test_caption_names_door_instead_of_cup(self) -> None:
        caption = build_agent_caption(
            task=DOOR_TASK_PROFILE.task_text,
            target_label="door",
            interest={"mode": "target_confirmation_finalize"},
            task_plan_events=[
                {
                    "event": "target_confirmation_completed",
                    "candidate_id": "track_7",
                    "status": "verified",
                }
            ],
        )

        self.assertIn("door", caption["plan"])
        self.assertNotIn("cup", caption["plan"])


if __name__ == "__main__":
    unittest.main()
