from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cup_confirmation import (  # noqa: E402
    CupConfirmationConfig,
    append_independent_observation,
    estimate_depth_surface_relief,
    evaluate_cup_confirmation,
    score_crop_verifier,
)


def _observation(
    step: int,
    yaw: float,
    position: tuple[float, float, float] = (1.0, 0.8, 2.0),
    visual_pass: bool = True,
    camera_xz: tuple[float, float] = (0.0, 0.0),
) -> dict:
    return {
        "step": step,
        "camera_xzyaw": [camera_xz[0], camera_xz[1], yaw],
        "position_3d": list(position),
        "crop_verifier_pass": visual_pass,
        "crop_verifier_status": "pass" if visual_pass else "negative",
        "crop_positive_score": 0.42 if visual_pass else 0.20,
        "crop_negative_score": 0.18 if visual_pass else 0.35,
        "depth_surface_relief_m": 0.08,
    }


class CupConfirmationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CupConfirmationConfig()

    def test_requires_new_independent_task_views(self) -> None:
        observations: list[dict] = []
        self.assertTrue(
            append_independent_observation(
                observations,
                _observation(step=401, yaw=0.0),
                self.config,
            )
        )
        self.assertFalse(
            append_independent_observation(
                observations,
                _observation(step=402, yaw=15.0),
                self.config,
            )
        )
        result = evaluate_cup_confirmation(observations, self.config)
        self.assertEqual(result["status"], "insufficient_task_views")

    def test_accepts_consistent_visual_positive_views(self) -> None:
        observations: list[dict] = []
        for item in (
            _observation(step=401, yaw=0.0),
            _observation(
                step=403,
                yaw=30.0,
                position=(1.04, 0.82, 2.02),
                camera_xz=(0.40, 0.0),
            ),
        ):
            self.assertTrue(
                append_independent_observation(observations, item, self.config)
            )
        result = evaluate_cup_confirmation(observations, self.config)
        self.assertTrue(result["verified"])
        self.assertEqual(result["status"], "verified")

    def test_rejects_geometry_inconsistency(self) -> None:
        observations = [
            _observation(step=401, yaw=0.0),
            _observation(
                step=403,
                yaw=30.0,
                position=(1.6, 0.8, 2.0),
                camera_xz=(0.40, 0.0),
            ),
        ]
        result = evaluate_cup_confirmation(observations, self.config)
        self.assertEqual(result["status"], "rejected_geometry_inconsistent")

    def test_rejects_crop_verifier_failures(self) -> None:
        observations = [
            _observation(step=401, yaw=0.0, visual_pass=False),
            _observation(
                step=403,
                yaw=30.0,
                visual_pass=False,
                camera_xz=(0.40, 0.0),
            ),
        ]
        result = evaluate_cup_confirmation(observations, self.config)
        self.assertEqual(result["status"], "rejected_visual_verifier")
        self.assertFalse(result["verified"])

    def test_rotation_without_translation_is_not_independent(self) -> None:
        observations = [_observation(step=401, yaw=0.0)]
        self.assertFalse(
            append_independent_observation(
                observations,
                _observation(step=402, yaw=90.0),
                self.config,
            )
        )

    def test_verifier_error_is_retryable_not_negative(self) -> None:
        first = _observation(step=401, yaw=0.0)
        first["crop_verifier_status"] = "error"
        first["crop_verifier_pass"] = False
        observations = [first]
        self.assertTrue(
            append_independent_observation(
                observations,
                _observation(step=402, yaw=0.0),
                self.config,
            )
        )
        result = evaluate_cup_confirmation(observations, self.config)
        self.assertEqual(result["status"], "insufficient_task_views")
        self.assertEqual(result["verifier_errors"], 1)

    def test_crop_verifier_requires_target_region_overlap(self) -> None:
        result = score_crop_verifier(
            detections=[
                {"label": "real drinking cup", "score": 0.81, "box": [70, 5, 92, 30]},
                {"label": "wall outlet", "score": 0.52, "box": [18, 18, 42, 44]},
            ],
            positive_labels={"real drinking cup", "mug"},
            target_box=[15, 15, 45, 48],
            min_positive_score=0.30,
            min_score_margin=0.05,
        )
        self.assertEqual(result["crop_verifier_status"], "negative")
        self.assertEqual(result["crop_associated_detection_count"], 1)

    def test_rejects_planar_depth_candidate(self) -> None:
        observations = [
            _observation(step=401, yaw=0.0),
            _observation(step=403, yaw=30.0, camera_xz=(0.40, 0.0)),
        ]
        for item in observations:
            item["depth_surface_relief_m"] = 0.005
        result = evaluate_cup_confirmation(observations, self.config)
        self.assertEqual(result["status"], "rejected_planar_surface")

    def test_depth_relief_separates_object_from_flat_wall(self) -> None:
        depth = np.full((80, 80), 2.0, dtype=np.float32)
        depth[25:55, 30:50] = 1.82
        object_result = estimate_depth_surface_relief(
            depth,
            [30, 25, 50, 55],
        )
        wall_result = estimate_depth_surface_relief(
            np.full((80, 80), 2.0, dtype=np.float32),
            [30, 25, 50, 55],
        )
        self.assertTrue(object_result["valid"])
        self.assertGreater(object_result["relief_m"], 0.10)
        self.assertAlmostEqual(wall_result["relief_m"], 0.0, places=4)

    def test_conflicting_visual_evidence_is_not_verified(self) -> None:
        observations = [
            _observation(step=401, yaw=0.0),
            _observation(step=403, yaw=15.0, camera_xz=(0.40, 0.0)),
            _observation(
                step=405,
                yaw=30.0,
                visual_pass=False,
                camera_xz=(0.80, 0.0),
            ),
            _observation(
                step=407,
                yaw=45.0,
                visual_pass=False,
                camera_xz=(1.20, 0.0),
            ),
        ]
        result = evaluate_cup_confirmation(observations, self.config)
        self.assertEqual(result["status"], "conflicting_visual_evidence")
        self.assertFalse(result["verified"])

    def test_consistent_inliers_recover_from_one_depth_outlier(self) -> None:
        observations = [
            _observation(step=401, yaw=0.0),
            _observation(
                step=403,
                yaw=15.0,
                position=(1.8, 0.8, 2.0),
                camera_xz=(0.40, 0.0),
            ),
            _observation(
                step=405,
                yaw=30.0,
                position=(1.03, 0.81, 2.02),
                camera_xz=(0.80, 0.0),
            ),
        ]
        result = evaluate_cup_confirmation(observations, self.config)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["geometry_inlier_views"], 2)
        self.assertGreater(result["raw_position_spread_m"], 0.30)


if __name__ == "__main__":
    unittest.main()
