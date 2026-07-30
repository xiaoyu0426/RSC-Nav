from __future__ import annotations

import json
import unittest

from src.target_vlm_verifier import (
    apply_target_vlm_verdict,
    extract_target_vlm_result,
    should_request_target_vlm,
)


class TargetVlmVerifierTests(unittest.TestCase):
    def test_extracts_list_response_and_normalizes_door_verdict(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [
                                {
                                    "candidate_id": "track_172",
                                    "verdict": "door",
                                    "confidence": 95,
                                    "observed_type": "hinged door",
                                    "reason": "floor-level passage",
                                }
                            ]
                        )
                    }
                }
            ]
        }

        result = extract_target_vlm_result(
            raw,
            candidate_id="track_172",
            target_label="door",
        )

        self.assertEqual(result["verdict"], "target")
        self.assertAlmostEqual(result["confidence"], 0.95)

    def test_vlm_target_overrides_ambiguous_grounding_result(self) -> None:
        result = apply_target_vlm_verdict(
            {
                "status": "rejected_visual_verifier",
                "verified": False,
            },
            {
                "candidate_id": "track_172",
                "verdict": "target",
                "confidence": 0.95,
            },
            min_confidence=0.70,
        )

        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["verified"])

    def test_vlm_non_target_rejects_false_positive(self) -> None:
        result = apply_target_vlm_verdict(
            {
                "status": "insufficient_visual_evidence",
                "verified": False,
            },
            {
                "candidate_id": "track_277",
                "verdict": "not_target",
                "confidence": 0.98,
            },
            min_confidence=0.70,
        )

        self.assertEqual(result["status"], "rejected_vlm_verifier")
        self.assertFalse(result["verified"])

    def test_unclear_vlm_result_remains_retryable(self) -> None:
        result = apply_target_vlm_verdict(
            {
                "status": "rejected_visual_verifier",
                "verified": False,
            },
            {
                "candidate_id": "track_172",
                "verdict": "unclear",
                "confidence": 0.0,
            },
            min_confidence=0.70,
        )

        self.assertEqual(result["status"], "insufficient_vlm_evidence")
        self.assertFalse(result["verified"])

    def test_requires_two_geometry_consistent_crop_views(self) -> None:
        self.assertTrue(
            should_request_target_vlm(
                {
                    "status": "rejected_visual_verifier",
                    "task_independent_views": 2,
                    "geometry_inlier_views": 2,
                    "crop_paths": ["a.jpg", "b.jpg"],
                },
                min_task_views=2,
            )
        )
        self.assertFalse(
            should_request_target_vlm(
                {
                    "status": "rejected_geometry_inconsistent",
                    "task_independent_views": 2,
                    "geometry_inlier_views": 1,
                    "crop_paths": ["a.jpg", "b.jpg"],
                },
                min_task_views=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
