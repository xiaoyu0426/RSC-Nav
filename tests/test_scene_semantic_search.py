from __future__ import annotations

import unittest

from src.scene_semantic_search import (
    apply_search_evidence,
    enrich_planner_request,
    initialize_search_beliefs,
    normalize_scene_understanding,
    rank_search_candidates,
    select_scene_keyframes,
)


class SceneSemanticSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {
                "id": "track_1",
                "kind": "target_object",
                "label": "cup",
                "world_xz": [5.0, 0.0],
                "confidence": 0.55,
                "freshness": 0.9,
                "independent_views": 5,
            },
            {
                "id": "track_2",
                "kind": "support_surface",
                "label": "sink",
                "world_xz": [3.0, 0.0],
                "confidence": 0.75,
                "freshness": 0.9,
                "independent_views": 6,
            },
            {
                "id": "track_3",
                "kind": "support_surface",
                "label": "table",
                "world_xz": [1.0, 0.0],
                "confidence": 0.80,
                "freshness": 0.9,
                "independent_views": 6,
            },
        ]

    def test_keyframes_cover_candidate_views_before_context(self) -> None:
        frames = [
            {"frame_index": index, "rgb_path": f"frame_{index:04d}.jpg"}
            for index in range(10)
        ]
        detections = [
            {
                "frame_index": 2,
                "online_track_id": 2,
                "score": 0.7,
                "box": [0, 0, 80, 80],
            },
            {
                "frame_index": 7,
                "online_track_id": 3,
                "score": 0.8,
                "box": [0, 0, 90, 90],
            },
        ]

        selected = select_scene_keyframes(
            frames,
            detections,
            candidate_ids=["track_2", "track_3"],
            max_images=3,
        )

        selected_ids = {
            candidate_id
            for item in selected
            for candidate_id in item["visible_candidate_ids"]
        }
        self.assertEqual(selected_ids, {"track_2", "track_3"})
        self.assertEqual(len(selected), 3)

    def test_normalization_is_fail_closed_for_ids_and_probabilities(self) -> None:
        normalized = normalize_scene_understanding(
            {
                "scene_summary": "A bathroom and dining area are visible.",
                "candidate_assessments": [
                    {
                        "candidate_id": "track_2",
                        "room_type": "bathroom",
                        "support_role": "washbasin",
                        "target_likelihood": 2.0,
                        "visual_confidence": 0.8,
                        "evidence_frame_ids": ["frame_0002", "invented"],
                        "reason": "A sink is visible.",
                    },
                    {
                        "candidate_id": "invented",
                        "target_likelihood": 1.0,
                    },
                ],
            },
            candidates=self.candidates,
            valid_frame_ids=["frame_0002"],
        )

        by_id = {
            item["candidate_id"]: item
            for item in normalized["candidate_assessments"]
        }
        self.assertEqual(set(by_id), {"track_1", "track_2", "track_3"})
        self.assertEqual(by_id["track_2"]["target_likelihood"], 1.0)
        self.assertEqual(
            by_id["track_2"]["evidence_frame_ids"],
            ["frame_0002"],
        )
        self.assertTrue(
            normalized["contract"][
                "likelihood_is_search_prior_not_object_confirmation"
            ]
        )
        bathroom_region = next(
            item
            for item in normalized["regions"]
            if item["room_type"] == "bathroom"
        )
        self.assertEqual(
            bathroom_region["anchor_candidate_ids"],
            ["track_2"],
        )
        self.assertEqual(bathroom_region["anchor_xz"], [3.0, 0.0])

    def test_enriched_request_exposes_grounded_scene_semantics(self) -> None:
        scene = normalize_scene_understanding(
            {
                "candidate_assessments": [
                    {
                        "candidate_id": "track_2",
                        "room_type": "bathroom",
                        "support_role": "washbasin",
                        "target_likelihood": 0.82,
                        "visual_confidence": 0.77,
                        "evidence_frame_ids": ["frame_0002"],
                        "reason": "Likely place for a cup.",
                    }
                ]
            },
            candidates=self.candidates,
            valid_frame_ids=["frame_0002"],
        )
        request = enrich_planner_request(
            {
                "schema_version": "old",
                "candidate_landmarks": self.candidates,
                "constraints": [],
            },
            scene,
        )

        self.assertEqual(
            request["candidate_landmarks"][1]["scene_semantics"][
                "room_type"
            ],
            "bathroom",
        )
        self.assertEqual(
            request["schema_version"],
            "phase5a_online_task_request_v2",
        )

    def test_dynamic_ranking_uses_location_prior_and_completed_evidence(self) -> None:
        scene = normalize_scene_understanding(
            {
                "candidate_assessments": [
                    {
                        "candidate_id": "track_2",
                        "room_type": "bathroom",
                        "support_role": "washbasin",
                        "target_likelihood": 0.88,
                        "visual_confidence": 0.82,
                        "reason": "Bathroom sink.",
                    },
                    {
                        "candidate_id": "track_3",
                        "room_type": "living_room",
                        "support_role": "side table",
                        "target_likelihood": 0.35,
                        "visual_confidence": 0.75,
                        "reason": "Less likely support.",
                    },
                ]
            },
            candidates=self.candidates,
            valid_frame_ids=[],
        )

        ranked = rank_search_candidates(
            self.candidates,
            current_xz=(0.0, 0.0),
            scene_understanding=scene,
            planner_order=["track_3", "track_2", "track_1"],
        )
        self.assertEqual(ranked[0]["candidate_id"], "track_1")
        self.assertLess(
            next(
                item["rank"]
                for item in ranked
                if item["candidate_id"] == "track_2"
            ),
            next(
                item["rank"]
                for item in ranked
                if item["candidate_id"] == "track_3"
            ),
        )

        after_inspection = rank_search_candidates(
            self.candidates,
            current_xz=(0.0, 0.0),
            scene_understanding=scene,
            completed_ids={"track_1", "track_2"},
        )
        self.assertEqual(
            [item["candidate_id"] for item in after_inspection],
            ["track_3"],
        )

    def test_negative_evidence_updates_posterior_once_and_reorders(self) -> None:
        scene = normalize_scene_understanding(
            {
                "candidate_assessments": [
                    {
                        "candidate_id": "track_2",
                        "room_type": "bathroom",
                        "support_role": "washbasin",
                        "target_likelihood": 0.88,
                        "visual_confidence": 0.9,
                    },
                    {
                        "candidate_id": "track_3",
                        "room_type": "dining_room",
                        "support_role": "dining table",
                        "target_likelihood": 0.60,
                        "visual_confidence": 0.9,
                    },
                ]
            },
            candidates=self.candidates[1:],
            valid_frame_ids=[],
        )
        beliefs = initialize_search_beliefs(
            self.candidates[1:],
            scene,
            step=10,
        )
        before = rank_search_candidates(
            self.candidates[1:],
            current_xz=(0.0, 0.0),
            scene_understanding=scene,
            beliefs=beliefs,
        )
        self.assertEqual(before[0]["candidate_id"], "track_2")

        update = apply_search_evidence(
            beliefs,
            candidate_id="track_2",
            event_id="surface_scan:track_2:10",
            outcome="no_target_evidence_observed",
            step=20,
            observable=True,
        )
        duplicate = apply_search_evidence(
            beliefs,
            candidate_id="track_2",
            event_id="surface_scan:track_2:10",
            outcome="no_target_evidence_observed",
            step=21,
            observable=True,
        )
        self.assertTrue(update["applied"])
        self.assertFalse(duplicate["applied"])
        self.assertLess(beliefs["track_2"]["posterior"], 0.50)

        after = rank_search_candidates(
            self.candidates[1:],
            current_xz=(0.0, 0.0),
            scene_understanding=scene,
            beliefs=beliefs,
        )
        self.assertEqual(after[0]["candidate_id"], "track_3")

    def test_inconclusive_cup_attempt_allows_support_surface_to_advance(self) -> None:
        scene = normalize_scene_understanding(
            {
                "candidate_assessments": [
                    {
                        "candidate_id": "track_1",
                        "target_likelihood": 0.75,
                        "visual_confidence": 0.7,
                    },
                    {
                        "candidate_id": "track_2",
                        "room_type": "bathroom",
                        "target_likelihood": 0.88,
                        "visual_confidence": 0.9,
                    },
                ]
            },
            candidates=self.candidates[:2],
            valid_frame_ids=[],
        )
        beliefs = initialize_search_beliefs(
            self.candidates[:2],
            scene,
            step=0,
        )
        apply_search_evidence(
            beliefs,
            candidate_id="track_1",
            event_id="cup_attempt_1",
            outcome="inconclusive_confirmation",
            step=5,
        )
        ranked = rank_search_candidates(
            self.candidates[:2],
            current_xz=(0.0, 0.0),
            scene_understanding=scene,
            attempts={"track_1": 1},
            beliefs=beliefs,
        )
        self.assertEqual(ranked[0]["candidate_id"], "track_2")


if __name__ == "__main__":
    unittest.main()
