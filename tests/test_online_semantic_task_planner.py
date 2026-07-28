from __future__ import annotations

import unittest

from src.online_semantic_task_planner import (
    build_online_planner_request,
    deterministic_online_plan,
    normalize_online_plan,
)


class OnlineSemanticTaskPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracks = [
            {
                "track_id": 4,
                "label": "cup",
                "position_3d": [1.0, 0.8, 2.0],
                "confidence": 0.61,
                "views": 6,
            },
            {
                "track_id": 8,
                "label": "table",
                "position_3d": [3.0, 0.8, 2.0],
                "confidence": 0.72,
                "views": 8,
            },
            {
                "track_id": 9,
                "label": "chair",
                "position_3d": [4.0, 0.8, 2.0],
                "confidence": 0.90,
                "views": 9,
            },
        ]
        self.memory = [
            {
                "semantic_id": 4,
                "confidence": 0.75,
                "freshness": 0.90,
                "status": "active",
                "negative_evidence_count": 0,
            },
            {
                "semantic_id": 8,
                "confidence": 0.80,
                "freshness": 0.85,
                "status": "active",
                "negative_evidence_count": 0,
            },
        ]

    def test_request_contains_only_task_relevant_candidates(self) -> None:
        request = build_online_planner_request(
            "找到所有水杯",
            current_xz=(0.0, 0.0),
            tracks=self.tracks,
            memory_items=self.memory,
        )
        self.assertEqual(
            {item["id"] for item in request["candidate_landmarks"]},
            {"track_4", "track_8"},
        )
        self.assertEqual(request["candidate_landmarks"][0]["kind"], "target_object")

    def test_deterministic_plan_puts_target_before_support_surface(self) -> None:
        request = build_online_planner_request(
            "找到所有水杯",
            current_xz=(0.0, 0.0),
            tracks=self.tracks,
            memory_items=self.memory,
        )
        output = deterministic_online_plan(request)
        self.assertEqual(output["ordered_candidate_ids"][0], "track_4")
        self.assertEqual(output["ordered_candidate_ids"][1], "track_8")

    def test_normalization_removes_invented_ids_and_keeps_exhaustive_fallback(self) -> None:
        request = build_online_planner_request(
            "找到所有水杯",
            current_xz=(0.0, 0.0),
            tracks=self.tracks,
            memory_items=self.memory,
        )
        output = normalize_online_plan(
            {
                "ordered_candidate_ids": ["invented", "track_8"],
                "task_plan": [
                    {"step": 1, "intent": "inspect", "target_id": "invented"}
                ],
                "stop_probability": 2.0,
            },
            request,
        )
        self.assertEqual(output["ordered_candidate_ids"], ["track_8", "track_4"])
        self.assertEqual(
            {item["target_id"] for item in output["task_plan"]},
            {"track_4", "track_8"},
        )
        self.assertEqual(output["stop_probability"], 1.0)

    def test_nearby_support_tracks_are_collapsed_into_search_regions(self) -> None:
        tracks = self.tracks + [
            {
                "track_id": 10,
                "label": "counter",
                "position_3d": [3.3, 0.8, 2.1],
                "confidence": 0.82,
                "views": 7,
            },
            {
                "track_id": 11,
                "label": "sink",
                "position_3d": [8.0, 0.8, 2.0],
                "confidence": 0.70,
                "views": 7,
            },
        ]
        request = build_online_planner_request(
            "找到所有水杯",
            current_xz=(0.0, 0.0),
            tracks=tracks,
            memory_items=self.memory,
            max_support_candidates=10,
            support_merge_radius_m=1.25,
        )
        support_ids = {
            item["id"]
            for item in request["candidate_landmarks"]
            if item["kind"] == "support_surface"
        }
        self.assertEqual(len(support_ids), 2)
        self.assertIn("track_11", support_ids)


if __name__ == "__main__":
    unittest.main()
