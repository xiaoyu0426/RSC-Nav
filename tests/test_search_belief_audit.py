from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.search_belief_audit import audit_search_belief_run


class SearchBeliefAuditTests(unittest.TestCase):
    def test_complete_api_search_belief_run_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            planner_dir = run_dir / "task_planner"
            planner_dir.mkdir()
            (planner_dir / "planner_request.json").write_text(
                json.dumps(
                    {
                        "candidate_landmarks": [
                            {"id": "track_1"},
                            {"id": "track_2"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "online_summary.json").write_text(
                json.dumps(
                    {
                        "num_steps": 500,
                        "task_injection_step": 300,
                        "task_execution_steps": 200,
                        "num_confirmed_cups": 0,
                        "confirmed_cups": [],
                        "causal_invariants": {
                            "all_decisions_use_current_or_past_frames": True,
                            "task_hidden_until_memory_ready": True,
                        },
                        "scene_vlm": {
                            "mode_used": "api",
                            "model": "qwen3-vl-plus",
                            "keyframes": [
                                {
                                    "frame_id": "frame_0100",
                                    "frame_index": 100,
                                }
                            ],
                        },
                        "task_planner": {
                            "mode_used": "api",
                            "model": "qwen3-max",
                        },
                        "scene_understanding": {
                            "candidate_assessments": [
                                {"candidate_id": "track_1"},
                                {"candidate_id": "track_2"},
                            ],
                            "regions": [
                                {
                                    "region_id": "kitchen",
                                    "anchor_candidate_ids": ["track_2"],
                                    "anchor_xz": [1.0, 2.0],
                                }
                            ],
                            "contract": {
                                "likelihood_is_search_prior_not_object_confirmation": True
                            },
                        },
                        "task_plan_events": [
                            {"event": "search_priority_replanned"},
                            {
                                "step": 410,
                                "event": "support_surface_inspection_completed",
                                "candidate_id": "track_1",
                                "outcome": "target_evidence_observed",
                                "observable_scan": True,
                                "observed_target_candidate_ids": ["track_9"],
                                "belief_update": {
                                    "applied": True,
                                    "outcome": "target_evidence_observed",
                                    "prior_posterior": 0.6,
                                    "posterior": 0.9,
                                },
                            },
                            {
                                "step": 450,
                                "event": "support_surface_inspection_completed",
                                "candidate_id": "track_2",
                                "outcome": "no_target_evidence_observed",
                                "observable_scan": True,
                                "observed_target_candidate_ids": [],
                                "belief_update": {
                                    "applied": True,
                                    "outcome": "no_target_evidence_observed",
                                    "prior_posterior": 0.7,
                                    "posterior": 0.2,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = audit_search_belief_run(
                run_dir,
                require_api=True,
                require_both_support_outcomes=True,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failed_checks"], [])
        self.assertEqual(report["metrics"]["num_support_inspections"], 2)

    def test_invented_vlm_candidate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            planner_dir = run_dir / "task_planner"
            planner_dir.mkdir()
            (planner_dir / "planner_request.json").write_text(
                json.dumps({"candidate_landmarks": [{"id": "track_1"}]}),
                encoding="utf-8",
            )
            (run_dir / "online_summary.json").write_text(
                json.dumps(
                    {
                        "task_injection_step": 10,
                        "causal_invariants": {
                            "all_decisions_use_current_or_past_frames": True,
                            "task_hidden_until_memory_ready": True,
                        },
                        "scene_vlm": {
                            "keyframes": [{"frame_index": 5}]
                        },
                        "scene_understanding": {
                            "candidate_assessments": [
                                {"candidate_id": "invented"}
                            ],
                            "regions": [],
                            "contract": {
                                "likelihood_is_search_prior_not_object_confirmation": True
                            },
                        },
                        "task_plan_events": [],
                    }
                ),
                encoding="utf-8",
            )

            report = audit_search_belief_run(run_dir)

        self.assertFalse(report["passed"])
        self.assertIn(
            "vlm_candidate_ids_grounded",
            report["failed_checks"],
        )


if __name__ == "__main__":
    unittest.main()
