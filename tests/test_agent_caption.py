from __future__ import annotations

import unittest

from src.agent_caption import build_agent_caption


class AgentCaptionTests(unittest.TestCase):
    def test_hidden_task_caption_does_not_claim_task_reasoning(self) -> None:
        caption = build_agent_caption(
            task=None,
            interest={"mode": "frontier_exploration"},
        )

        self.assertEqual(caption["stage"], "FAMILIARIZE")
        self.assertIn("task is still hidden", caption["why"].lower())
        self.assertIn("No task-specific evidence", caption["evidence"])

    def test_positive_support_evidence_remains_a_hypothesis(self) -> None:
        caption = build_agent_caption(
            task="find cups",
            interest={
                "mode": "semantic_target_scan",
                "task_search_ranking": [
                    {
                        "candidate_id": "track_99",
                        "label": "table",
                        "posterior": 0.75,
                    }
                ],
            },
            task_plan_events=[
                {
                    "event": "support_surface_inspection_completed",
                    "candidate_id": "track_99",
                    "outcome": "target_evidence_observed",
                    "observable_scan": True,
                    "observed_target_candidate_ids": ["track_170"],
                    "belief_update": {
                        "prior_posterior": 0.75,
                        "posterior": 0.94,
                    },
                }
            ],
        )

        self.assertIn("Cup evidence found", caption["plan"])
        self.assertIn("not yet verified", caption["why"])
        self.assertIn("0.75 -> 0.94", caption["evidence"])

    def test_rejected_candidate_is_not_reported_as_found(self) -> None:
        caption = build_agent_caption(
            task="find cups",
            interest={"mode": "cup_confirmation_finalize"},
            task_plan_events=[
                {
                    "event": "cup_confirmation_completed",
                    "candidate_id": "track_90",
                    "status": "rejected_planar_surface",
                    "task_independent_views": 2,
                    "visual_passes": 0,
                    "visual_negatives": 2,
                }
            ],
        )

        self.assertEqual(caption["stage"], "REPLAN")
        self.assertIn("Reject cup hypothesis", caption["plan"])
        self.assertNotIn("verified", caption["plan"].lower())

    def test_verified_word_requires_verified_status(self) -> None:
        rejected = build_agent_caption(
            task="find cups",
            interest={"mode": "cup_confirmation_finalize"},
            task_plan_events=[
                {
                    "event": "cup_confirmation_completed",
                    "candidate_id": "track_1",
                    "status": "rejected_visual_verifier",
                }
            ],
        )
        verified = build_agent_caption(
            task="find cups",
            interest={"mode": "cup_confirmation_finalize"},
            task_plan_events=[
                {
                    "event": "cup_confirmation_completed",
                    "candidate_id": "track_2",
                    "status": "verified",
                }
            ],
        )

        self.assertNotIn("Verify cup", rejected["plan"])
        self.assertEqual(verified["stage"], "CONFIRMED")
        self.assertIn("Verify cup", verified["plan"])

    def test_negative_support_evidence_lowers_belief(self) -> None:
        caption = build_agent_caption(
            task="find cups",
            interest={"mode": "semantic_target_scan"},
            task_plan_events=[
                {
                    "event": "support_surface_inspection_completed",
                    "candidate_id": "track_254",
                    "outcome": "no_target_evidence_observed",
                    "observable_scan": True,
                    "belief_update": {
                        "prior_posterior": 0.70,
                        "posterior": 0.21,
                    },
                }
            ],
        )

        self.assertIn("No cup evidence", caption["plan"])
        self.assertIn("0.70 -> 0.21", caption["evidence"])


if __name__ == "__main__":
    unittest.main()
