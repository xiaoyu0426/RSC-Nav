from __future__ import annotations

import unittest

import numpy as np

from src.interest_exploration import (
    InterestConfig,
    approach_cell_for_target,
    choose_semantic_target,
    cluster_frontiers,
    deep_familiarization_status,
    frontier_mask,
    plan_observed_path,
    planning_free_mask,
    rank_frontier_clusters,
    rank_frontiers,
    reachable_free_mask,
    smooth_motion_evidence_weight,
    visible_unknown_gain,
)


class InterestExplorationTests(unittest.TestCase):
    def test_deep_familiarization_requires_saturation_and_reobservation(self) -> None:
        free = np.ones((8, 8), dtype=bool)
        observations = np.full((8, 8), 2, dtype=np.int32)
        ready, metrics = deep_familiarization_status(
            step=120,
            explored_cell_history=[100] * 61,
            free=free,
            observation_count=observations,
            min_steps=100,
            max_steps=200,
            saturation_window=60,
            max_new_cells=20,
            min_reobserve_ratio=0.75,
        )
        self.assertTrue(ready)
        self.assertEqual(metrics["reason"], "saturated_and_reobserved")

    def test_deep_familiarization_uses_causal_max_step_fallback(self) -> None:
        free = np.ones((8, 8), dtype=bool)
        observations = np.ones((8, 8), dtype=np.int32)
        ready, metrics = deep_familiarization_status(
            step=200,
            explored_cell_history=list(range(61)),
            free=free,
            observation_count=observations,
            min_steps=100,
            max_steps=200,
            saturation_window=60,
            max_new_cells=20,
            min_reobserve_ratio=0.75,
        )
        self.assertTrue(ready)
        self.assertEqual(metrics["reason"], "max_familiarization_steps")

    def test_frontier_is_free_cell_adjacent_to_unknown(self) -> None:
        explored = np.zeros((9, 9), dtype=bool)
        explored[3:6, 3:6] = True
        free = explored.copy()

        frontier = frontier_mask(explored, free)

        self.assertFalse(frontier[4, 4])
        self.assertTrue(frontier[3, 4])
        self.assertTrue(frontier[5, 4])

    def test_frontier_ranking_prefers_information_over_revisits(self) -> None:
        explored = np.zeros((15, 15), dtype=bool)
        explored[5:10, 5:10] = True
        free = explored.copy()
        visits = np.zeros_like(explored, dtype=np.float32)
        visits[5, 7] = 30

        ranked = rank_frontiers(
            explored,
            free,
            visits,
            current_cell=(7, 7),
            resolution=0.1,
            config=InterestConfig(min_frontier_distance_m=0.0),
        )

        self.assertTrue(ranked)
        self.assertNotEqual(ranked[0]["cell"], (5, 7))

    def test_frontier_clusters_merge_adjacent_cells(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:5, 2:6] = True
        mask[14:17, 14:18] = True

        clusters = cluster_frontiers(mask, min_cells=4)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(sorted(len(cluster) for cluster in clusters), [12, 12])

    def test_visible_unknown_gain_respects_occluding_wall(self) -> None:
        explored = np.zeros((30, 30), dtype=bool)
        explored[10:21, 2:15] = True
        occupied = np.zeros_like(explored)
        occupied[10:21, 15] = True

        blocked_unknown, _ = visible_unknown_gain(
            explored,
            occupied,
            viewpoint=(15, 8),
            yaw_rad=np.pi / 2.0,
            resolution=0.25,
            max_range_m=5.0,
        )
        occupied[:, 15] = False
        open_unknown, _ = visible_unknown_gain(
            explored,
            occupied,
            viewpoint=(15, 8),
            yaw_rad=np.pi / 2.0,
            resolution=0.25,
            max_range_m=5.0,
        )

        self.assertGreater(open_unknown, blocked_unknown)

    def test_hierarchical_frontier_reports_path_and_cluster_gain(self) -> None:
        explored = np.zeros((30, 30), dtype=bool)
        explored[8:22, 5:20] = True
        free = explored.copy()
        occupied = np.zeros_like(explored)
        observations = np.zeros_like(explored, dtype=np.float32)

        ranked = rank_frontier_clusters(
            explored,
            free,
            occupied,
            observations,
            current_cell=(15, 10),
            resolution=0.1,
            config=InterestConfig(
                min_unknown_gain=0.0,
                min_frontier_distance_m=0.0,
                min_frontier_cluster_cells=2,
            ),
        )

        self.assertTrue(ranked)
        self.assertGreater(ranked[0]["cluster_size"], 1)
        self.assertGreater(ranked[0]["path_distance_m"], 0.0)
        self.assertGreater(ranked[0]["visible_unknown_cells"], 0)

    def test_motion_weight_is_low_stationary_and_damped_when_extreme(self) -> None:
        stationary = smooth_motion_evidence_weight(0.0, 0.0)
        nominal = smooth_motion_evidence_weight(0.25, 15.0)
        extreme = smooth_motion_evidence_weight(3.0, 180.0)

        self.assertLess(stationary, nominal)
        self.assertLess(extreme, nominal)
        self.assertGreaterEqual(stationary, 0.08)

    def test_semantic_target_uses_only_detected_track_state(self) -> None:
        tracks = [
            {
                "track_id": 1,
                "label": "table",
                "confidence": 0.50,
                "views": 4,
                "position_3d": [1.0, 0.8, 0.0],
            },
            {
                "track_id": 2,
                "label": "sink",
                "confidence": 0.70,
                "views": 5,
                "position_3d": [4.0, 0.9, 0.0],
            },
        ]

        target = choose_semantic_target(tracks, current_xz=(0.0, 0.0), scanned_ids=set())
        self.assertIsNotNone(target)
        self.assertEqual(target["track_id"], 1)

        target = choose_semantic_target(tracks, current_xz=(0.0, 0.0), scanned_ids={1})
        self.assertIsNotNone(target)
        self.assertEqual(target["track_id"], 2)

    def test_observed_path_routes_around_wall(self) -> None:
        free = np.ones((15, 15), dtype=bool)
        occupied = np.zeros_like(free)
        occupied[7, :] = True
        occupied[7, 7] = False
        safe = planning_free_mask(free, occupied, inflation_radius_cells=0)
        path = plan_observed_path(safe, (2, 2), (12, 2))
        self.assertTrue(path)
        self.assertIn((7, 7), path)
        self.assertTrue(all(safe[cell] for cell in path))

    def test_unreachable_frontier_is_filtered(self) -> None:
        explored = np.ones((12, 12), dtype=bool)
        free = np.zeros_like(explored)
        free[1:5, 1:5] = True
        free[8:11, 8:11] = True
        explored[4:8, :] = False
        visits = np.zeros_like(explored, dtype=np.int32)
        ranked = rank_frontiers(
            explored,
            free,
            visits,
            current_cell=(2, 2),
            resolution=0.1,
            config=InterestConfig(min_frontier_distance_m=0.0, candidate_stride=1),
        )
        self.assertTrue(all(item["cell"][0] < 5 for item in ranked))

    def test_semantic_approach_cell_is_reachable(self) -> None:
        free = np.zeros((20, 20), dtype=bool)
        free[2:18, 2:18] = True
        reachable = reachable_free_mask(free, (3, 3))
        cell = approach_cell_for_target(reachable, (10, 10), desired_radius_cells=4)
        self.assertIsNotNone(cell)
        self.assertTrue(reachable[cell])
        self.assertAlmostEqual(np.linalg.norm(np.asarray(cell) - np.asarray((10, 10))), 4.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
