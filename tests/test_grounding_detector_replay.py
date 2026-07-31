from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "grounding_detector_replay.py"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "grounding_detector_replay",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GroundingDetectorReplayTest(unittest.TestCase):
    def test_track_assignment_matches_nearest_same_label(self) -> None:
        tracks = []
        first = {
            "canonical_label": "door",
            "position_3d": [0.0, 0.0, 0.0],
            "score": 0.5,
        }
        nearby = {
            "canonical_label": "door",
            "position_3d": [0.2, 0.0, 0.0],
            "score": 0.4,
        }
        different_label = {
            "canonical_label": "window",
            "position_3d": [0.1, 0.0, 0.0],
            "score": 0.6,
        }

        MODULE._assign_track(tracks, first, merge_radius_m=0.75)
        MODULE._assign_track(tracks, nearby, merge_radius_m=0.75)
        MODULE._assign_track(
            tracks,
            different_label,
            merge_radius_m=0.75,
        )

        self.assertEqual(first["online_track_id"], 1)
        self.assertEqual(nearby["online_track_id"], 1)
        self.assertEqual(different_label["online_track_id"], 2)
        self.assertEqual(len(tracks), 2)
        self.assertTrue(np.isfinite(tracks[0].position).all())

    def test_frame_selection_is_deterministic(self) -> None:
        frames = [{"frame_index": index} for index in range(10)]

        selected = MODULE._select_frames(
            frames,
            start=1,
            end=8,
            stride=3,
            max_frames=2,
        )

        self.assertEqual(
            [frame["frame_index"] for frame in selected],
            [1, 4],
        )


if __name__ == "__main__":
    unittest.main()
