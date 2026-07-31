from __future__ import annotations

import importlib.util
import subprocess
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
    def test_cli_help_runs_without_manual_pythonpath(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--frames-metadata", completed.stdout)

    def test_implementation_manifest_covers_runtime_dependencies(self) -> None:
        manifest = MODULE._implementation_artifact_manifest(
            [
                SCRIPT,
                ROOT / "scripts" / "m25_groundingdino_export.py",
                ROOT / "src" / "semantic_task_profile.py",
            ]
        )

        self.assertEqual(
            {record["path"] for record in manifest["files"]},
            {
                "scripts/grounding_detector_replay.py",
                "scripts/m25_groundingdino_export.py",
                "src/semantic_task_profile.py",
            },
        )
        self.assertEqual(
            manifest["manifest_sha256"],
            MODULE._canonical_sha256({"files": manifest["files"]}),
        )

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

    def test_door_window_suppression_is_causal_and_score_gated(self) -> None:
        profile = MODULE.get_task_profile("door")
        detections = [
            {
                "label": "door",
                "score": 0.40,
                "box": [0.0, 0.0, 10.0, 10.0],
            },
            {
                "label": "doorway",
                "score": 0.70,
                "box": [20.0, 0.0, 30.0, 10.0],
            },
            {
                "label": "window",
                "score": 0.50,
                "box": [0.0, 0.0, 10.0, 10.0],
            },
            {
                "label": "window",
                "score": 0.60,
                "box": [20.0, 0.0, 30.0, 10.0],
            },
            {
                "label": "chair",
                "score": 0.80,
                "box": [0.0, 0.0, 10.0, 10.0],
            },
        ]

        kept = MODULE._suppress_door_with_window(
            detections,
            profile=profile,
            iou_threshold=0.5,
        )

        self.assertNotIn(detections[0], kept)
        self.assertIn(detections[1], kept)
        self.assertIn(detections[2], kept)
        self.assertIn(detections[3], kept)
        self.assertIn(detections[4], kept)

    def test_door_window_suppression_respects_iou_and_disabled_mode(self) -> None:
        profile = MODULE.get_task_profile("door")
        detections = [
            {
                "label": "door",
                "score": 0.40,
                "box": [0.0, 0.0, 10.0, 10.0],
            },
            {
                "label": "window",
                "score": 0.50,
                "box": [8.0, 0.0, 18.0, 10.0],
            },
        ]

        kept = MODULE._suppress_door_with_window(
            detections,
            profile=profile,
            iou_threshold=0.5,
        )

        self.assertEqual(kept, detections)
        self.assertEqual(
            MODULE._suppress_door_with_window(
                detections,
                profile=profile,
                iou_threshold=None,
            ),
            detections,
        )

    def test_box_iou_rejects_invalid_boxes(self) -> None:
        self.assertAlmostEqual(
            MODULE._box_iou(
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 0.0, 15.0, 10.0],
            ),
            1.0 / 3.0,
        )
        with self.assertRaisesRegex(ValueError, "four-element"):
            MODULE._box_iou([0.0, 1.0], [0.0, 0.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
