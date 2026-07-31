from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "grounding_semantic_replay_gt.py"
SPEC = importlib.util.spec_from_file_location(
    "grounding_semantic_replay_gt",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GroundingSemanticReplayGroundTruthTest(unittest.TestCase):
    def test_scene_asset_manifest_hashes_complete_semantic_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_dir = root / "00861-test"
            scene_dir.mkdir()
            for name, content in (
                ("test.basis.glb", b"render"),
                ("test.glb", b"source"),
                ("test.basis.navmesh", b"navmesh"),
                ("test.semantic.glb", b"semantic"),
                ("test.semantic.txt", b"labels"),
            ):
                (scene_dir / name).write_bytes(content)
            dataset_config = root / "dataset.scene_dataset_config.json"
            dataset_config.write_text("{}", encoding="utf-8")

            manifest = MODULE._scene_asset_manifest(
                scene_dir / "test.basis.glb",
                dataset_config=dataset_config,
            )

            self.assertEqual(manifest["scene_id"], "00861-test")
            self.assertEqual(manifest["scene_key"], "test")
            self.assertEqual(len(manifest["files"]), 6)
            first_hash = MODULE._scene_asset_identity_sha256(manifest)
            aliased_manifest = dict(manifest)
            aliased_manifest["scene_id"] = "fake-scene"
            self.assertEqual(
                first_hash,
                MODULE._scene_asset_identity_sha256(aliased_manifest),
            )
            (scene_dir / "test.semantic.txt").write_bytes(b"new-labels")
            changed_manifest = MODULE._scene_asset_manifest(
                scene_dir / "test.basis.glb",
                dataset_config=dataset_config,
            )
            self.assertNotEqual(
                first_hash,
                MODULE._scene_asset_identity_sha256(changed_manifest),
            )

            (scene_dir / "test.semantic.txt").unlink()
            with self.assertRaisesRegex(ValueError, "semantic.txt"):
                MODULE._scene_asset_manifest(
                    scene_dir / "test.basis.glb",
                    dataset_config=dataset_config,
                )

    def test_visible_instances_extracts_exclusive_box(self) -> None:
        semantic = np.asarray(
            [
                [0, 0, 0, 0],
                [0, 7, 7, 0],
                [0, 7, 7, 0],
            ],
            dtype=np.int32,
        )
        object_index = {
            7: {
                "semantic_id": 7,
                "object_id": "door_7",
                "raw_category": "door",
                "canonical_label": "door",
                "world_center_xyz": [1.0, 2.0, 3.0],
                "world_size_xyz": [0.5, 2.0, 0.1],
            }
        }

        rows = MODULE._visible_instances(
            semantic,
            object_index,
            min_area_px=2,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["area_px"], 4)
        self.assertEqual(rows[0]["box"], [1, 1, 3, 3])

    def test_visible_instances_respects_minimum_area(self) -> None:
        semantic = np.asarray([[5, 0], [0, 0]], dtype=np.int32)
        object_index = {
            5: {
                "semantic_id": 5,
                "object_id": "window_5",
                "raw_category": "window",
                "canonical_label": "window",
                "world_center_xyz": [0.0, 0.0, 0.0],
                "world_size_xyz": [1.0, 1.0, 0.1],
            }
        }

        rows = MODULE._visible_instances(
            semantic,
            object_index,
            min_area_px=2,
        )

        self.assertEqual(rows, [])

    def test_category_mapping_is_strict(self) -> None:
        self.assertEqual(MODULE._canonical_gt_label("door"), "door")
        self.assertEqual(MODULE._canonical_gt_label(" Window "), "window")
        self.assertIsNone(MODULE._canonical_gt_label("cabinet door"))
        self.assertIsNone(MODULE._canonical_gt_label("door frame"))

    def test_object_aabb_accepts_habitat_callable_api(self) -> None:
        class CallableAabb:
            def center(self):
                return [1.0, 2.0, 3.0]

            def size(self):
                return [0.5, 2.0, 0.1]

        class SemanticObject:
            aabb = CallableAabb()

        center, size = MODULE._object_aabb(SemanticObject())

        self.assertEqual(center, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(size[0], 0.5)
        self.assertAlmostEqual(size[1], 2.0)
        self.assertAlmostEqual(size[2], 0.1)

    def test_rgb_integrity_requires_complete_bounded_checks(self) -> None:
        good = [
            {
                "available": True,
                "mae": 1.8,
                "p95_abs_error": 6.0,
            },
            {
                "available": True,
                "mae": 2.0,
                "p95_abs_error": 8.0,
            },
        ]

        passed = MODULE._rgb_integrity_report(
            good,
            enabled=True,
            max_mae=3.0,
            max_p95_abs_error=12.0,
        )
        missing = MODULE._rgb_integrity_report(
            [*good, {"available": False}],
            enabled=True,
            max_mae=3.0,
            max_p95_abs_error=12.0,
        )
        excessive = MODULE._rgb_integrity_report(
            [{**good[0], "mae": 3.1}],
            enabled=True,
            max_mae=3.0,
            max_p95_abs_error=12.0,
        )

        self.assertTrue(passed["passed"])
        self.assertEqual(passed["observed_max_mae"], 2.0)
        self.assertFalse(missing["passed"])
        self.assertFalse(excessive["passed"])

    def test_depth_replay_integrity_checks_values_and_validity(self) -> None:
        source = np.asarray(
            [[1.0, 2.0], [0.0, 4.0]],
            dtype=np.float32,
        )
        exact = MODULE._depth_replay_check(
            source.copy(),
            source,
            source_path_value="depth.npy",
            frame_index=7,
            large_error_m=0.01,
        )
        shifted = MODULE._depth_replay_check(
            np.asarray([[1.1, 2.0], [3.0, 4.0]], dtype=np.float32),
            source,
            source_path_value="depth.npy",
            frame_index=7,
            large_error_m=0.01,
        )

        self.assertTrue(exact["available"])
        self.assertEqual(exact["mae_m"], 0.0)
        self.assertEqual(exact["validity_disagreement_ratio"], 0.0)
        self.assertGreater(shifted["mae_m"], 0.0)
        self.assertEqual(shifted["validity_disagreement_ratio"], 0.25)

        passed = MODULE._depth_integrity_report(
            [exact],
            enabled=True,
            max_mae_m=1e-4,
            max_p95_abs_error_m=1e-4,
            max_validity_disagreement_ratio=0.0,
            max_large_error_ratio=1e-4,
            large_error_m=0.01,
        )
        failed = MODULE._depth_integrity_report(
            [shifted],
            enabled=True,
            max_mae_m=1e-4,
            max_p95_abs_error_m=1e-4,
            max_validity_disagreement_ratio=0.0,
            max_large_error_ratio=1e-4,
            large_error_m=0.01,
        )

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])

    def test_rotation_matrix_conversion_matches_yaw_quaternion(self) -> None:
        angle = np.deg2rad(30.0)
        matrix = [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]

        quaternion = MODULE._quaternion_from_rotation_matrix(
            matrix,
            lambda w, x, y, z: (w, x, y, z),
        )

        self.assertAlmostEqual(quaternion[0], np.cos(angle / 2.0))
        self.assertAlmostEqual(quaternion[1], 0.0)
        self.assertAlmostEqual(quaternion[2], np.sin(angle / 2.0))
        self.assertAlmostEqual(quaternion[3], 0.0)

    def test_visible_world_geometry_uses_saved_depth_and_pose(self) -> None:
        depth = np.full((3, 3), 2.0, dtype=np.float32)
        ys = np.asarray([1] * 8, dtype=np.int64)
        xs = np.asarray([1] * 8, dtype=np.int64)

        geometry = MODULE._visible_world_geometry(
            ys,
            xs,
            depth=depth,
            sensor_position_xyz=[1.0, 2.0, 3.0],
            sensor_rotation_matrix=np.eye(3),
            min_depth_m=0.05,
            max_depth_m=6.0,
        )

        self.assertIsNotNone(geometry)
        self.assertEqual(
            geometry["world_visible_center_xyz"],
            [1.0, 2.0, 1.0],
        )
        self.assertEqual(geometry["visible_depth_median"], 2.0)
        self.assertEqual(geometry["visible_projected_points"], 8)

    def test_frame_selection_is_deterministic(self) -> None:
        frames = [{"frame_index": index} for index in range(10)]

        selected = MODULE._select_frames(
            frames,
            start=2,
            end=8,
            stride=3,
            max_frames=2,
        )

        self.assertEqual(
            [frame["frame_index"] for frame in selected],
            [2, 5],
        )


if __name__ == "__main__":
    unittest.main()
