from __future__ import annotations

import copy
import importlib.metadata
import json
import platform
import tempfile
import unittest
from pathlib import Path

from scripts.grounding_box_audit import (
    _canonical_payload_sha256,
    _detection_contract,
    _hard_negative_category,
    _sha256,
    _validate_embedded_detection_output,
    _validate_semantic_gt_integrity,
    _verify_detector_artifact_files,
    _verify_frozen_source_files,
    audit_payloads,
    box_iou,
    build_audit_report,
    classwise_nms,
    interpolated_average_precision,
    main,
)


class GroundingBoxAuditTests(unittest.TestCase):
    def test_formal_source_verification_rehashes_actual_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_dir = root / "test-scene"
            scene_dir.mkdir()
            for name, content in (
                ("test.basis.glb", b"render"),
                ("test.basis.navmesh", b"navmesh"),
                ("test.glb", b"source"),
                ("test.semantic.glb", b"semantic"),
                ("test.semantic.txt", b"labels"),
            ):
                (scene_dir / name).write_bytes(content)
            dataset_config = root / "dataset.json"
            dataset_config.write_text("{}", encoding="utf-8")
            rgb_path = root / "frame_0000_rgb.jpg"
            rgb_path.write_bytes(b"rgb")
            depth_path = root / "frame_0000_depth.npy"
            depth_path.write_bytes(b"depth")
            metadata_path = root / "frames_metadata.json"
            metadata = {
                "frames": [
                    {
                        "frame_index": 0,
                        "rgb_path": str(rgb_path),
                        "depth_npy": str(depth_path),
                    }
                ]
            }
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            frame_record = {
                "frame_index": 0,
                "rgb_sha256": _sha256(rgb_path),
                "depth_sha256": _sha256(depth_path),
            }
            frame_indices_sha = _canonical_payload_sha256(
                {"frame_indices": [0]}
            )
            input_hashes_sha = _canonical_payload_sha256(
                {"frames": [frame_record]}
            )
            scene_files = []
            for path in sorted(scene_dir.iterdir(), key=lambda item: item.name):
                scene_files.append(
                    {
                        "role": "scene_asset",
                        "path": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
            scene_files.append(
                {
                    "role": "scene_dataset_config",
                    "path": dataset_config.name,
                    "size_bytes": dataset_config.stat().st_size,
                    "sha256": _sha256(dataset_config),
                }
            )
            scene_bundle = {
                "schema_version": 1,
                "scene_id": "test-scene",
                "scene_key": "test",
                "files": scene_files,
            }
            semantic_script = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "grounding_semantic_replay_gt.py"
            )
            semantic_gt = {
                "ground_truth_contract": {
                    "policy_access": False,
                },
                "source": {
                    "generator_script": str(semantic_script),
                    "generator_sha256": _sha256(semantic_script),
                    "frames_metadata": str(metadata_path),
                    "frames_metadata_sha256": _sha256(metadata_path),
                    "scene": str(scene_dir / "test.basis.glb"),
                    "scene_id": "test-scene",
                    "scene_sha256": _sha256(
                        scene_dir / "test.basis.glb"
                    ),
                    "scene_asset_bundle": scene_bundle,
                    "scene_asset_bundle_sha256": (
                        _canonical_payload_sha256(
                            {
                                "schema_version": 1,
                                "files": scene_files,
                            }
                        )
                    ),
                    "scene_dataset_config": str(dataset_config),
                    "scene_dataset_config_sha256": _sha256(dataset_config),
                    "input_frame_hashes_sha256": input_hashes_sha,
                },
                "selection": {
                    "frame_start": 0,
                    "frame_end": None,
                    "frame_stride": 1,
                    "max_frames": None,
                    "selected_num_frames": 1,
                    "selected_frame_indices_sha256": frame_indices_sha,
                },
                "rgb_replay_checks": [
                    {
                        "frame_index": 0,
                        "available": True,
                        "mae": 0.0,
                        "p95_abs_error": 0.0,
                        "max_abs_error": 0.0,
                    }
                ],
                "rgb_replay_integrity": {
                    "enabled": True,
                    "required_checks": 1,
                    "available_checks": 1,
                    "max_mae_allowed": 3.0,
                    "max_p95_abs_error_allowed": 12.0,
                    "observed_max_mae": 0.0,
                    "observed_max_p95_abs_error": 0.0,
                    "passed": True,
                },
                "depth_replay_checks": [
                    {
                        "frame_index": 0,
                        "available": True,
                        "mae_m": 0.0,
                        "p95_abs_error_m": 0.0,
                        "validity_disagreement_ratio": 0.0,
                        "large_error_threshold_m": 0.01,
                        "large_error_ratio": 0.0,
                    }
                ],
                "depth_replay_integrity": {
                    "enabled": True,
                    "required_checks": 1,
                    "available_checks": 1,
                    "max_mae_m_allowed": 0.0001,
                    "max_p95_abs_error_m_allowed": 0.0001,
                    "max_validity_disagreement_ratio_allowed": 0.00001,
                    "large_error_threshold_m": 0.01,
                    "max_large_error_ratio_allowed": 0.0001,
                    "observed_max_mae_m": 0.0,
                    "observed_max_p95_abs_error_m": 0.0,
                    "observed_max_validity_disagreement_ratio": 0.0,
                    "observed_max_large_error_ratio": 0.0,
                    "passed": True,
                },
                "frames": [
                    {
                        "frame_index": 0,
                        "source_rgb_sha256": frame_record["rgb_sha256"],
                        "source_depth_sha256": frame_record[
                            "depth_sha256"
                        ],
                        "instances": [],
                    }
                ],
            }
            implementation_paths = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "grounding_detector_replay.py",
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "m25_groundingdino_export.py",
                Path(__file__).resolve().parents[1]
                / "src"
                / "semantic_task_profile.py",
            )
            implementation_files = [
                {
                    "path": path.relative_to(
                        Path(__file__).resolve().parents[1]
                    ).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in implementation_paths
            ]
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (model_dir / "weights.bin").write_bytes(b"weights")
            model_files = [
                {
                    "path": path.relative_to(model_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(model_dir.iterdir())
            ]
            algorithm_contract = {
                "schema_version": 1,
                "generator_sha256": implementation_files[0]["sha256"],
                "implementation_artifacts": {
                    "files": implementation_files,
                    "manifest_sha256": _canonical_payload_sha256(
                        {"files": implementation_files}
                    ),
                },
                "runtime": {
                    "python": platform.python_version(),
                    "packages": {
                        name: importlib.metadata.version(name)
                        for name in (
                            "numpy",
                            "Pillow",
                            "torch",
                            "transformers",
                        )
                    },
                },
                "labels": ["door"],
                "model_artifacts": {
                    "local_directory": True,
                    "directory": str(model_dir),
                    "files": model_files,
                    "manifest_sha256": _canonical_payload_sha256(
                        {"files": model_files}
                    ),
                },
                "semantic_oracle_access": False,
            }
            run_contract = {
                "frames_metadata_sha256": _sha256(metadata_path),
                "frame_start": 0,
                "frame_end": None,
                "frame_stride": 1,
                "max_frames": None,
                "selected_frames": 1,
                "selected_frame_indices_sha256": frame_indices_sha,
                "input_frame_hashes_sha256": input_hashes_sha,
            }
            detections = {
                "source": {
                    **run_contract,
                    "frames_metadata": str(metadata_path),
                    "input_frame_hashes": [frame_record],
                },
                "algorithm_contract": algorithm_contract,
                "algorithm_sha256": _canonical_payload_sha256(
                    algorithm_contract
                ),
                "run_contract": run_contract,
                "run_contract_sha256": _canonical_payload_sha256(
                    run_contract
                ),
                "detections": [],
            }
            detections_path = root / "detections.json"
            semantic_gt_path = root / "semantic_gt.json"
            detections_path.write_text(
                json.dumps(detections),
                encoding="utf-8",
            )
            semantic_gt_path.write_text(
                json.dumps(semantic_gt),
                encoding="utf-8",
            )

            integrity = _validate_semantic_gt_integrity(semantic_gt)
            source_verification = _verify_frozen_source_files(
                detections,
                semantic_gt,
                semantic_integrity=integrity,
            )
            self.assertTrue(source_verification["passed"])
            self.assertTrue(
                _verify_detector_artifact_files(
                    algorithm_contract
                )["passed"]
            )
            report = build_audit_report(
                detections_path=detections_path,
                semantic_gt_path=semantic_gt_path,
                detector_contract_path=None,
                score_threshold=0.0,
                match_iou=0.5,
                nms_iou=1.0,
                require_source_file_verification=True,
            )
            self.assertTrue(
                report["inputs"]["formal_detection_contract_eligible"]
            )
            from scripts.grounding_audit_compare import (
                _reverify_formal_audit,
            )

            audit_path = root / "audit.json"
            _reverify_formal_audit(report, audit_path=audit_path)
            tampered_report = copy.deepcopy(report)
            tampered_report["coverage"]["detections"]["raw_records"] = 999
            with self.assertRaisesRegex(
                ValueError,
                "does not match live recomputation",
            ):
                _reverify_formal_audit(
                    tampered_report,
                    audit_path=audit_path,
                )

            rgb_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                ValueError,
                "actual RGB-D hashes",
            ):
                build_audit_report(
                    detections_path=detections_path,
                    semantic_gt_path=semantic_gt_path,
                    detector_contract_path=None,
                    score_threshold=0.0,
                    match_iou=0.5,
                    nms_iou=1.0,
                    require_source_file_verification=True,
                )

    def test_hard_negative_taxonomy_covers_hm3d_aliases(self) -> None:
        self.assertEqual(
            _hard_negative_category("kitchen cabinet"),
            "cabinet door",
        )
        self.assertEqual(
            _hard_negative_category("storage cabinet"),
            "cabinet door",
        )
        self.assertEqual(
            _hard_negative_category("bath cupboard"),
            "cabinet door",
        )
        self.assertEqual(
            _hard_negative_category("refrigerator"),
            "refrigerator door",
        )
        self.assertEqual(
            _hard_negative_category("refrigerator cabinet"),
            "refrigerator door",
        )

    def test_box_iou_and_interpolated_ap_are_exact_and_deterministic(self) -> None:
        self.assertAlmostEqual(
            box_iou([0, 0, 10, 10], [5, 0, 15, 10]),
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            interpolated_average_precision([1, 0, 1], [0, 1, 0], 2),
            5.0 / 6.0,
        )
        self.assertEqual(interpolated_average_precision([], [], 2), 0.0)
        self.assertIsNone(interpolated_average_precision([], [], 0))

    def test_audit_reports_detection_track_and_xz_metrics(self) -> None:
        detections = {
            "detections": [
                self._detection(0, "door", 0.95, [0, 0, 10, 10], "t1", [0.1, 0, 0]),
                self._detection(0, "door", 0.90, [0, 0, 10, 10], "dup", [0, 0, 0]),
                self._detection(1, "door", 0.85, [0, 0, 10, 10], "t2", [1.3, 0, 0]),
                self._detection(2, "door", 0.80, [0, 0, 10, 10], "t2", [2.0, 0, 0.4]),
                self._detection(0, "window", 0.88, [20, 0, 30, 10], "w1", [5, 0, 0]),
                self._detection(0, "window", 0.70, [40, 0, 50, 10], "w2", [9, 0, 0]),
            ]
        }
        semantic_gt = {
            "frames": [
                {
                    "frame_index": 0,
                    "instances": [
                        self._instance("d1", "door", [0, 0, 10, 10], [0, 0, 0]),
                        self._instance("w1", "window", [20, 0, 30, 10], [5, 0, 0]),
                    ],
                },
                {
                    "frame_index": 1,
                    "instances": [
                        self._instance("d1", "door", [0, 0, 10, 10], [1, 0, 0])
                    ],
                },
                {
                    "frame_index": 2,
                    "instances": [
                        self._instance("d2", "door", [0, 0, 10, 10], [2, 0, 0])
                    ],
                },
            ]
        }

        result = audit_payloads(
            detections,
            semantic_gt,
            score_threshold=0.75,
            match_iou=0.50,
            nms_iou=0.50,
        )

        baseline = result["variants"]["baseline"]
        door = baseline["per_class"]["door"]
        window = baseline["per_class"]["window"]
        self.assertAlmostEqual(door["ap50"], 5.0 / 6.0)
        self.assertEqual(door["ap75"], door["ap50"])
        operating = door["operating_point"]
        self.assertEqual(operating["predictions"], 4)
        self.assertEqual(operating["tp"], 3)
        self.assertEqual(operating["fp"], 1)
        self.assertEqual(operating["fn"], 0)
        self.assertEqual(operating["duplicate_fp"], 1)
        self.assertEqual(operating["precision"], 0.75)
        self.assertEqual(operating["recall"], 1.0)
        self.assertAlmostEqual(
            operating["fp_per_100_evaluated_frames"], 100.0 / 3.0
        )
        self.assertAlmostEqual(
            operating["duplicate_fp_per_100_evaluated_frames"],
            100.0 / 3.0,
        )
        self.assertEqual(
            operating["tp_iou_distribution"]["distribution"]["median"], 1.0
        )
        self.assertEqual(
            operating["tp_iou_distribution"]["distribution"]["p90"], 1.0
        )
        self.assertEqual(
            operating["physical_instance_recall"]["gt_physical_instances"], 2
        )
        self.assertEqual(
            operating["physical_instance_recall"]["recall"], 1.0
        )
        self.assertEqual(window["operating_point"]["tp"], 1)
        self.assertEqual(window["operating_point"]["fp"], 0)

        track = baseline["track_association"]["door"]
        self.assertTrue(track["available"])
        self.assertEqual(track["fragmented_gt_count"], 1)
        self.assertEqual(track["wrong_merge_track_count"], 1)
        self.assertEqual(track["associated_track_count"], 2)
        self.assertEqual(track["wrong_merge_rate"], 0.5)
        self.assertEqual(track["fragmentation_tracks_per_gt"], 1.5)
        self.assertEqual(track["tracks_per_gt"], 1.5)
        d1 = next(
            item
            for item in track["per_gt_track_count"]
            if item["semantic_id"] == "d1"
        )
        self.assertEqual(d1["online_track_ids"], ["t1", "t2"])

        xz = baseline["xz_error_m"]["door"]["distribution"]
        self.assertEqual(xz["count"], 3)
        for actual, expected in zip(xz["values_m"], [0.1, 0.3, 0.4]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(xz["mean"], 0.8 / 3.0)

        nms = result["variants"]["class_nms"]
        self.assertEqual(
            nms["per_class"]["door"]["operating_point"]["duplicate_fp"], 0
        )
        self.assertEqual(
            nms["per_class"]["door"]["operating_point"]["precision"], 1.0
        )
        self.assertEqual(
            result["parameters"]["nms"]["suppressed_target_detections"], 1
        )
        self.assertFalse(nms["track_association"]["door"]["available"])
        self.assertIn(
            "posthoc",
            nms["track_association"]["door"]["reason"],
        )
        self.assertEqual(
            result["coverage"]["target_frame_coverage"]["overlap_frame_count"], 3
        )

    def test_physical_recall_deduplicates_semantic_id_and_reports_tp_iou(
        self,
    ) -> None:
        detections = {
            "detections": [
                self._detection(
                    0, "door", 0.9, [0, 0, 6, 10], "t1", [0, 0, 0]
                ),
                self._detection(
                    1, "door", 0.8, [0, 0, 8, 10], "t1", [1, 0, 0]
                ),
                self._detection(
                    99, "door", 0.99, [20, 0, 30, 10], "outside", [9, 0, 0]
                ),
            ]
        }
        semantic_gt = {
            "frames": [
                {
                    "frame_index": 0,
                    "instances": [
                        self._instance(
                            "d1", "door", [0, 0, 10, 10], [0, 0, 0]
                        )
                    ],
                },
                {
                    "frame_index": 1,
                    "instances": [
                        self._instance(
                            "d1", "door", [0, 0, 10, 10], [1, 0, 0]
                        )
                    ],
                },
                {
                    "frame_index": 2,
                    "instances": [
                        self._instance(
                            "d2", "door", [0, 0, 10, 10], [2, 0, 0]
                        )
                    ],
                },
            ]
        }

        result = audit_payloads(
            detections, semantic_gt, score_threshold=0.0, match_iou=0.5
        )
        operating = result["variants"]["baseline"]["per_class"]["door"][
            "operating_point"
        ]

        self.assertEqual(operating["predictions"], 2)
        self.assertAlmostEqual(operating["recall"], 2.0 / 3.0)
        physical = operating["physical_instance_recall"]
        self.assertEqual(physical["gt_physical_instances"], 2)
        self.assertEqual(physical["matched_physical_instances"], 1)
        self.assertEqual(physical["recall"], 0.5)
        iou = operating["tp_iou_distribution"]["distribution"]
        self.assertAlmostEqual(iou["median"], 0.7)
        self.assertAlmostEqual(iou["p90"], 0.78)
        self.assertEqual(
            result["coverage"]["detections"][
                "excluded_unannotated_frame_records"
            ],
            1,
        )

    def test_xz_error_prefers_visible_mask_center_over_object_aabb(self) -> None:
        detections = {
            "detections": [
                self._detection(
                    0,
                    "door",
                    0.9,
                    [0, 0, 10, 10],
                    "track",
                    [0.1, 0.0, 0.0],
                )
            ]
        }
        instance = self._instance(
            "door-1",
            "door",
            [0, 0, 10, 10],
            [99.0, 0.0, 99.0],
        )
        instance["world_visible_center_xyz"] = [0.0, 0.0, 0.0]
        instance["visible_depth_median"] = 2.5
        semantic_gt = {
            "frames": [{"frame_index": 0, "instances": [instance]}]
        }

        result = audit_payloads(detections, semantic_gt)
        xz = result["variants"]["baseline"]["xz_error_m"]["door"]

        self.assertAlmostEqual(
            xz["distribution"]["median"],
            0.1,
        )
        self.assertEqual(
            xz["stratification"]["by_semantic_id"][0][
                "usable_position_pairs"
            ],
            1,
        )
        depth_stratum = next(
            item
            for item in xz["stratification"]["by_visible_depth_m"]
            if item["lower_inclusive_m"] == 2.0
        )
        self.assertEqual(depth_stratum["matched_tp_pairs"], 1)
        self.assertAlmostEqual(
            depth_stratum["distribution"]["median"],
            0.1,
        )

    def test_xz_error_does_not_fall_back_to_object_aabb_center(self) -> None:
        detections = {
            "detections": [
                self._detection(
                    0,
                    "door",
                    0.9,
                    [0, 0, 10, 10],
                    "track",
                    [0.1, 0.0, 0.0],
                )
            ]
        }
        semantic_gt = {
            "frames": [
                {
                    "frame_index": 0,
                    "instances": [
                        {
                            "semantic_id": "door-1",
                            "object_id": "object-door-1",
                            "raw_category": "door",
                            "canonical_label": "door",
                            "box": [0, 0, 10, 10],
                            "world_center_xyz": [0.0, 0.0, 0.0],
                            "area_px": 100,
                        }
                    ],
                }
            ]
        }

        result = audit_payloads(detections, semantic_gt)
        xz = result["variants"]["baseline"]["xz_error_m"]["door"]

        self.assertFalse(xz["available"])
        self.assertEqual(xz["usable_position_pairs"], 0)
        self.assertEqual(xz["missing_position_pairs"], 1)
        self.assertEqual(
            xz["missing_reason_counts"],
            {"missing_gt_visible_center": 1},
        )
        self.assertEqual(
            xz["missing_pairs"][0]["semantic_id"],
            "door-1",
        )

    def test_door_fp_attribution_uses_raw_non_target_categories(self) -> None:
        raw_categories = [
            "window",
            "cabinet door",
            "mirror",
            "wall-panel",
            "refrigerator_door",
            "chair",
        ]
        detections = {
            "detections": [
                self._detection(
                    frame_index,
                    "door",
                    0.9,
                    [0, 0, 10, 10],
                    f"track-{frame_index}",
                    [float(frame_index), 0, 0],
                )
                for frame_index in range(len(raw_categories))
            ]
            + [
                self._detection(
                    0,
                    "door",
                    0.8,
                    [0, 0, 10, 10],
                    "duplicate-window-fp",
                    [0, 0, 0],
                )
            ]
        }
        semantic_gt = {
            "frames": [
                {
                    "frame_index": frame_index,
                    "instances": [
                        self._raw_non_target_instance(
                            f"negative-{frame_index}", raw_category
                        )
                    ],
                }
                for frame_index, raw_category in enumerate(raw_categories)
            ]
        }

        result = audit_payloads(
            detections, semantic_gt, match_iou=0.5, nms_iou=0.5
        )
        baseline = result["variants"]["baseline"]
        operating = baseline["per_class"]["door"]["operating_point"]
        hard_negative = baseline["hard_negative_door_fp"]

        self.assertEqual(
            baseline["per_class"]["window"]["ground_truth_instances"], 0
        )
        self.assertEqual(operating["fp"], 7)
        self.assertAlmostEqual(
            operating["fp_per_100_evaluated_frames"], 700.0 / 6.0
        )
        self.assertEqual(hard_negative["recognized_hard_negative_gt_instances"], 5)
        self.assertEqual(hard_negative["attributed_hard_negative_fp"], 6)
        self.assertEqual(hard_negative["unattributed_door_fp"], 1)
        self.assertAlmostEqual(
            hard_negative["hard_negative_fp_per_100_evaluated_frames"],
            100.0,
        )
        self.assertEqual(
            hard_negative["hard_negative_fp_per_100_frames"],
            hard_negative["hard_negative_fp_per_100_evaluated_frames"],
        )
        for category in (
            "cabinet door",
            "mirror",
            "wall panel",
            "refrigerator door",
        ):
            self.assertEqual(
                hard_negative["attributed_fp_by_category"][category], 1
            )
        self.assertEqual(
            hard_negative["attributed_fp_by_category"]["window"], 2
        )
        nms_hard_negative = result["variants"]["class_nms"][
            "hard_negative_door_fp"
        ]
        self.assertEqual(nms_hard_negative["attributed_hard_negative_fp"], 5)
        self.assertAlmostEqual(
            nms_hard_negative["hard_negative_fp_per_100_frames"],
            500.0 / 6.0,
        )
        self.assertTrue(
            result["parameters"]["hard_negative_policy"][
                "raw_category_never_creates_target_gt"
            ]
        )

    def test_nms_is_per_frame_and_canonical_class(self) -> None:
        detections = [
            {
                "_source_index": 0,
                "frame_index": 0,
                "canonical_label": "door",
                "score": 0.9,
                "box": [0, 0, 10, 10],
            },
            {
                "_source_index": 1,
                "frame_index": 0,
                "canonical_label": "window",
                "score": 0.8,
                "box": [0, 0, 10, 10],
            },
            {
                "_source_index": 2,
                "frame_index": 1,
                "canonical_label": "door",
                "score": 0.7,
                "box": [0, 0, 10, 10],
            },
            {
                "_source_index": 3,
                "frame_index": 0,
                "canonical_label": "door",
                "score": 0.6,
                "box": [1, 1, 9, 9],
            },
        ]
        kept, suppressed = classwise_nms(detections, 0.5)
        self.assertEqual([item["_source_index"] for item in kept], [0, 1, 2])
        self.assertEqual([item["_source_index"] for item in suppressed], [3])

    def test_equal_score_order_and_serialized_result_are_deterministic(
        self,
    ) -> None:
        detections = {
            "detections": [
                self._detection(
                    0, "door", 0.8, [20, 0, 30, 10], "false", [5, 0, 0]
                ),
                self._detection(
                    0, "door", 0.8, [0, 0, 10, 10], "true", [0, 0, 0]
                ),
            ]
        }
        semantic_gt = {
            "frames": [
                {
                    "frame_index": 0,
                    "instances": [
                        self._instance(
                            "d1", "door", [0, 0, 10, 10], [0, 0, 0]
                        )
                    ],
                }
            ]
        }

        first = audit_payloads(detections, semantic_gt)
        second = audit_payloads(detections, semantic_gt)

        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )
        self.assertAlmostEqual(
            first["variants"]["baseline"]["per_class"]["door"]["ap50"],
            0.5,
        )

    def test_missing_truth_and_association_fields_are_explicitly_unavailable(
        self,
    ) -> None:
        detections = {
            "detections": [
                {
                    "frame_index": 0,
                    "label": "door",
                    "score": 0.9,
                    "box": [0, 0, 10, 10],
                }
            ]
        }
        semantic_gt = {
            "frames": [
                {
                    "frame_index": 0,
                    "instances": [
                        {
                            "semantic_id": "d1",
                            "object_id": "object-d1",
                            "raw_category": "door",
                            "canonical_label": "door",
                            "box": [0, 0, 10, 10],
                            "area_px": 100,
                        },
                        "malformed-instance",
                    ],
                }
            ]
        }

        result = audit_payloads(detections, semantic_gt)
        baseline = result["variants"]["baseline"]

        self.assertIsNone(baseline["per_class"]["window"]["ap50"])
        self.assertIsNone(
            baseline["per_class"]["window"]["operating_point"]["precision"]
        )
        self.assertFalse(baseline["track_association"]["door"]["available"])
        self.assertFalse(baseline["xz_error_m"]["door"]["available"])
        unavailable = {
            item["metric"] for item in result["unavailable_metrics"]
        }
        self.assertIn("variants.baseline.per_class.window.ap50", unavailable)
        self.assertIn(
            "variants.baseline.track_association.door", unavailable
        )
        self.assertIn("variants.baseline.xz_error_m.door", unavailable)
        self.assertEqual(
            result["parameters"]["ground_truth_policy"],
            "semantic_gt instances are the only truth source; "
            "VLM verdicts are never read",
        )
        self.assertEqual(result["coverage"]["semantic_gt"]["raw_instances"], 2)
        self.assertEqual(
            result["coverage"]["semantic_gt"]["invalid_record_reasons"][
                "record_not_object"
            ],
            1,
        )

    def test_cli_writes_strict_json_and_nms_can_be_disabled(self) -> None:
        frame_indices_sha = _canonical_payload_sha256(
            {"frame_indices": [0]}
        )
        input_hashes_sha = _canonical_payload_sha256(
            {
                "frames": [
                    {
                        "frame_index": 0,
                        "rgb_sha256": "a" * 64,
                        "depth_sha256": "b" * 64,
                    }
                ]
            }
        )
        scene_asset_files = [
            {
                "role": "scene_asset",
                "path": "test.basis.glb",
                "size_bytes": 1,
                "sha256": "4" * 64,
            },
            {
                "role": "scene_asset",
                "path": "test.basis.navmesh",
                "size_bytes": 1,
                "sha256": "5" * 64,
            },
            {
                "role": "scene_asset",
                "path": "test.semantic.glb",
                "size_bytes": 1,
                "sha256": "6" * 64,
            },
            {
                "role": "scene_asset",
                "path": "test.semantic.txt",
                "size_bytes": 1,
                "sha256": "7" * 64,
            },
            {
                "role": "scene_dataset_config",
                "path": "dataset.json",
                "size_bytes": 1,
                "sha256": "8" * 64,
            },
        ]
        scene_asset_bundle = {
            "schema_version": 1,
            "scene_id": "test-scene",
            "scene_key": "test",
            "files": scene_asset_files,
        }
        detections = {
            "detections": [
                self._detection(0, "door", 0.9, [0, 0, 10, 10], 1, [0, 0, 0])
            ]
        }
        semantic_gt = {
            "ground_truth_contract": {
                "policy_access": False,
            },
            "source": {
                "generator_sha256": "c" * 64,
                "scene": "/tmp/test-scene/test.basis.glb",
                "scene_id": "test-scene",
                "scene_sha256": "d" * 64,
                "scene_asset_bundle": scene_asset_bundle,
                "scene_asset_bundle_sha256": (
                    _canonical_payload_sha256(
                        {
                            "schema_version": 1,
                            "files": scene_asset_files,
                        }
                    )
                ),
                "frames_metadata_sha256": "e" * 64,
                "input_frame_hashes_sha256": input_hashes_sha,
            },
            "selection": {
                "selected_num_frames": 1,
                "selected_frame_indices_sha256": frame_indices_sha,
            },
            "rgb_replay_checks": [
                {
                    "frame_index": 0,
                    "available": True,
                    "mae": 0.0,
                    "p95_abs_error": 0.0,
                    "max_abs_error": 0.0,
                }
            ],
            "rgb_replay_integrity": {
                "enabled": True,
                "passed": True,
                "required_checks": 1,
                "available_checks": 1,
                "max_mae_allowed": 3.0,
                "max_p95_abs_error_allowed": 12.0,
                "observed_max_mae": 0.0,
                "observed_max_p95_abs_error": 0.0,
            },
            "depth_replay_checks": [
                {
                    "frame_index": 0,
                    "available": True,
                    "mae_m": 0.0,
                    "p95_abs_error_m": 0.0,
                    "validity_disagreement_ratio": 0.0,
                    "large_error_threshold_m": 0.01,
                    "large_error_ratio": 0.0,
                }
            ],
            "depth_replay_integrity": {
                "enabled": True,
                "passed": True,
                "required_checks": 1,
                "available_checks": 1,
                "max_mae_m_allowed": 0.0001,
                "max_p95_abs_error_m_allowed": 0.0001,
                "max_validity_disagreement_ratio_allowed": 0.00001,
                "large_error_threshold_m": 0.01,
                "max_large_error_ratio_allowed": 0.0001,
                "observed_max_mae_m": 0.0,
                "observed_max_p95_abs_error_m": 0.0,
                "observed_max_validity_disagreement_ratio": 0.0,
                "observed_max_large_error_ratio": 0.0,
            },
            "frames": [
                {
                    "frame_index": 0,
                    "source_rgb_sha256": "a" * 64,
                    "source_depth_sha256": "b" * 64,
                    "instances": [
                        self._instance(7, "door", [0, 0, 10, 10], [0, 0, 0])
                    ],
                }
            ]
        }
        integrity = _validate_semantic_gt_integrity(semantic_gt)
        self.assertEqual(integrity["frame_count"], 1)
        for label, mutate, expected in (
            (
                "disabled",
                lambda value: value["rgb_replay_integrity"].update(
                    {"enabled": False}
                ),
                "was not enabled",
            ),
            (
                "missing frame",
                lambda value: value.update({"depth_replay_checks": []}),
                "not full-frame",
            ),
            (
                "wrong frame",
                lambda value: value["rgb_replay_checks"][0].update(
                    {"frame_index": 1}
                ),
                "do not match selected frame order",
            ),
        ):
            with self.subTest(integrity_case=label):
                invalid = copy.deepcopy(semantic_gt)
                mutate(invalid)
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_semantic_gt_integrity(invalid)
        coordinated_fake = copy.deepcopy(semantic_gt)
        coordinated_fake["rgb_replay_checks"][0]["mae"] = 999.0
        coordinated_fake["rgb_replay_integrity"][
            "observed_max_mae"
        ] = 999.0
        coordinated_fake["depth_replay_checks"][0]["mae_m"] = 999.0
        coordinated_fake["depth_replay_integrity"][
            "observed_max_mae_m"
        ] = 999.0
        with self.assertRaisesRegex(
            ValueError,
            "passed flag does not match recomputed metrics",
        ):
            _validate_semantic_gt_integrity(coordinated_fake)
        raised_threshold = copy.deepcopy(semantic_gt)
        raised_threshold["rgb_replay_checks"][0]["mae"] = 999.0
        raised_threshold["rgb_replay_integrity"][
            "observed_max_mae"
        ] = 999.0
        raised_threshold["rgb_replay_integrity"][
            "max_mae_allowed"
        ] = 1000.0
        with self.assertRaisesRegex(
            ValueError,
            "frozen threshold max_mae_allowed",
        ):
            _validate_semantic_gt_integrity(raised_threshold)
        coordinated_scene_alias = copy.deepcopy(semantic_gt)
        coordinated_scene_alias["source"]["scene_id"] = "fake-scene"
        coordinated_scene_alias["source"]["scene_asset_bundle"][
            "scene_id"
        ] = "fake-scene"
        with self.assertRaisesRegex(
            ValueError,
            "scene identity does not match",
        ):
            _validate_semantic_gt_integrity(coordinated_scene_alias)

        embedded = copy.deepcopy(detections)
        implementation_files = [
            {
                "path": "scripts/grounding_detector_replay.py",
                "size_bytes": 1,
                "sha256": "f" * 64,
            },
            {
                "path": "scripts/m25_groundingdino_export.py",
                "size_bytes": 2,
                "sha256": "8" * 64,
            },
            {
                "path": "src/semantic_task_profile.py",
                "size_bytes": 3,
                "sha256": "7" * 64,
            },
        ]
        embedded_algorithm = {
            "schema_version": 1,
            "generator_sha256": "f" * 64,
            "implementation_artifacts": {
                "files": implementation_files,
                "manifest_sha256": _canonical_payload_sha256(
                    {"files": implementation_files}
                ),
            },
            "runtime": {
                "python": "3.10.0",
                "packages": {
                    "numpy": "1.0",
                    "Pillow": "1.0",
                    "torch": "1.0",
                    "transformers": "1.0",
                },
            },
            "labels": ["door"],
            "model_artifacts": {
                "local_directory": False,
                "identifier": "test/model",
            },
            "semantic_oracle_access": False,
        }
        embedded_run = {
            "frames_metadata_sha256": "e" * 64,
            "frame_start": 0,
            "frame_end": None,
            "frame_stride": 1,
            "max_frames": None,
            "selected_frames": 1,
            "selected_frame_indices_sha256": frame_indices_sha,
            "input_frame_hashes_sha256": input_hashes_sha,
        }
        embedded_source = {
            **embedded_run,
            "input_frame_hashes": [
                {
                    "frame_index": 0,
                    "rgb_sha256": "a" * 64,
                    "depth_sha256": "b" * 64,
                }
            ],
        }
        embedded.update(
            {
                "source": embedded_source,
                "algorithm_contract": embedded_algorithm,
                "algorithm_sha256": _canonical_payload_sha256(
                    embedded_algorithm
                ),
                "run_contract": embedded_run,
                "run_contract_sha256": _canonical_payload_sha256(
                    embedded_run
                ),
            }
        )
        _, embedded_provenance = _detection_contract(
            embedded,
            external_contract_path=None,
            semantic_integrity=integrity,
        )
        self.assertTrue(embedded_provenance["formal_eligible"])
        coordinated_stride = copy.deepcopy(embedded)
        coordinated_stride["detections"][0]["frame_index"] = 1
        coordinated_stride["source"]["input_frame_hashes"][0][
            "frame_index"
        ] = 1
        coordinated_stride["source"]["frame_stride"] = 2
        coordinated_stride["run_contract"]["frame_stride"] = 2
        stride_indices_sha = _canonical_payload_sha256(
            {"frame_indices": [1]}
        )
        stride_inputs_sha = _canonical_payload_sha256(
            {
                "frames": coordinated_stride["source"][
                    "input_frame_hashes"
                ]
            }
        )
        for contract in (
            coordinated_stride["source"],
            coordinated_stride["run_contract"],
        ):
            contract["selected_frame_indices_sha256"] = stride_indices_sha
            contract["input_frame_hashes_sha256"] = stride_inputs_sha
        with self.assertRaisesRegex(
            ValueError,
            "violate the declared selection",
        ):
            _validate_embedded_detection_output(
                coordinated_stride,
                algorithm_contract=coordinated_stride[
                    "algorithm_contract"
                ],
                run_contract=coordinated_stride["run_contract"],
            )

        for label, mutate, expected in (
            (
                "oracle access",
                lambda value: value["algorithm_contract"].update(
                    {"semantic_oracle_access": True}
                ),
                "semantic_oracle_access=false",
            ),
            (
                "input hash record",
                lambda value: value["source"]["input_frame_hashes"][0].update(
                    {"rgb_sha256": "9" * 64}
                ),
                "input frame hash manifest does not match",
            ),
            (
                "implementation dependency",
                lambda value: value["algorithm_contract"][
                    "implementation_artifacts"
                ]["files"][1].update({"sha256": "6" * 64}),
                "implementation artifact manifest hash mismatch",
            ),
            (
                "selection contract",
                lambda value: value["run_contract"].update(
                    {"frame_stride": 2}
                ),
                "source.frame_stride does not match run_contract",
            ),
            (
                "outside frame",
                lambda value: value["detections"][0].update(
                    {"frame_index": 1}
                ),
                "outside the frozen selection",
            ),
        ):
            with self.subTest(embedded_case=label):
                invalid = copy.deepcopy(embedded)
                mutate(invalid)
                invalid["algorithm_sha256"] = _canonical_payload_sha256(
                    invalid["algorithm_contract"]
                )
                invalid["run_contract_sha256"] = (
                    _canonical_payload_sha256(invalid["run_contract"])
                )
                with self.assertRaisesRegex(ValueError, expected):
                    _detection_contract(
                        invalid,
                        external_contract_path=None,
                        semantic_integrity=integrity,
                    )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            detections_path = root / "detections.json"
            gt_path = root / "semantic_gt.json"
            output_path = root / "audit.json"
            contract_path = root / "detector_contract.json"
            detector_contract = {
                "algorithm_contract": {
                    "backend": "test-detector",
                },
                "run_contract": {
                    "frames_metadata_sha256": "e" * 64,
                    "selected_frame_indices_sha256": frame_indices_sha,
                    "input_frame_hashes_sha256": input_hashes_sha,
                    "selected_frames": 1,
                }
            }
            detections_path.write_text(json.dumps(detections), encoding="utf-8")
            gt_path.write_text(json.dumps(semantic_gt), encoding="utf-8")
            contract_path.write_text(
                json.dumps(detector_contract),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--detections-json",
                    str(detections_path),
                    "--semantic-gt-json",
                    str(gt_path),
                    "--output-json",
                    str(output_path),
                    "--detection-algorithm-contract-json",
                    str(contract_path),
                    "--nms-iou",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 0)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(set(written["variants"]), {"baseline"})
            self.assertFalse(written["parameters"]["nms"]["enabled"])
            self.assertEqual(written["variants"]["baseline"]["per_class"]["door"]["ap50"], 1.0)
            self.assertEqual(len(written["inputs"]["detections_sha256"]), 64)
            self.assertEqual(len(written["inputs"]["semantic_gt_sha256"]), 64)
            self.assertEqual(len(written["inputs"]["evaluator_sha256"]), 64)

    @staticmethod
    def _detection(
        frame_index: int,
        label: str,
        score: float,
        box: list[int],
        track_id: object,
        position: list[float],
    ) -> dict[str, object]:
        return {
            "frame_index": frame_index,
            "canonical_label": label,
            "score": score,
            "box": box,
            "position_3d": position,
            "online_track_id": track_id,
        }

    @staticmethod
    def _instance(
        semantic_id: object,
        label: str,
        box: list[int],
        center: list[float],
    ) -> dict[str, object]:
        return {
            "semantic_id": semantic_id,
            "object_id": f"object-{semantic_id}",
            "raw_category": label,
            "canonical_label": label,
            "box": box,
            "world_center_xyz": center,
            "world_visible_center_xyz": center,
            "area_px": (box[2] - box[0]) * (box[3] - box[1]),
        }

    @staticmethod
    def _raw_non_target_instance(
        semantic_id: object,
        raw_category: str,
    ) -> dict[str, object]:
        return {
            "semantic_id": semantic_id,
            "object_id": f"object-{semantic_id}",
            "raw_category": raw_category,
            "canonical_label": None,
            "box": [0, 0, 10, 10],
            "world_center_xyz": [0, 0, 0],
            "area_px": 100,
        }


if __name__ == "__main__":
    unittest.main()
