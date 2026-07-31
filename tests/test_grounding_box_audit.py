from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.grounding_box_audit import (
    audit_payloads,
    box_iou,
    classwise_nms,
    interpolated_average_precision,
    main,
)


class GroundingBoxAuditTests(unittest.TestCase):
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
        detections = {
            "detections": [
                self._detection(0, "door", 0.9, [0, 0, 10, 10], 1, [0, 0, 0])
            ]
        }
        semantic_gt = {
            "frames": [
                {
                    "frame_index": 0,
                    "instances": [
                        self._instance(7, "door", [0, 0, 10, 10], [0, 0, 0])
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            detections_path = root / "detections.json"
            gt_path = root / "semantic_gt.json"
            output_path = root / "audit.json"
            detections_path.write_text(json.dumps(detections), encoding="utf-8")
            gt_path.write_text(json.dumps(semantic_gt), encoding="utf-8")

            exit_code = main(
                [
                    "--detections-json",
                    str(detections_path),
                    "--semantic-gt-json",
                    str(gt_path),
                    "--output-json",
                    str(output_path),
                    "--nms-iou",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 0)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(set(written["variants"]), {"baseline"})
            self.assertFalse(written["parameters"]["nms"]["enabled"])
            self.assertEqual(written["variants"]["baseline"]["per_class"]["door"]["ap50"], 1.0)

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
