from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.grounding_audit_compare import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MANIFEST_SCHEMA_VERSION,
    PRIMARY_METRIC,
    compare_manifest_payload,
    main,
)


EVALUATION_CONTRACT = {"contract_version": "test_v1"}
PARAMETERS_HASH = hashlib.sha256(
    json.dumps(
        EVALUATION_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
GROUND_TRUTH_HASH = "b" * 64


class GroundingAuditCompareTests(unittest.TestCase):
    def test_positive_paired_comparison_extracts_all_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = self._three_cluster_manifest(root)

            result = compare_manifest_payload(manifest, manifest_dir=root)

            self.assertEqual(result["cluster_count"], 3)
            self.assertEqual(result["bootstrap"]["replicates"], 10_000)
            self.assertEqual(result["bootstrap"]["seed"], 20_260_731)
            self.assertFalse(result["bootstrap"]["frame_level_resampling"])
            first = result["clusters"][0]["metrics"][PRIMARY_METRIC]
            self.assertEqual(first["baseline"], 10.0)
            self.assertEqual(first["candidate"], 5.0)
            self.assertEqual(first["paired_delta"], -5.0)
            self.assertEqual(first["relative_reduction"]["value"], 0.5)

            primary = result["comparisons"][PRIMARY_METRIC]
            self.assertTrue(primary["available"])
            self.assertEqual(primary["macro_average"]["baseline"], 20.0)
            self.assertEqual(primary["macro_average"]["candidate"], 10.0)
            self.assertEqual(primary["macro_average"]["paired_delta"], -10.0)
            self.assertEqual(
                primary["relative_reduction"]["macro_average"],
                0.5,
            )

            expected_metrics = {
                PRIMARY_METRIC,
                "door_recall",
                "door_physical_instance_recall",
                "door_tp_median_iou",
                "door_fp_per_100_frames",
                "door_duplicate_fp_per_100_frames",
                "door_xz_median_m",
                "door_xz_p90_m",
            }
            self.assertEqual(set(result["comparisons"]), expected_metrics)
            self.assertTrue(
                all(
                    result["comparisons"][name]["available"]
                    for name in expected_metrics
                )
            )
            provenance = result["clusters"][0]
            self.assertEqual(
                provenance["parameters_sha256"],
                PARAMETERS_HASH,
            )
            self.assertEqual(
                provenance["ground_truth_sha256"],
                GROUND_TRUTH_HASH,
            )
            self.assertRegex(
                provenance["baseline"]["audit_json_sha256"],
                r"^[0-9a-f]{64}$",
            )

    def test_rejects_duplicate_incomplete_hash_and_variant_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline_path = self._write_audit(
                root / "baseline.json",
                {"baseline": self._variant()},
            )
            candidate_path = self._write_audit(
                root / "candidate.json",
                {"candidate": self._variant()},
            )
            manifest = self._manifest(
                [
                    self._cluster(
                        "scene/trajectory",
                        baseline_path.name,
                        candidate_path.name,
                    )
                ]
            )

            cases: list[tuple[str, dict, str]] = []

            duplicate = copy.deepcopy(manifest)
            duplicate["clusters"].append(copy.deepcopy(duplicate["clusters"][0]))
            cases.append(("duplicate", duplicate, "duplicate cluster_id"))

            incomplete = copy.deepcopy(manifest)
            del incomplete["clusters"][0]["candidate"]["audit_json"]
            cases.append(("incomplete", incomplete, "audit_json"))

            parameters_mismatch = copy.deepcopy(manifest)
            parameters_mismatch["clusters"][0]["candidate"][
                "parameters_sha256"
            ] = "c" * 64
            cases.append(
                (
                    "parameters mismatch",
                    parameters_mismatch,
                    "parameters_sha256",
                )
            )

            ground_truth_mismatch = copy.deepcopy(manifest)
            ground_truth_mismatch["clusters"][0]["candidate"][
                "ground_truth_sha256"
            ] = "d" * 64
            cases.append(
                (
                    "ground truth mismatch",
                    ground_truth_mismatch,
                    "ground_truth_sha256",
                )
            )

            invalid_hash = copy.deepcopy(manifest)
            invalid_hash["clusters"][0]["baseline"][
                "ground_truth_sha256"
            ] = "not-a-sha256"
            cases.append(
                ("invalid hash", invalid_hash, "64 hex characters")
            )

            missing_variant = copy.deepcopy(manifest)
            missing_variant["clusters"][0]["candidate"]["variant"] = "absent"
            cases.append(
                ("missing variant", missing_variant, "does not exist")
            )

            for label, invalid_manifest, message in cases:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, message):
                        compare_manifest_payload(
                            invalid_manifest,
                            manifest_dir=root,
                        )

    def test_rejects_audit_contract_and_ground_truth_provenance_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline_path = self._write_audit(
                root / "baseline.json",
                {"baseline": self._variant()},
            )
            candidate_path = self._write_audit(
                root / "candidate.json",
                {"candidate": self._variant()},
            )
            manifest = self._manifest(
                [
                    self._cluster(
                        "scene/trajectory",
                        baseline_path.name,
                        candidate_path.name,
                    )
                ]
            )

            tampered_contract = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
            tampered_contract["evaluation_contract"]["contract_version"] = (
                "tampered"
            )
            baseline_path.write_text(
                json.dumps(tampered_contract, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "does not match its evaluation_contract",
            ):
                compare_manifest_payload(manifest, manifest_dir=root)

            self._write_audit(
                baseline_path,
                {"baseline": self._variant()},
            )
            tampered_gt = json.loads(
                candidate_path.read_text(encoding="utf-8")
            )
            tampered_gt["inputs"]["semantic_gt_sha256"] = "c" * 64
            candidate_path.write_text(
                json.dumps(tampered_gt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "manifest ground_truth_sha256 does not match audit",
            ):
                compare_manifest_payload(manifest, manifest_dir=root)

    def test_output_is_byte_deterministic_and_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = self._three_cluster_manifest(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            first_path = root / "first.json"
            second_path = root / "second.json"

            self.assertEqual(
                main(
                    [
                        "--manifest-json",
                        str(manifest_path),
                        "--output-json",
                        str(first_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "--manifest-json",
                        str(manifest_path),
                        "--output-json",
                        str(second_path),
                    ]
                ),
                0,
            )

            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            parsed = json.loads(
                first_path.read_text(encoding="utf-8"),
                parse_constant=lambda value: self.fail(
                    f"non-standard JSON constant: {value}"
                ),
            )
            self.assertEqual(parsed["bootstrap"]["replicates"], BOOTSTRAP_REPLICATES)
            self.assertEqual(parsed["bootstrap"]["seed"], BOOTSTRAP_SEED)

    def test_missing_metric_is_explicitly_unavailable_without_partial_macro(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = self._variant()
            candidate = self._variant()
            candidate["xz_error_m"]["door"] = {
                "available": False,
                "reason": "no usable position pairs",
                "distribution": None,
            }
            baseline_path = self._write_audit(
                root / "baseline.json",
                {"baseline": baseline},
            )
            candidate_path = self._write_audit(
                root / "candidate.json",
                {"candidate": candidate},
            )
            manifest = self._manifest(
                [
                    self._cluster(
                        "scene/trajectory",
                        baseline_path.name,
                        candidate_path.name,
                    )
                ]
            )

            result = compare_manifest_payload(manifest, manifest_dir=root)

            cluster_metric = result["clusters"][0]["metrics"][
                "door_xz_median_m"
            ]
            self.assertFalse(cluster_metric["available"])
            self.assertIsNone(cluster_metric["candidate"])
            self.assertIsNone(cluster_metric["paired_delta"])
            self.assertIn("no usable position pairs", cluster_metric["reason"])
            aggregate = result["comparisons"]["door_xz_median_m"]
            self.assertFalse(aggregate["available"])
            self.assertIsNone(aggregate["macro_average"])
            self.assertIsNone(aggregate["paired_delta_bootstrap_95_ci"])
            self.assertEqual(
                aggregate["unavailable_cluster_ids"],
                ["scene/trajectory"],
            )
            self.assertTrue(
                any(
                    item["metric"] == "door_xz_median_m"
                    for item in result["unavailable_metrics"]
                )
            )
            self.assertTrue(
                result["comparisons"][PRIMARY_METRIC]["available"]
            )

    def test_bootstrap_ci_and_delta_signs_follow_registered_definitions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            clusters = []
            for index in range(3):
                baseline = self._variant(
                    hard_negative_fp=20.0,
                    recall=0.60,
                    xz_median=0.50,
                    xz_p90=0.90,
                )
                candidate = self._variant(
                    hard_negative_fp=15.0,
                    recall=0.70,
                    xz_median=0.40,
                    xz_p90=0.70,
                )
                baseline_path = self._write_audit(
                    root / f"baseline-{index}.json",
                    {"baseline": baseline},
                )
                candidate_path = self._write_audit(
                    root / f"candidate-{index}.json",
                    {"candidate": candidate},
                )
                clusters.append(
                    self._cluster(
                        f"scene-{index}/trajectory",
                        baseline_path.name,
                        candidate_path.name,
                    )
                )

            result = compare_manifest_payload(
                self._manifest(clusters),
                manifest_dir=root,
            )

            primary = result["comparisons"][PRIMARY_METRIC]
            self.assertEqual(primary["macro_average"]["paired_delta"], -5.0)
            self.assertEqual(
                primary["paired_delta_bootstrap_95_ci"],
                {"lower": -5.0, "upper": -5.0},
            )
            self.assertEqual(
                primary["relative_reduction"]["macro_average"],
                0.25,
            )
            self.assertEqual(
                primary["relative_reduction"]["bootstrap_95_ci"],
                {"lower": 0.25, "upper": 0.25},
            )
            recall = result["comparisons"]["door_recall"]
            self.assertAlmostEqual(
                recall["macro_average"]["paired_delta"],
                0.10,
            )
            self.assertGreater(
                recall["paired_delta_bootstrap_95_ci"]["lower"],
                0.0,
            )
            xz = result["comparisons"]["door_xz_median_m"]
            self.assertAlmostEqual(
                xz["macro_average"]["paired_delta"],
                -0.10,
            )
            self.assertLess(
                xz["paired_delta_bootstrap_95_ci"]["upper"],
                0.0,
            )

    def test_zero_primary_baseline_keeps_delta_but_marks_ratio_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline_path = self._write_audit(
                root / "baseline.json",
                {"baseline": self._variant(hard_negative_fp=0.0)},
            )
            candidate_path = self._write_audit(
                root / "candidate.json",
                {"candidate": self._variant(hard_negative_fp=0.0)},
            )
            manifest = self._manifest(
                [
                    self._cluster(
                        "scene/trajectory",
                        baseline_path.name,
                        candidate_path.name,
                    )
                ]
            )

            result = compare_manifest_payload(manifest, manifest_dir=root)

            primary = result["comparisons"][PRIMARY_METRIC]
            self.assertTrue(primary["available"])
            self.assertEqual(primary["macro_average"]["paired_delta"], 0.0)
            self.assertFalse(primary["relative_reduction"]["available"])
            self.assertIn(
                "scene/trajectory",
                primary["relative_reduction"]["unavailable_cluster_ids"],
            )

    @staticmethod
    def _manifest(clusters: list[dict]) -> dict:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "clusters": clusters,
        }

    @staticmethod
    def _cluster(
        cluster_id: str,
        baseline_path: str,
        candidate_path: str,
    ) -> dict:
        return {
            "cluster_id": cluster_id,
            "baseline": {
                "audit_json": baseline_path,
                "variant": "baseline",
                "parameters_sha256": PARAMETERS_HASH,
                "ground_truth_sha256": GROUND_TRUTH_HASH,
            },
            "candidate": {
                "audit_json": candidate_path,
                "variant": "candidate",
                "parameters_sha256": PARAMETERS_HASH,
                "ground_truth_sha256": GROUND_TRUTH_HASH,
            },
        }

    def _three_cluster_manifest(self, root: Path) -> dict:
        clusters = []
        for index, hard_negative_fp in enumerate((10.0, 20.0, 30.0)):
            baseline = self._variant(
                hard_negative_fp=hard_negative_fp,
                recall=0.60 + index * 0.10,
                physical_recall=0.50 + index * 0.10,
                tp_median_iou=0.60 + index * 0.05,
                door_fp=12.0 + index * 10.0,
                duplicate_fp=3.0 + index,
                xz_median=0.40 + index * 0.10,
                xz_p90=0.80 + index * 0.10,
            )
            candidate = self._variant(
                hard_negative_fp=hard_negative_fp / 2.0,
                recall=0.70 + index * 0.10,
                physical_recall=0.60 + index * 0.10,
                tp_median_iou=0.65 + index * 0.05,
                door_fp=6.0 + index * 5.0,
                duplicate_fp=2.0 + index * 0.5,
                xz_median=0.30 + index * 0.10,
                xz_p90=0.70 + index * 0.10,
            )
            baseline_path = self._write_audit(
                root / f"baseline-{index}.json",
                {"baseline": baseline},
            )
            candidate_path = self._write_audit(
                root / f"candidate-{index}.json",
                {"candidate": candidate},
            )
            clusters.append(
                self._cluster(
                    f"scene-{index}/trajectory-{index}",
                    baseline_path.name,
                    candidate_path.name,
                )
            )
        return self._manifest(clusters)

    @staticmethod
    def _write_audit(path: Path, variants: dict) -> Path:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "grounding_box_audit_v1",
                    "evaluation_contract": EVALUATION_CONTRACT,
                    "evaluation_parameters_sha256": PARAMETERS_HASH,
                    "inputs": {
                        "semantic_gt_sha256": GROUND_TRUTH_HASH,
                    },
                    "variants": variants,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _variant(
        *,
        hard_negative_fp: float = 10.0,
        recall: float = 0.75,
        physical_recall: float = 0.80,
        tp_median_iou: float = 0.65,
        door_fp: float = 12.0,
        duplicate_fp: float = 2.0,
        xz_median: float = 0.30,
        xz_p90: float = 0.60,
    ) -> dict:
        return {
            "hard_negative_door_fp": {
                "hard_negative_fp_per_100_frames": hard_negative_fp,
            },
            "per_class": {
                "door": {
                    "operating_point": {
                        "recall": recall,
                        "physical_instance_recall": {
                            "available": True,
                            "recall": physical_recall,
                        },
                        "tp_iou_distribution": {
                            "available": True,
                            "distribution": {
                                "median": tp_median_iou,
                            },
                        },
                        "fp_per_100_evaluated_frames": door_fp,
                        "duplicate_fp_per_100_evaluated_frames": duplicate_fp,
                    }
                }
            },
            "xz_error_m": {
                "door": {
                    "available": True,
                    "distribution": {
                        "median": xz_median,
                        "p90": xz_p90,
                    },
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
