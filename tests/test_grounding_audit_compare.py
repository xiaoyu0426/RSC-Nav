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
    FORMAL_EVIDENCE_CONTRACT_SHA256,
    MANIFEST_SCHEMA_VERSION,
    PRIMARY_METRIC,
    _load_evidence_contract,
    _validate_formal_cluster_minimum,
    compare_manifest_payload,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
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
DETECTIONS_HASH = "c" * 64
GROUND_TRUTH_GENERATOR_HASH = "d" * 64
SCENE_HASH = "e" * 64
FRAMES_METADATA_HASH = "f" * 64
SELECTED_FRAME_INDICES_HASH = "1" * 64
INPUT_FRAME_HASHES_HASH = "2" * 64
DETECTION_RUN_CONTRACT_HASH = "3" * 64
SEMANTIC_INTEGRITY_CONTRACT = {
    "contract": "full_frame_rgb_depth_replay_v1",
    "frame_count": 1,
}
SEMANTIC_INTEGRITY_HASH = hashlib.sha256(
    json.dumps(
        SEMANTIC_INTEGRITY_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GroundingAuditCompareTests(unittest.TestCase):
    def test_formal_evidence_contract_blocks_manifest_whitelist_expansion(
        self,
    ) -> None:
        contract_dir = ROOT / "audit" / "grounding_box_v2"
        manifest = {
            "evidence_contract_json": "evidence_contract_amendment.json",
            "evidence_contract_sha256": (
                FORMAL_EVIDENCE_CONTRACT_SHA256
            ),
        }
        loaded = _load_evidence_contract(
            manifest,
            manifest_dir=contract_dir,
            formal_acceptance_eligible=True,
            manifest_allowed_paths=["detection_algorithm.labels"],
        )
        self.assertIsNotNone(loaded)

        with self.assertRaisesRegex(
            ValueError,
            "do not match the frozen evidence contract",
        ):
            _load_evidence_contract(
                manifest,
                manifest_dir=contract_dir,
                formal_acceptance_eligible=True,
                manifest_allowed_paths=[
                    "detection_algorithm.labels",
                    "detection_algorithm.model_id",
                ],
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            forged = json.loads(
                (
                    contract_dir / "evidence_contract_amendment.json"
                ).read_text(encoding="utf-8")
            )
            forged["single_variable_contract"][
                "allowed_algorithm_contract_differences"
            ].append("detection_algorithm.model_id")
            forged_path = root / "forged.json"
            forged_path.write_text(
                json.dumps(forged, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            forged_manifest = {
                "evidence_contract_json": str(forged_path),
                "evidence_contract_sha256": _file_hash(forged_path),
            }
            with self.assertRaisesRegex(
                ValueError,
                "not the audited frozen hash",
            ):
                _load_evidence_contract(
                    forged_manifest,
                    manifest_dir=root,
                    formal_acceptance_eligible=True,
                    manifest_allowed_paths=[
                        "detection_algorithm.labels",
                        "detection_algorithm.model_id",
                    ],
                )

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
                        baseline_path,
                        candidate_path,
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

            audit_hash_mismatch = copy.deepcopy(manifest)
            audit_hash_mismatch["clusters"][0]["baseline"][
                "audit_json_sha256"
            ] = "e" * 64
            cases.append(
                (
                    "audit hash mismatch",
                    audit_hash_mismatch,
                    "audit_json_sha256 does not match",
                )
            )

            algorithm_mismatch = copy.deepcopy(manifest)
            algorithm_mismatch["clusters"][0]["candidate"][
                "algorithm_sha256"
            ] = "e" * 64
            cases.append(
                (
                    "algorithm mismatch",
                    algorithm_mismatch,
                    "algorithm_sha256 does not match",
                )
            )

            detections_mismatch = copy.deepcopy(manifest)
            detections_mismatch["clusters"][0]["candidate"][
                "detections_sha256"
            ] = "e" * 64
            cases.append(
                (
                    "detections mismatch",
                    detections_mismatch,
                    "detections_sha256 does not match",
                )
            )

            generator_mismatch = copy.deepcopy(manifest)
            generator_mismatch["clusters"][0]["candidate"][
                "ground_truth_generator_sha256"
            ] = "e" * 64
            cases.append(
                (
                    "generator mismatch",
                    generator_mismatch,
                    "ground_truth_generator_sha256 does not match",
                )
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
                        baseline_path,
                        candidate_path,
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
            manifest["clusters"][0]["baseline"]["audit_json_sha256"] = (
                _file_hash(baseline_path)
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
            manifest["clusters"][0]["baseline"]["audit_json_sha256"] = (
                _file_hash(baseline_path)
            )
            tampered_gt = json.loads(
                candidate_path.read_text(encoding="utf-8")
            )
            tampered_gt["inputs"]["semantic_gt_sha256"] = "c" * 64
            candidate_path.write_text(
                json.dumps(tampered_gt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest["clusters"][0]["candidate"]["audit_json_sha256"] = (
                _file_hash(candidate_path)
            )
            with self.assertRaisesRegex(
                ValueError,
                "manifest ground_truth_sha256 does not match audit",
            ):
                compare_manifest_payload(manifest, manifest_dir=root)

    def test_rejects_paired_identity_and_unregistered_algorithm_changes(
        self,
    ) -> None:
        def run_case(label: str) -> None:
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
                            baseline_path,
                            candidate_path,
                        )
                    ]
                )
                expected = ""
                if label == "manifest scene alias":
                    manifest["clusters"][0]["scene_id"] = "fake-scene"
                    expected = "manifest scene_id does not match audit"
                elif label == "run contract mismatch":
                    candidate = json.loads(
                        candidate_path.read_text(encoding="utf-8")
                    )
                    candidate["inputs"][
                        "detection_run_contract_sha256"
                    ] = "4" * 64
                    candidate_path.write_text(
                        json.dumps(candidate, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    manifest["clusters"][0]["candidate"][
                        "audit_json_sha256"
                    ] = _file_hash(candidate_path)
                    expected = "paired input detection_run_contract_sha256"
                elif label == "undeclared algorithm change":
                    candidate = json.loads(
                        candidate_path.read_text(encoding="utf-8")
                    )
                    contract = candidate["variant_algorithms"]["candidate"][
                        "contract"
                    ]
                    contract["hidden_variable"] = True
                    algorithm_sha256 = _canonical_hash(contract)
                    candidate["variant_algorithms"]["candidate"][
                        "sha256"
                    ] = algorithm_sha256
                    candidate_path.write_text(
                        json.dumps(candidate, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    manifest["clusters"][0]["candidate"][
                        "audit_json_sha256"
                    ] = _file_hash(candidate_path)
                    manifest["clusters"][0]["candidate"][
                        "algorithm_sha256"
                    ] = algorithm_sha256
                    expected = "do not exactly match the preregistered paths"
                elif label == "declared but unchanged":
                    manifest[
                        "allowed_algorithm_contract_differences"
                    ].append("hidden_variable")
                    expected = "declared_but_unchanged"
                elif label == "ancestor path":
                    manifest[
                        "allowed_algorithm_contract_differences"
                    ] = ["operation", "operation.mode"]
                    expected = "ancestor/descendant"
                else:
                    self.fail(f"unknown test case {label}")

                with self.assertRaisesRegex(ValueError, expected):
                    compare_manifest_payload(manifest, manifest_dir=root)

        for label in (
            "manifest scene alias",
            "run contract mismatch",
            "undeclared algorithm change",
            "declared but unchanged",
            "ancestor path",
        ):
            with self.subTest(case=label):
                run_case(label)

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
                        baseline_path,
                        candidate_path,
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
                        baseline_path,
                        candidate_path,
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
                        baseline_path,
                        candidate_path,
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

    def test_formal_manifest_enforces_scene_trajectory_minimum(self) -> None:
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
                        baseline_path,
                        candidate_path,
                    )
                ]
            )
            manifest["formal_acceptance_eligible"] = True

            with self.assertRaisesRegex(ValueError, "frozen evidence contract"):
                compare_manifest_payload(manifest, manifest_dir=root)

    def test_formal_validation_rejects_reused_trajectory_fingerprint(
        self,
    ) -> None:
        clusters = self._formal_clusters()
        for cluster in clusters:
            cluster["trajectory_sha256"] = "a" * 64

        with self.assertRaisesRegex(
            ValueError,
            "reuses an input trajectory fingerprint",
        ):
            _validate_formal_cluster_minimum(clusters)

    def test_formal_validation_freezes_identity_and_algorithms(self) -> None:
        _validate_formal_cluster_minimum(self._formal_clusters())
        cases = (
            (
                "scene id maps to two assets",
                lambda clusters: clusters[1].update(
                    {"scene_sha256": "f" * 64}
                ),
                "one scene_id to multiple scene asset hashes",
            ),
            (
                "asset aliases two scenes",
                lambda clusters: clusters[2].update(
                    {"scene_sha256": clusters[0]["scene_sha256"]}
                ),
                "aliases one scene asset hash",
            ),
            (
                "external detector contract",
                lambda clusters: clusters[0]["candidate"].update(
                    {"formal_detection_contract_eligible": False}
                ),
                "requires embedded candidate detector contracts",
            ),
            (
                "baseline algorithm drift",
                lambda clusters: clusters[0]["baseline"].update(
                    {"algorithm_sha256": "e" * 64}
                ),
                "one frozen baseline algorithm hash",
            ),
            (
                "candidate algorithm drift",
                lambda clusters: clusters[0]["candidate"].update(
                    {"algorithm_sha256": "e" * 64}
                ),
                "one frozen candidate algorithm hash",
            ),
            (
                "ground truth generator drift",
                lambda clusters: clusters[0]["baseline"].update(
                    {"ground_truth_generator_sha256": "e" * 64}
                ),
                "one frozen ground-truth generator hash",
            ),
            (
                "evaluation parameter drift",
                lambda clusters: clusters[0].update(
                    {"parameters_sha256": "e" * 64}
                ),
                "one frozen evaluation parameters hash",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(case=label):
                clusters = self._formal_clusters()
                mutate(clusters)
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_formal_cluster_minimum(clusters)

    @staticmethod
    def _formal_clusters() -> list[dict]:
        clusters = []
        for index in range(6):
            scene_index = index // 2
            clusters.append(
                {
                    "scene_id": f"scene-{scene_index}",
                    "scene_sha256": f"{scene_index + 4:x}" * 64,
                    "trajectory_id": f"trajectory-{index % 2}",
                    "trajectory_sha256": f"{index + 7:x}" * 64,
                    "parameters_sha256": PARAMETERS_HASH,
                    "baseline": {
                        "algorithm_sha256": "b" * 64,
                        "ground_truth_generator_sha256": (
                            GROUND_TRUTH_GENERATOR_HASH
                        ),
                        "formal_detection_contract_eligible": True,
                    },
                    "candidate": {
                        "algorithm_sha256": "c" * 64,
                        "formal_detection_contract_eligible": True,
                    },
                }
            )
        return clusters

    @staticmethod
    def _manifest(clusters: list[dict]) -> dict:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "role": "unit_test",
            "formal_acceptance_eligible": False,
            "allowed_algorithm_contract_differences": ["operation"],
            "clusters": clusters,
        }

    @staticmethod
    def _cluster(
        cluster_id: str,
        baseline_path: Path,
        candidate_path: Path,
    ) -> dict:
        baseline_payload = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
        candidate_payload = json.loads(
            candidate_path.read_text(encoding="utf-8")
        )
        _, trajectory_id = cluster_id.split("/", 1)
        scene_id = baseline_payload["inputs"]["scene_id"]
        return {
            "cluster_id": cluster_id,
            "scene_id": scene_id,
            "scene_sha256": baseline_payload["inputs"]["scene_sha256"],
            "trajectory_id": trajectory_id,
            "trajectory_sha256": baseline_payload["inputs"][
                "input_frame_hashes_sha256"
            ],
            "baseline": {
                "audit_json": baseline_path.name,
                "audit_json_sha256": _file_hash(baseline_path),
                "variant": "baseline",
                "algorithm_sha256": baseline_payload[
                    "variant_algorithms"
                ]["baseline"]["sha256"],
                "parameters_sha256": PARAMETERS_HASH,
                "detections_sha256": DETECTIONS_HASH,
                "ground_truth_sha256": GROUND_TRUTH_HASH,
                "ground_truth_generator_sha256": (
                    GROUND_TRUTH_GENERATOR_HASH
                ),
            },
            "candidate": {
                "audit_json": candidate_path.name,
                "audit_json_sha256": _file_hash(candidate_path),
                "variant": "candidate",
                "algorithm_sha256": candidate_payload[
                    "variant_algorithms"
                ]["candidate"]["sha256"],
                "parameters_sha256": PARAMETERS_HASH,
                "detections_sha256": DETECTIONS_HASH,
                "ground_truth_sha256": GROUND_TRUTH_HASH,
                "ground_truth_generator_sha256": (
                    GROUND_TRUTH_GENERATOR_HASH
                ),
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
                    baseline_path,
                    candidate_path,
                )
            )
        return self._manifest(clusters)

    @staticmethod
    def _write_audit(path: Path, variants: dict) -> Path:
        variant_algorithms = {}
        for name in variants:
            contract = {"operation": name}
            variant_algorithms[name] = {
                "contract": contract,
                "sha256": _canonical_hash(contract),
            }
        path.write_text(
            json.dumps(
                {
                    "schema_version": "grounding_box_audit_v1",
                    "evaluation_contract": EVALUATION_CONTRACT,
                    "evaluation_parameters_sha256": PARAMETERS_HASH,
                    "inputs": {
                        "detections_sha256": DETECTIONS_HASH,
                        "semantic_gt_sha256": GROUND_TRUTH_HASH,
                        "semantic_gt_generator_sha256": (
                            GROUND_TRUTH_GENERATOR_HASH
                        ),
                        "scene_id": "scene",
                        "scene_sha256": SCENE_HASH,
                        "frames_metadata_sha256": FRAMES_METADATA_HASH,
                        "selected_frame_indices_sha256": (
                            SELECTED_FRAME_INDICES_HASH
                        ),
                        "input_frame_hashes_sha256": (
                            INPUT_FRAME_HASHES_HASH
                        ),
                        "detection_run_contract_sha256": (
                            DETECTION_RUN_CONTRACT_HASH
                        ),
                        "semantic_gt_integrity_contract": (
                            SEMANTIC_INTEGRITY_CONTRACT
                        ),
                        "semantic_gt_integrity_sha256": (
                            SEMANTIC_INTEGRITY_HASH
                        ),
                        "formal_detection_contract_eligible": True,
                    },
                    "variant_algorithms": variant_algorithms,
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
