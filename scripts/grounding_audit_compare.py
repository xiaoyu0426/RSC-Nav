#!/usr/bin/env python3
"""Compare paired grounding audits across scene-trajectory clusters.

The manifest schema is::

    {
      "schema_version": "grounding_audit_compare_manifest_v1",
      "role": "formal_unseen_replay",
      "formal_acceptance_eligible": true,
      "evidence_contract_json": "evidence_contract_amendment.json",
      "evidence_contract_sha256": "<frozen evidence contract hash>",
      "allowed_algorithm_contract_differences": ["detection_algorithm.labels"],
      "clusters": [
        {
          "cluster_id": "scene_a/trajectory_01",
          "scene_id": "scene_a",
          "scene_sha256": "<scene asset hash>",
          "trajectory_id": "trajectory_01",
          "trajectory_sha256": "<paired input-frame hash>",
          "baseline": {
            "audit_json": "baseline/audit.json",
            "audit_json_sha256": "<64 hex characters>",
            "variant": "baseline",
            "algorithm_sha256": "<64 hex characters>",
            "parameters_sha256": "<64 hex characters>",
            "detections_sha256": "<64 hex characters>",
            "ground_truth_sha256": "<64 hex characters>",
            "ground_truth_generator_sha256": "<64 hex characters>"
          },
          "candidate": {
            "audit_json": "candidate/audit.json",
            "audit_json_sha256": "<64 hex characters>",
            "variant": "candidate",
            "algorithm_sha256": "<candidate algorithm hash>",
            "parameters_sha256": "<same parameters hash>",
            "detections_sha256": "<candidate detections hash>",
            "ground_truth_sha256": "<same ground-truth hash>",
            "ground_truth_generator_sha256": "<same generator hash>"
          }
        }
      ]
    }

Audit paths are resolved relative to the manifest. Every declared parameter
and ground-truth hash is verified against the referenced audit before metrics
are read. The comparator reads only the named variants and preregistered
numeric metric paths; it never reads VLM verdicts or frame-level records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.grounding_box_audit import (
        FROZEN_REPLAY_INTEGRITY_THRESHOLDS,
        build_audit_report,
    )
except ModuleNotFoundError:
    from grounding_box_audit import (
        FROZEN_REPLAY_INTEGRITY_THRESHOLDS,
        build_audit_report,
    )


SCHEMA_VERSION = "grounding_audit_compare_v1"
MANIFEST_SCHEMA_VERSION = "grounding_audit_compare_manifest_v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_731
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
FORMAL_EVIDENCE_CONTRACT_PROTOCOL = (
    "grounding-box-contrastive-prompt-v1-evidence-amendment"
)
FORMAL_EVIDENCE_CONTRACT_SHA256 = (
    "a1d5111427164d93fe2a5b461a2468bda59b6f740c723706d066fae5baf6135c"
)
FORMAL_ALLOWED_ALGORITHM_DIFFERENCES = (
    "detection_algorithm.labels",
)

PRIMARY_METRIC = "hard_negative_door_fp_per_100_frames"
METRIC_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": PRIMARY_METRIC,
        "source_path": (
            "hard_negative_door_fp",
            "hard_negative_fp_per_100_frames",
        ),
        "unit": "false_positives_per_100_evaluated_frames",
        "improvement_direction": "lower",
        "primary": True,
    },
    {
        "name": "door_recall",
        "source_path": (
            "per_class",
            "door",
            "operating_point",
            "recall",
        ),
        "unit": "fraction",
        "improvement_direction": "higher",
        "primary": False,
    },
    {
        "name": "door_physical_instance_recall",
        "source_path": (
            "per_class",
            "door",
            "operating_point",
            "physical_instance_recall",
            "recall",
        ),
        "unit": "fraction",
        "improvement_direction": "higher",
        "primary": False,
    },
    {
        "name": "door_tp_median_iou",
        "source_path": (
            "per_class",
            "door",
            "operating_point",
            "tp_iou_distribution",
            "distribution",
            "median",
        ),
        "unit": "iou",
        "improvement_direction": "higher",
        "primary": False,
    },
    {
        "name": "door_fp_per_100_frames",
        "source_path": (
            "per_class",
            "door",
            "operating_point",
            "fp_per_100_evaluated_frames",
        ),
        "unit": "false_positives_per_100_evaluated_frames",
        "improvement_direction": "lower",
        "primary": False,
    },
    {
        "name": "door_duplicate_fp_per_100_frames",
        "source_path": (
            "per_class",
            "door",
            "operating_point",
            "duplicate_fp_per_100_evaluated_frames",
        ),
        "unit": "false_positives_per_100_evaluated_frames",
        "improvement_direction": "lower",
        "primary": False,
    },
    {
        "name": "door_xz_median_m",
        "source_path": (
            "xz_error_m",
            "door",
            "distribution",
            "median",
        ),
        "unit": "meters",
        "improvement_direction": "lower",
        "primary": False,
    },
    {
        "name": "door_xz_p90_m",
        "source_path": (
            "xz_error_m",
            "door",
            "distribution",
            "p90",
        ),
        "unit": "meters",
        "improvement_direction": "lower",
        "primary": False,
    },
)


def compare_manifest_payload(
    manifest: Mapping[str, Any],
    *,
    manifest_dir: Path,
) -> dict[str, Any]:
    """Validate a manifest and compare every baseline/candidate cluster pair."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest JSON root must be an object")
    schema_version = manifest.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "manifest schema_version must be "
            f"{MANIFEST_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    raw_clusters = manifest.get("clusters")
    if not isinstance(raw_clusters, list) or not raw_clusters:
        raise ValueError("manifest clusters must be a non-empty list")
    role = _required_string(manifest, "role", "manifest")
    formal_acceptance_eligible = manifest.get(
        "formal_acceptance_eligible"
    )
    if not isinstance(formal_acceptance_eligible, bool):
        raise ValueError(
            "manifest formal_acceptance_eligible must be a boolean"
        )
    allowed_algorithm_differences = manifest.get(
        "allowed_algorithm_contract_differences"
    )
    if not isinstance(allowed_algorithm_differences, list) or not all(
        isinstance(item, str) and item.strip()
        for item in allowed_algorithm_differences
    ):
        raise ValueError(
            "manifest allowed_algorithm_contract_differences must be a "
            "list of non-empty dotted paths"
        )
    allowed_algorithm_differences = [
        item.strip() for item in allowed_algorithm_differences
    ]
    if len(set(allowed_algorithm_differences)) != len(
        allowed_algorithm_differences
    ):
        raise ValueError(
            "manifest allowed_algorithm_contract_differences has duplicates"
        )
    _validate_allowed_difference_paths(allowed_algorithm_differences)

    manifest_dir = manifest_dir.expanduser().resolve()
    evidence_contract = _load_evidence_contract(
        manifest,
        manifest_dir=manifest_dir,
        formal_acceptance_eligible=formal_acceptance_eligible,
        manifest_allowed_paths=allowed_algorithm_differences,
    )
    audit_cache: dict[Path, tuple[Mapping[str, Any], str]] = {}
    cluster_results: list[dict[str, Any]] = []
    seen_cluster_ids: set[str] = set()
    unavailable_metrics: list[dict[str, str]] = []

    for cluster_index, raw_cluster in enumerate(raw_clusters):
        context = f"clusters[{cluster_index}]"
        if not isinstance(raw_cluster, Mapping):
            raise ValueError(f"{context} must be an object")
        cluster_id = _required_string(raw_cluster, "cluster_id", context)
        scene_id = _required_string(raw_cluster, "scene_id", context)
        scene_sha256 = _required_sha256(
            raw_cluster,
            "scene_sha256",
            context,
        )
        trajectory_id = _required_string(
            raw_cluster,
            "trajectory_id",
            context,
        )
        trajectory_sha256 = _required_sha256(
            raw_cluster,
            "trajectory_sha256",
            context,
        )
        if cluster_id in seen_cluster_ids:
            raise ValueError(f"duplicate cluster_id: {cluster_id!r}")
        seen_cluster_ids.add(cluster_id)

        baseline = _load_side(
            raw_cluster,
            "baseline",
            cluster_id=cluster_id,
            manifest_dir=manifest_dir,
            audit_cache=audit_cache,
            require_live_reverification=formal_acceptance_eligible,
        )
        candidate = _load_side(
            raw_cluster,
            "candidate",
            cluster_id=cluster_id,
            manifest_dir=manifest_dir,
            audit_cache=audit_cache,
            require_live_reverification=formal_acceptance_eligible,
        )
        if baseline["parameters_sha256"] != candidate["parameters_sha256"]:
            raise ValueError(
                f"cluster {cluster_id!r} parameters_sha256 mismatch: "
                f"baseline={baseline['parameters_sha256']} "
                f"candidate={candidate['parameters_sha256']}"
            )
        if baseline["ground_truth_sha256"] != candidate["ground_truth_sha256"]:
            raise ValueError(
                f"cluster {cluster_id!r} ground_truth_sha256 mismatch: "
                f"baseline={baseline['ground_truth_sha256']} "
                f"candidate={candidate['ground_truth_sha256']}"
            )
        if (
            baseline["ground_truth_generator_sha256"]
            != candidate["ground_truth_generator_sha256"]
        ):
            raise ValueError(
                f"cluster {cluster_id!r} ground_truth_generator_sha256 "
                "mismatch"
            )
        _validate_paired_input_identity(
            baseline,
            candidate,
            cluster_id=cluster_id,
            manifest_scene_id=scene_id,
            manifest_scene_sha256=scene_sha256,
            manifest_trajectory_sha256=trajectory_sha256,
        )
        _validate_algorithm_difference_scope(
            baseline["algorithm_contract"],
            candidate["algorithm_contract"],
            allowed_paths=allowed_algorithm_differences,
            cluster_id=cluster_id,
        )
        if evidence_contract is not None:
            _validate_registered_algorithm_labels(
                baseline["algorithm_contract"],
                candidate["algorithm_contract"],
                evidence_contract=evidence_contract,
                cluster_id=cluster_id,
            )

        metric_results: dict[str, Any] = {}
        for spec in METRIC_SPECS:
            name = str(spec["name"])
            baseline_value, baseline_reason = _extract_numeric_metric(
                baseline["variant_payload"],
                spec["source_path"],
            )
            candidate_value, candidate_reason = _extract_numeric_metric(
                candidate["variant_payload"],
                spec["source_path"],
            )
            paired = _paired_metric(
                baseline_value,
                candidate_value,
                baseline_reason=baseline_reason,
                candidate_reason=candidate_reason,
                include_relative_reduction=(name == PRIMARY_METRIC),
            )
            metric_results[name] = paired
            if not paired["available"]:
                unavailable_metrics.append(
                    {
                        "cluster_id": cluster_id,
                        "metric": name,
                        "reason": paired["reason"],
                    }
                )
            elif (
                name == PRIMARY_METRIC
                and not paired["relative_reduction"]["available"]
            ):
                unavailable_metrics.append(
                    {
                        "cluster_id": cluster_id,
                        "metric": f"{name}.relative_reduction",
                        "reason": paired["relative_reduction"]["reason"],
                    }
                )

        cluster_results.append(
            {
                "cluster_id": cluster_id,
                "scene_id": scene_id,
                "scene_sha256": scene_sha256,
                "trajectory_id": trajectory_id,
                "trajectory_sha256": trajectory_sha256,
                "parameters_sha256": baseline["parameters_sha256"],
                "ground_truth_sha256": baseline["ground_truth_sha256"],
                "baseline": _side_provenance(baseline),
                "candidate": _side_provenance(candidate),
                "metrics": metric_results,
            }
        )

    if formal_acceptance_eligible:
        _validate_formal_cluster_minimum(cluster_results)

    comparisons = {
        str(spec["name"]): _aggregate_metric(
            cluster_results,
            str(spec["name"]),
            include_relative_reduction=(
                str(spec["name"]) == PRIMARY_METRIC
            ),
        )
        for spec in METRIC_SPECS
    }
    for metric_name, comparison in comparisons.items():
        if not comparison["available"]:
            unavailable_metrics.append(
                {
                    "cluster_id": "__macro__",
                    "metric": metric_name,
                    "reason": comparison["reason"],
                }
            )
        relative = comparison.get("relative_reduction")
        if relative is not None and not relative["available"]:
            unavailable_metrics.append(
                {
                    "cluster_id": "__macro__",
                    "metric": f"{metric_name}.relative_reduction",
                    "reason": relative["reason"],
                }
            )

    metric_definitions = {
        str(spec["name"]): {
            "source_path": "variants.<manifest-side-variant>."
            + ".".join(spec["source_path"]),
            "unit": spec["unit"],
            "improvement_direction": spec["improvement_direction"],
            "primary": spec["primary"],
            "paired_delta_definition": "candidate_minus_baseline",
        }
        for spec in METRIC_SPECS
    }
    metric_definitions[PRIMARY_METRIC]["relative_reduction_definition"] = (
        "(baseline_minus_candidate)_divided_by_baseline"
    )
    metric_definitions[PRIMARY_METRIC][
        "relative_reduction_improvement_direction"
    ] = "higher"

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_canonical_sha256": _canonical_sha256(manifest),
        "primary_metric": PRIMARY_METRIC,
        "role": role,
        "formal_acceptance_eligible": formal_acceptance_eligible,
        "evidence_contract": (
            evidence_contract["provenance"]
            if evidence_contract is not None
            else None
        ),
        "allowed_algorithm_contract_differences": (
            allowed_algorithm_differences
        ),
        "cluster_count": len(cluster_results),
        "bootstrap": {
            "unit": "scene_trajectory_cluster_pair",
            "frame_level_resampling": False,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "interval": "percentile",
            "percentile_method": "linear_interpolation",
            "minimum_clusters_for_interval": 2,
        },
        "metric_definitions": metric_definitions,
        "clusters": cluster_results,
        "comparisons": comparisons,
        "unavailable_metrics": unavailable_metrics,
    }


def _load_evidence_contract(
    manifest: Mapping[str, Any],
    *,
    manifest_dir: Path,
    formal_acceptance_eligible: bool,
    manifest_allowed_paths: Sequence[str],
) -> dict[str, Any] | None:
    path_value = manifest.get("evidence_contract_json")
    sha_value = manifest.get("evidence_contract_sha256")
    if path_value is None and sha_value is None:
        if formal_acceptance_eligible:
            raise ValueError(
                "formal comparison requires a frozen evidence contract"
            )
        return None
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("manifest evidence_contract_json must be a path")
    if (
        not isinstance(sha_value, str)
        or not SHA256_PATTERN.fullmatch(sha_value)
    ):
        raise ValueError(
            "manifest evidence_contract_sha256 must be 64 hex characters"
        )
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != sha_value.lower():
        raise ValueError(
            "manifest evidence_contract_sha256 does not match file"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("evidence contract root must be an object")
    protocol_id = _required_string(
        payload,
        "protocol_id",
        "evidence contract",
    )
    if formal_acceptance_eligible:
        if actual_sha256 != FORMAL_EVIDENCE_CONTRACT_SHA256:
            raise ValueError(
                "formal comparison evidence contract hash is not the "
                "audited frozen hash"
            )
        if protocol_id != FORMAL_EVIDENCE_CONTRACT_PROTOCOL:
            raise ValueError(
                "formal comparison evidence contract protocol is invalid"
            )

    single_variable = payload.get("single_variable_contract")
    if not isinstance(single_variable, Mapping):
        raise ValueError(
            "evidence contract single_variable_contract is required"
        )
    registered_paths = single_variable.get(
        "allowed_algorithm_contract_differences"
    )
    if not isinstance(registered_paths, list) or not all(
        isinstance(item, str) and item.strip() for item in registered_paths
    ):
        raise ValueError(
            "evidence contract allowed difference paths are invalid"
        )
    registered_paths = [item.strip() for item in registered_paths]
    if registered_paths != list(manifest_allowed_paths):
        raise ValueError(
            "manifest allowed algorithm differences do not match the frozen "
            "evidence contract"
        )
    if formal_acceptance_eligible and tuple(registered_paths) != (
        FORMAL_ALLOWED_ALGORITHM_DIFFERENCES
    ):
        raise ValueError(
            "formal comparison allowed algorithm differences are not frozen"
        )
    if payload.get("semantic_replay_integrity_thresholds") != (
        FROZEN_REPLAY_INTEGRITY_THRESHOLDS
    ):
        raise ValueError(
            "evidence contract semantic replay thresholds do not match the "
            "audited implementation"
        )
    baseline_labels = single_variable.get("baseline_labels")
    candidate_labels = single_variable.get("candidate_labels")
    for name, labels in (
        ("baseline_labels", baseline_labels),
        ("candidate_labels", candidate_labels),
    ):
        if (
            not isinstance(labels, list)
            or not labels
            or not all(
                isinstance(item, str) and item.strip() for item in labels
            )
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(f"evidence contract {name} is invalid")

    amendment = payload.get("amends_preregistration")
    if not isinstance(amendment, Mapping):
        raise ValueError(
            "evidence contract amends_preregistration is required"
        )
    prereg_path_value = amendment.get("path")
    if not isinstance(prereg_path_value, str) or not prereg_path_value.strip():
        raise ValueError("evidence contract preregistration path is invalid")
    prereg_path = (path.parent / prereg_path_value).resolve()
    if not prereg_path.is_file():
        raise FileNotFoundError(prereg_path)
    prereg_sha256 = _required_sha256(
        amendment,
        "sha256_before_amendment",
        "evidence contract amends_preregistration",
    )
    if _file_sha256(prereg_path) != prereg_sha256:
        raise ValueError(
            "evidence contract preregistration hash does not match file"
        )
    return {
        "baseline_labels": list(baseline_labels),
        "candidate_labels": list(candidate_labels),
        "provenance": {
            "path": str(path),
            "sha256": actual_sha256,
            "protocol_id": protocol_id,
            "preregistration_path": str(prereg_path),
            "preregistration_sha256": prereg_sha256,
        },
    }


def _validate_registered_algorithm_labels(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    evidence_contract: Mapping[str, Any],
    cluster_id: str,
) -> None:
    for side_name, contract, expected_key in (
        ("baseline", baseline, "baseline_labels"),
        ("candidate", candidate, "candidate_labels"),
    ):
        detector = contract.get("detection_algorithm")
        if not isinstance(detector, Mapping):
            raise ValueError(
                f"cluster {cluster_id!r} {side_name} algorithm lacks "
                "detection_algorithm"
            )
        labels = detector.get("labels")
        if labels != evidence_contract[expected_key]:
            raise ValueError(
                f"cluster {cluster_id!r} {side_name} labels do not match "
                "the frozen evidence contract"
            )


def _load_side(
    cluster: Mapping[str, Any],
    side_name: str,
    *,
    cluster_id: str,
    manifest_dir: Path,
    audit_cache: dict[Path, tuple[Mapping[str, Any], str]],
    require_live_reverification: bool,
) -> dict[str, Any]:
    side = cluster.get(side_name)
    context = f"cluster {cluster_id!r} {side_name}"
    if not isinstance(side, Mapping):
        raise ValueError(f"{context} must be an object")

    audit_json = _required_string(side, "audit_json", context)
    variant_name = _required_string(side, "variant", context)
    declared_audit_sha256 = _required_sha256(
        side,
        "audit_json_sha256",
        context,
    )
    algorithm_sha256 = _required_sha256(side, "algorithm_sha256", context)
    parameters_sha256 = _required_sha256(side, "parameters_sha256", context)
    detections_sha256 = _required_sha256(
        side,
        "detections_sha256",
        context,
    )
    ground_truth_sha256 = _required_sha256(
        side,
        "ground_truth_sha256",
        context,
    )
    ground_truth_generator_sha256 = _required_sha256(
        side,
        "ground_truth_generator_sha256",
        context,
    )
    audit_path = Path(audit_json).expanduser()
    if not audit_path.is_absolute():
        audit_path = manifest_dir / audit_path
    audit_path = audit_path.resolve()
    if audit_path not in audit_cache:
        if not audit_path.is_file():
            raise ValueError(f"{context} audit_json does not exist: {audit_json}")
        payload = _strict_json_load(audit_path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{context} audit JSON root must be an object")
        audit_cache[audit_path] = (payload, _file_sha256(audit_path))
    audit_payload, audit_sha256 = audit_cache[audit_path]
    if declared_audit_sha256 != audit_sha256:
        raise ValueError(
            f"{context} manifest audit_json_sha256 does not match audit file"
        )

    evaluation_contract = audit_payload.get("evaluation_contract")
    if not isinstance(evaluation_contract, Mapping):
        raise ValueError(
            f"{context} audit JSON must contain an evaluation_contract object"
        )
    audit_parameters_sha256 = _required_sha256(
        audit_payload,
        "evaluation_parameters_sha256",
        f"{context} audit JSON",
    )
    computed_parameters_sha256 = _canonical_sha256(evaluation_contract)
    if audit_parameters_sha256 != computed_parameters_sha256:
        raise ValueError(
            f"{context} audit evaluation_parameters_sha256 does not match "
            "its evaluation_contract"
        )
    if parameters_sha256 != audit_parameters_sha256:
        raise ValueError(
            f"{context} manifest parameters_sha256 does not match audit "
            "evaluation_parameters_sha256"
        )

    inputs = audit_payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"{context} audit JSON must contain an inputs object")
    audit_ground_truth_sha256 = _required_sha256(
        inputs,
        "semantic_gt_sha256",
        f"{context} audit JSON inputs",
    )
    if ground_truth_sha256 != audit_ground_truth_sha256:
        raise ValueError(
            f"{context} manifest ground_truth_sha256 does not match audit "
            "inputs.semantic_gt_sha256"
        )
    audit_detections_sha256 = _required_sha256(
        inputs,
        "detections_sha256",
        f"{context} audit JSON inputs",
    )
    if detections_sha256 != audit_detections_sha256:
        raise ValueError(
            f"{context} manifest detections_sha256 does not match audit "
            "inputs.detections_sha256"
        )
    audit_ground_truth_generator_sha256 = _required_sha256(
        inputs,
        "semantic_gt_generator_sha256",
        f"{context} audit JSON inputs",
    )
    if (
        ground_truth_generator_sha256
        != audit_ground_truth_generator_sha256
    ):
        raise ValueError(
            f"{context} manifest ground_truth_generator_sha256 does not "
            "match audit inputs.semantic_gt_generator_sha256"
        )
    audit_scene_id = _required_string(
        inputs,
        "scene_id",
        f"{context} audit JSON inputs",
    )
    audit_scene_sha256 = _required_sha256(
        inputs,
        "scene_sha256",
        f"{context} audit JSON inputs",
    )
    audit_frames_metadata_sha256 = _required_sha256(
        inputs,
        "frames_metadata_sha256",
        f"{context} audit JSON inputs",
    )
    audit_selected_frame_indices_sha256 = _required_sha256(
        inputs,
        "selected_frame_indices_sha256",
        f"{context} audit JSON inputs",
    )
    audit_input_frame_hashes_sha256 = _required_sha256(
        inputs,
        "input_frame_hashes_sha256",
        f"{context} audit JSON inputs",
    )
    audit_detection_run_contract_sha256 = _required_sha256(
        inputs,
        "detection_run_contract_sha256",
        f"{context} audit JSON inputs",
    )
    semantic_integrity = inputs.get("semantic_gt_integrity_contract")
    if not isinstance(semantic_integrity, Mapping):
        raise ValueError(
            f"{context} audit inputs must contain semantic GT integrity"
        )
    semantic_integrity_sha256 = _required_sha256(
        inputs,
        "semantic_gt_integrity_sha256",
        f"{context} audit JSON inputs",
    )
    if semantic_integrity_sha256 != _canonical_sha256(semantic_integrity):
        raise ValueError(
            f"{context} semantic GT integrity hash does not match contract"
        )
    if semantic_integrity.get("contract") != (
        "full_frame_rgb_depth_replay_v1"
    ):
        raise ValueError(
            f"{context} semantic GT integrity contract is not full-frame"
        )
    formal_detection_contract_eligible = inputs.get(
        "formal_detection_contract_eligible"
    )
    if not isinstance(formal_detection_contract_eligible, bool):
        raise ValueError(
            f"{context} formal_detection_contract_eligible must be boolean"
        )

    variants = audit_payload.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError(f"{context} audit JSON must contain a variants object")
    if variant_name not in variants:
        raise ValueError(
            f"{context} variant {variant_name!r} does not exist in audit JSON"
        )
    variant_payload = variants[variant_name]
    if not isinstance(variant_payload, Mapping):
        raise ValueError(
            f"{context} variant {variant_name!r} must be an object"
        )
    variant_algorithms = audit_payload.get("variant_algorithms")
    if not isinstance(variant_algorithms, Mapping):
        raise ValueError(
            f"{context} audit JSON must contain a variant_algorithms object"
        )
    algorithm_record = variant_algorithms.get(variant_name)
    if not isinstance(algorithm_record, Mapping):
        raise ValueError(
            f"{context} variant algorithm {variant_name!r} does not exist"
        )
    algorithm_contract = algorithm_record.get("contract")
    if not isinstance(algorithm_contract, Mapping):
        raise ValueError(
            f"{context} variant algorithm must contain a contract object"
        )
    audit_algorithm_sha256 = _required_sha256(
        algorithm_record,
        "sha256",
        f"{context} variant algorithm",
    )
    if audit_algorithm_sha256 != _canonical_sha256(algorithm_contract):
        raise ValueError(
            f"{context} variant algorithm sha256 does not match its contract"
        )
    if algorithm_sha256 != audit_algorithm_sha256:
        raise ValueError(
            f"{context} manifest algorithm_sha256 does not match audit "
            "variant algorithm"
        )
    if require_live_reverification:
        _reverify_formal_audit(
            audit_payload,
            audit_path=audit_json,
        )
    return {
        "audit_json": audit_json,
        "audit_json_sha256": audit_sha256,
        "variant": variant_name,
        "variant_payload": variant_payload,
        "algorithm_contract": algorithm_contract,
        "algorithm_sha256": audit_algorithm_sha256,
        "parameters_sha256": audit_parameters_sha256,
        "detections_sha256": audit_detections_sha256,
        "ground_truth_sha256": audit_ground_truth_sha256,
        "ground_truth_generator_sha256": (
            audit_ground_truth_generator_sha256
        ),
        "scene_id": audit_scene_id,
        "scene_sha256": audit_scene_sha256,
        "frames_metadata_sha256": audit_frames_metadata_sha256,
        "selected_frame_indices_sha256": (
            audit_selected_frame_indices_sha256
        ),
        "input_frame_hashes_sha256": audit_input_frame_hashes_sha256,
        "detection_run_contract_sha256": (
            audit_detection_run_contract_sha256
        ),
        "semantic_gt_integrity_sha256": semantic_integrity_sha256,
        "formal_detection_contract_eligible": (
            formal_detection_contract_eligible
        ),
    }


def _reverify_formal_audit(
    audit_payload: Mapping[str, Any],
    *,
    audit_path: Path,
) -> None:
    inputs = audit_payload.get("inputs")
    parameters = audit_payload.get("parameters")
    if not isinstance(inputs, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError(
            f"formal audit {audit_path} lacks inputs or parameters"
        )
    if inputs.get("detection_contract_source") != "embedded":
        raise ValueError(
            f"formal audit {audit_path} does not use an embedded detector "
            "contract"
        )
    nms = parameters.get("nms")
    if not isinstance(nms, Mapping):
        raise ValueError(f"formal audit {audit_path} lacks NMS parameters")
    detections_path = Path(
        _required_string(inputs, "detections_json", "formal audit inputs")
    )
    semantic_gt_path = Path(
        _required_string(inputs, "semantic_gt_json", "formal audit inputs")
    )
    reconstructed = build_audit_report(
        detections_path=detections_path,
        semantic_gt_path=semantic_gt_path,
        detector_contract_path=None,
        score_threshold=_required_finite_number(
            parameters,
            "score_threshold",
            "formal audit parameters",
        ),
        match_iou=_required_finite_number(
            parameters,
            "fixed_operating_point_iou",
            "formal audit parameters",
        ),
        nms_iou=_required_finite_number(
            nms,
            "iou_threshold",
            "formal audit NMS parameters",
        ),
        require_source_file_verification=True,
    )
    if _canonical_sha256(reconstructed) != _canonical_sha256(audit_payload):
        raise ValueError(
            f"formal audit {audit_path} does not match live recomputation"
        )


def _required_finite_number(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> float:
    value = payload.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{context}.{key} must be a finite number")
    return float(value)


def _side_provenance(side: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "audit_json": str(side["audit_json"]),
        "audit_json_sha256": str(side["audit_json_sha256"]),
        "variant": str(side["variant"]),
        "algorithm_sha256": str(side["algorithm_sha256"]),
        "detections_sha256": str(side["detections_sha256"]),
        "ground_truth_generator_sha256": str(
            side["ground_truth_generator_sha256"]
        ),
        "detection_run_contract_sha256": str(
            side["detection_run_contract_sha256"]
        ),
        "formal_detection_contract_eligible": bool(
            side["formal_detection_contract_eligible"]
        ),
    }


def _paired_metric(
    baseline: float | None,
    candidate: float | None,
    *,
    baseline_reason: str | None,
    candidate_reason: str | None,
    include_relative_reduction: bool,
) -> dict[str, Any]:
    reasons = []
    if baseline is None:
        reasons.append(f"baseline: {baseline_reason or 'metric unavailable'}")
    if candidate is None:
        reasons.append(f"candidate: {candidate_reason or 'metric unavailable'}")
    if reasons:
        result: dict[str, Any] = {
            "available": False,
            "baseline": baseline,
            "candidate": candidate,
            "paired_delta": None,
            "reason": "; ".join(reasons),
        }
    else:
        assert baseline is not None and candidate is not None
        result = {
            "available": True,
            "baseline": baseline,
            "candidate": candidate,
            "paired_delta": candidate - baseline,
        }

    if include_relative_reduction:
        if not result["available"]:
            result["relative_reduction"] = {
                "available": False,
                "value": None,
                "reason": result["reason"],
            }
        elif baseline == 0.0:
            result["relative_reduction"] = {
                "available": False,
                "value": None,
                "reason": "baseline is zero; relative reduction is undefined",
            }
        else:
            assert candidate is not None
            result["relative_reduction"] = {
                "available": True,
                "value": (baseline - candidate) / baseline,
            }
    return result


def _aggregate_metric(
    clusters: Sequence[Mapping[str, Any]],
    metric_name: str,
    *,
    include_relative_reduction: bool,
) -> dict[str, Any]:
    rows = [cluster["metrics"][metric_name] for cluster in clusters]
    unavailable_cluster_ids = [
        str(cluster["cluster_id"])
        for cluster, row in zip(clusters, rows)
        if not row["available"]
    ]
    result: dict[str, Any] = {
        "available": not unavailable_cluster_ids,
        "cluster_count": len(clusters),
        "available_cluster_count": len(clusters) - len(unavailable_cluster_ids),
        "unavailable_cluster_ids": unavailable_cluster_ids,
    }
    if unavailable_cluster_ids:
        result.update(
            {
                "macro_average": None,
                "paired_delta_bootstrap_95_ci": None,
                "reason": (
                    "formal paired comparison requires every cluster; "
                    "unavailable in: "
                    + ", ".join(unavailable_cluster_ids)
                ),
            }
        )
    else:
        baseline_values = [float(row["baseline"]) for row in rows]
        candidate_values = [float(row["candidate"]) for row in rows]
        delta_values = [float(row["paired_delta"]) for row in rows]
        result.update(
            {
                "macro_average": {
                    "baseline": _mean(baseline_values),
                    "candidate": _mean(candidate_values),
                    "paired_delta": _mean(delta_values),
                },
                "paired_delta_bootstrap_95_ci": _paired_bootstrap_ci(
                    delta_values
                ),
            }
        )

    if include_relative_reduction:
        unavailable_relative_ids = [
            str(cluster["cluster_id"])
            for cluster, row in zip(clusters, rows)
            if not row["relative_reduction"]["available"]
        ]
        if unavailable_relative_ids:
            result["relative_reduction"] = {
                "available": False,
                "cluster_count": len(clusters),
                "available_cluster_count": (
                    len(clusters) - len(unavailable_relative_ids)
                ),
                "unavailable_cluster_ids": unavailable_relative_ids,
                "macro_average": None,
                "bootstrap_95_ci": None,
                "reason": (
                    "formal macro relative reduction requires every cluster; "
                    "unavailable in: "
                    + ", ".join(unavailable_relative_ids)
                ),
            }
        else:
            values = [
                float(row["relative_reduction"]["value"]) for row in rows
            ]
            result["relative_reduction"] = {
                "available": True,
                "cluster_count": len(clusters),
                "available_cluster_count": len(clusters),
                "unavailable_cluster_ids": [],
                "macro_average": _mean(values),
                "bootstrap_95_ci": _paired_bootstrap_ci(values),
            }
    return result


def _paired_bootstrap_ci(
    cluster_values: Sequence[float],
) -> dict[str, float] | None:
    if not cluster_values:
        raise ValueError("paired bootstrap requires at least one cluster")
    if len(cluster_values) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(cluster_values)
    bootstrap_means = []
    for _ in range(BOOTSTRAP_REPLICATES):
        bootstrap_means.append(
            sum(cluster_values[rng.randrange(count)] for _ in range(count))
            / count
        )
    bootstrap_means.sort()
    return {
        "lower": _quantile(bootstrap_means, 0.025),
        "upper": _quantile(bootstrap_means, 0.975),
    }


def _validate_formal_cluster_minimum(
    clusters: Sequence[Mapping[str, Any]],
) -> None:
    by_scene: Counter[str] = Counter()
    scene_hash_by_id: dict[str, str] = {}
    scene_id_by_hash: dict[str, str] = {}
    seen_scene_trajectories: set[tuple[str, str]] = set()
    seen_trajectory_sha256: set[str] = set()
    for item in clusters:
        scene_id = str(item["scene_id"])
        scene_sha256 = str(item["scene_sha256"])
        trajectory_id = str(item["trajectory_id"])
        trajectory_sha256 = str(item["trajectory_sha256"])
        existing_scene_hash = scene_hash_by_id.setdefault(
            scene_id,
            scene_sha256,
        )
        if existing_scene_hash != scene_sha256:
            raise ValueError(
                "formal comparison maps one scene_id to multiple scene asset "
                f"hashes: {scene_id}"
            )
        existing_scene_id = scene_id_by_hash.setdefault(
            scene_sha256,
            scene_id,
        )
        if existing_scene_id != scene_id:
            raise ValueError(
                "formal comparison aliases one scene asset hash under "
                f"multiple scene IDs: {existing_scene_id}, {scene_id}"
            )
        pair = (scene_id, trajectory_id)
        if pair in seen_scene_trajectories:
            raise ValueError(
                "formal comparison has duplicate scene/trajectory identity: "
                f"{scene_id}/{trajectory_id}"
            )
        if trajectory_sha256 in seen_trajectory_sha256:
            raise ValueError(
                "formal comparison reuses an input trajectory fingerprint: "
                f"{trajectory_sha256}"
            )
        seen_scene_trajectories.add(pair)
        seen_trajectory_sha256.add(trajectory_sha256)
        by_scene[scene_id] += 1
    if len(clusters) < 6:
        raise ValueError(
            "formal comparison requires at least 6 scene-trajectory clusters"
        )
    if len(by_scene) < 3:
        raise ValueError(
            "formal comparison requires at least 3 distinct scenes"
        )
    if len(scene_id_by_hash) < 3:
        raise ValueError(
            "formal comparison requires at least 3 distinct scene asset hashes"
        )
    underrepresented = sorted(
        scene_id for scene_id, count in by_scene.items() if count < 2
    )
    if underrepresented:
        raise ValueError(
            "formal comparison requires at least 2 trajectories per scene; "
            "underrepresented: "
            + ", ".join(underrepresented)
        )
    for side_name in ("baseline", "candidate"):
        if not all(
            bool(item[side_name]["formal_detection_contract_eligible"])
            for item in clusters
        ):
            raise ValueError(
                f"formal comparison requires embedded {side_name} detector "
                "contracts for every cluster"
            )
    frozen_values = (
        (
            {
                str(item["parameters_sha256"])
                for item in clusters
            },
            "evaluation parameters",
        ),
        (
            {
                str(item["baseline"]["ground_truth_generator_sha256"])
                for item in clusters
            },
            "ground-truth generator",
        ),
        (
            {
                str(item["baseline"]["algorithm_sha256"])
                for item in clusters
            },
            "baseline algorithm",
        ),
    )
    for values, label in frozen_values:
        if len(values) != 1:
            raise ValueError(
                f"formal comparison requires one frozen {label} hash"
            )
    candidate_algorithms = {
        str(item["candidate"]["algorithm_sha256"]) for item in clusters
    }
    if len(candidate_algorithms) != 1:
        raise ValueError(
            "formal comparison requires one frozen candidate algorithm hash"
        )


def _validate_paired_input_identity(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    cluster_id: str,
    manifest_scene_id: str,
    manifest_scene_sha256: str,
    manifest_trajectory_sha256: str,
) -> None:
    paired_fields = (
        "scene_id",
        "scene_sha256",
        "frames_metadata_sha256",
        "selected_frame_indices_sha256",
        "input_frame_hashes_sha256",
        "detection_run_contract_sha256",
        "semantic_gt_integrity_sha256",
    )
    for field in paired_fields:
        if baseline[field] != candidate[field]:
            raise ValueError(
                f"cluster {cluster_id!r} paired input {field} mismatch"
            )
    if baseline["scene_id"] != manifest_scene_id:
        raise ValueError(
            f"cluster {cluster_id!r} manifest scene_id does not match audit"
        )
    if baseline["scene_sha256"] != manifest_scene_sha256:
        raise ValueError(
            f"cluster {cluster_id!r} manifest scene_sha256 does not match audit"
        )
    if baseline["input_frame_hashes_sha256"] != manifest_trajectory_sha256:
        raise ValueError(
            f"cluster {cluster_id!r} manifest trajectory_sha256 does not "
            "match paired input frames"
        )


def _validate_algorithm_difference_scope(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    allowed_paths: Sequence[str],
    cluster_id: str,
) -> None:
    actual_paths = _algorithm_difference_paths(baseline, candidate)
    allowed_path_set = set(allowed_paths)
    undeclared = sorted(actual_paths - allowed_path_set)
    unchanged_declared = sorted(allowed_path_set - actual_paths)
    if undeclared or unchanged_declared:
        details = []
        if undeclared:
            details.append("undeclared=" + ",".join(undeclared))
        if unchanged_declared:
            details.append(
                "declared_but_unchanged=" + ",".join(unchanged_declared)
            )
        raise ValueError(
            f"cluster {cluster_id!r} algorithm contract difference paths "
            "do not exactly match the preregistered paths: "
            + "; ".join(details)
        )


def _validate_allowed_difference_paths(paths: Sequence[str]) -> None:
    split_paths: list[tuple[str, ...]] = []
    for path in paths:
        parts = tuple(path.split("."))
        if any(not part for part in parts):
            raise ValueError(
                f"invalid allowed algorithm difference path {path!r}"
            )
        split_paths.append(parts)
    for index, parts in enumerate(split_paths):
        for other_index, other in enumerate(split_paths):
            if index == other_index:
                continue
            if len(parts) < len(other) and other[: len(parts)] == parts:
                raise ValueError(
                    "allowed algorithm difference paths cannot contain "
                    f"ancestor/descendant pairs: {'.'.join(parts)!r}, "
                    f"{'.'.join(other)!r}"
                )


def _algorithm_difference_paths(
    baseline: Any,
    candidate: Any,
    *,
    prefix: str = "",
) -> set[str]:
    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        differences: set[str] = set()
        for key in sorted(set(baseline) | set(candidate)):
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if key not in baseline or key not in candidate:
                differences.add(child_path)
                continue
            differences.update(
                _algorithm_difference_paths(
                    baseline[key],
                    candidate[key],
                    prefix=child_path,
                )
            )
        return differences
    if baseline != candidate:
        if not prefix:
            raise ValueError("algorithm contracts must be objects")
        return {prefix}
    return set()


def _extract_numeric_metric(
    variant: Mapping[str, Any],
    source_path: Sequence[str],
) -> tuple[float | None, str | None]:
    current: Any = variant
    walked: list[str] = []
    for key in source_path:
        if not isinstance(current, Mapping):
            return None, f"{'.'.join(walked) or '<variant>'} is not an object"
        if current.get("available") is False:
            reason = current.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                reason = f"{'.'.join(walked) or '<variant>'} marked unavailable"
            return None, reason
        if key not in current:
            return None, f"missing metric path {'.'.join((*walked, key))}"
        current = current[key]
        walked.append(key)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None, f"metric {'.'.join(source_path)} is not numeric"
    value = float(current)
    if not math.isfinite(value):
        return None, f"metric {'.'.join(source_path)} is not finite"
    return value, None


def _required_string(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} {key} must be a non-empty string")
    return value.strip()


def _required_sha256(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> str:
    value = _required_string(payload, key, context)
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{context} {key} must be exactly 64 hex characters")
    return value.lower()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - fraction)
        + float(sorted_values[upper]) * fraction
    )


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {path}: {error}") from error


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    manifest = _strict_json_load(manifest_path)
    result = compare_manifest_payload(
        manifest,
        manifest_dir=manifest_path.parent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
