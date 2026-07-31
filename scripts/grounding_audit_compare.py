#!/usr/bin/env python3
"""Compare paired grounding audits across scene-trajectory clusters.

The manifest schema is::

    {
      "schema_version": "grounding_audit_compare_manifest_v1",
      "clusters": [
        {
          "cluster_id": "scene_a/trajectory_01",
          "baseline": {
            "audit_json": "baseline/audit.json",
            "variant": "baseline",
            "parameters_sha256": "<64 hex characters>",
            "ground_truth_sha256": "<64 hex characters>"
          },
          "candidate": {
            "audit_json": "candidate/audit.json",
            "variant": "candidate",
            "parameters_sha256": "<same parameters hash>",
            "ground_truth_sha256": "<same ground-truth hash>"
          }
        }
      ]
    }

Audit paths are resolved relative to the manifest. The comparator reads only
the named variants and preregistered numeric metric paths; it never reads VLM
verdicts or frame-level records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "grounding_audit_compare_v1"
MANIFEST_SCHEMA_VERSION = "grounding_audit_compare_manifest_v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_731
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

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

    manifest_dir = manifest_dir.expanduser().resolve()
    audit_cache: dict[Path, tuple[Mapping[str, Any], str]] = {}
    cluster_results: list[dict[str, Any]] = []
    seen_cluster_ids: set[str] = set()
    unavailable_metrics: list[dict[str, str]] = []

    for cluster_index, raw_cluster in enumerate(raw_clusters):
        context = f"clusters[{cluster_index}]"
        if not isinstance(raw_cluster, Mapping):
            raise ValueError(f"{context} must be an object")
        cluster_id = _required_string(raw_cluster, "cluster_id", context)
        if cluster_id in seen_cluster_ids:
            raise ValueError(f"duplicate cluster_id: {cluster_id!r}")
        seen_cluster_ids.add(cluster_id)

        baseline = _load_side(
            raw_cluster,
            "baseline",
            cluster_id=cluster_id,
            manifest_dir=manifest_dir,
            audit_cache=audit_cache,
        )
        candidate = _load_side(
            raw_cluster,
            "candidate",
            cluster_id=cluster_id,
            manifest_dir=manifest_dir,
            audit_cache=audit_cache,
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
                "parameters_sha256": baseline["parameters_sha256"],
                "ground_truth_sha256": baseline["ground_truth_sha256"],
                "baseline": _side_provenance(baseline),
                "candidate": _side_provenance(candidate),
                "metrics": metric_results,
            }
        )

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
        "cluster_count": len(cluster_results),
        "bootstrap": {
            "unit": "scene_trajectory_cluster_pair",
            "frame_level_resampling": False,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "interval": "percentile",
            "percentile_method": "linear_interpolation",
        },
        "metric_definitions": metric_definitions,
        "clusters": cluster_results,
        "comparisons": comparisons,
        "unavailable_metrics": unavailable_metrics,
    }


def _load_side(
    cluster: Mapping[str, Any],
    side_name: str,
    *,
    cluster_id: str,
    manifest_dir: Path,
    audit_cache: dict[Path, tuple[Mapping[str, Any], str]],
) -> dict[str, Any]:
    side = cluster.get(side_name)
    context = f"cluster {cluster_id!r} {side_name}"
    if not isinstance(side, Mapping):
        raise ValueError(f"{context} must be an object")

    audit_json = _required_string(side, "audit_json", context)
    variant_name = _required_string(side, "variant", context)
    parameters_sha256 = _required_sha256(side, "parameters_sha256", context)
    ground_truth_sha256 = _required_sha256(
        side,
        "ground_truth_sha256",
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
    return {
        "audit_json": audit_json,
        "audit_json_sha256": audit_sha256,
        "variant": variant_name,
        "variant_payload": variant_payload,
        "parameters_sha256": parameters_sha256,
        "ground_truth_sha256": ground_truth_sha256,
    }


def _side_provenance(side: Mapping[str, Any]) -> dict[str, str]:
    return {
        "audit_json": str(side["audit_json"]),
        "audit_json_sha256": str(side["audit_json_sha256"]),
        "variant": str(side["variant"]),
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


def _paired_bootstrap_ci(cluster_values: Sequence[float]) -> dict[str, float]:
    if not cluster_values:
        raise ValueError("paired bootstrap requires at least one cluster")
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
