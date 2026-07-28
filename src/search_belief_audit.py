from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_search_belief_run(
    run_dir: str | Path,
    require_api: bool = False,
    require_both_support_outcomes: bool = False,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    summary = _read_json(run_path / "online_summary.json")
    planner_request = _read_json(
        run_path / "task_planner" / "planner_request.json"
    )
    scene = summary.get("scene_understanding") or {}
    scene_vlm = summary.get("scene_vlm") or {}
    planner = summary.get("task_planner") or {}
    events = list(summary.get("task_plan_events", []))
    support_events = [
        item
        for item in events
        if item.get("event") == "support_surface_inspection_completed"
    ]
    candidate_ids = {
        str(item["id"])
        for item in planner_request.get("candidate_landmarks", [])
        if item.get("id") is not None
    }
    assessment_ids = {
        str(item.get("candidate_id"))
        for item in scene.get("candidate_assessments", [])
    }
    keyframes = list(scene_vlm.get("keyframes", []))
    task_injection_step = summary.get("task_injection_step")

    posterior_updates = [
        item.get("belief_update", {})
        for item in support_events
        if item.get("belief_update", {}).get("applied")
    ]
    positive_updates = [
        item
        for item in posterior_updates
        if item.get("outcome") == "target_evidence_observed"
    ]
    negative_updates = [
        item
        for item in posterior_updates
        if item.get("outcome") == "no_target_evidence_observed"
    ]
    confirmed_cups = list(summary.get("confirmed_cups", []))

    checks = {
        "causal_frames_only": bool(
            summary.get("causal_invariants", {}).get(
                "all_decisions_use_current_or_past_frames"
            )
        ),
        "task_hidden_until_memory_ready": bool(
            summary.get("causal_invariants", {}).get(
                "task_hidden_until_memory_ready"
            )
        ),
        "vlm_candidate_ids_grounded": bool(assessment_ids)
        and assessment_ids <= candidate_ids,
        "vlm_keyframes_causal": bool(keyframes)
        and task_injection_step is not None
        and all(
            int(item.get("frame_index", task_injection_step))
            <= int(task_injection_step)
            for item in keyframes
        ),
        "scene_regions_spatially_anchored": bool(scene.get("regions"))
        and all(
            len(item.get("anchor_xz") or []) == 2
            and bool(item.get("anchor_candidate_ids"))
            and set(map(str, item.get("anchor_candidate_ids", [])))
            <= candidate_ids
            for item in scene.get("regions", [])
        ),
        "support_inspection_completed": bool(support_events),
        "positive_updates_increase_posterior": all(
            float(item["posterior"]) > float(item["prior_posterior"])
            for item in positive_updates
        ),
        "negative_updates_decrease_posterior": all(
            float(item["posterior"]) < float(item["prior_posterior"])
            for item in negative_updates
        ),
        "search_priority_replanned": any(
            item.get("event") == "search_priority_replanned"
            for item in events
        ),
        "vlm_never_directly_confirms_object": bool(
            scene.get("contract", {}).get(
                "likelihood_is_search_prior_not_object_confirmation"
            )
        )
        and all(
            bool(item.get("confirmation", {}).get("verified"))
            for item in confirmed_cups
        ),
    }
    if require_api:
        checks["real_scene_vlm_api_used"] = scene_vlm.get("mode_used") == "api"
        checks["real_task_planner_api_used"] = planner.get("mode_used") == "api"
    if require_both_support_outcomes:
        checks["positive_support_outcome_observed"] = bool(positive_updates)
        checks["negative_support_outcome_observed"] = bool(negative_updates)

    failed_checks = [
        name
        for name, passed in checks.items()
        if not bool(passed)
    ]
    return {
        "schema_version": "phase5a_search_belief_audit_v1",
        "run_dir": str(run_path),
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "metrics": {
            "num_steps": int(summary.get("num_steps", 0)),
            "task_injection_step": task_injection_step,
            "task_execution_steps": int(
                summary.get("task_execution_steps", 0)
            ),
            "scene_vlm_mode": scene_vlm.get("mode_used"),
            "scene_vlm_model": scene_vlm.get("model"),
            "planner_mode": planner.get("mode_used"),
            "planner_model": planner.get("model"),
            "num_scene_regions": len(scene.get("regions", [])),
            "num_search_replans": sum(
                item.get("event") == "search_priority_replanned"
                for item in events
            ),
            "num_support_inspections": len(support_events),
            "num_positive_support_updates": len(positive_updates),
            "num_negative_support_updates": len(negative_updates),
            "num_confirmed_cups": int(
                summary.get("num_confirmed_cups", 0)
            ),
        },
        "support_updates": [
            {
                "step": item.get("step"),
                "candidate_id": item.get("candidate_id"),
                "outcome": item.get("outcome"),
                "observable_scan": item.get("observable_scan"),
                "prior_posterior": item.get(
                    "belief_update",
                    {},
                ).get("prior_posterior"),
                "posterior": item.get("belief_update", {}).get("posterior"),
                "observed_target_candidate_ids": item.get(
                    "observed_target_candidate_ids",
                    [],
                ),
            }
            for item in support_events
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value
