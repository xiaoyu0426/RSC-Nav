from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Iterable

try:
    from semantic_task_profile import CUP_TASK_PROFILE, normalize_label
except ModuleNotFoundError:  # Package import in tests.
    from .semantic_task_profile import CUP_TASK_PROFILE, normalize_label


TARGET_LABELS = set(CUP_TASK_PROFILE.target_aliases)
SUPPORT_LABELS = set(CUP_TASK_PROFILE.support_labels)


def build_online_planner_request(
    task_text: str,
    current_xz: tuple[float, float],
    tracks: Iterable[dict[str, Any]],
    memory_items: Iterable[dict[str, Any]],
    max_candidates: int = 32,
    max_support_candidates: int = 6,
    support_merge_radius_m: float = 1.25,
    target_labels: Iterable[str] | None = None,
    support_labels: Iterable[str] | None = None,
    target_label: str | None = None,
) -> dict[str, Any]:
    target_aliases = {
        normalize_label(value)
        for value in (
            TARGET_LABELS if target_labels is None else target_labels
        )
    }
    support_set = {
        normalize_label(value)
        for value in (
            SUPPORT_LABELS if support_labels is None else support_labels
        )
    }
    canonical_target = normalize_label(
        target_label or CUP_TASK_PROFILE.target_label
    )
    memory_by_semantic_id = {
        int(item["semantic_id"]): item
        for item in memory_items
        if item.get("semantic_id") is not None
    }
    candidates = []
    for track in tracks:
        label = canonical_label(
            str(track.get("label", "")),
            target_labels=target_aliases,
            target_label=canonical_target,
        )
        if label != canonical_target and label not in support_set:
            continue
        track_id = int(track["track_id"])
        position = list(track.get("position_3d") or [])
        if len(position) < 3:
            continue
        memory = memory_by_semantic_id.get(track_id, {})
        distance_m = (
            (float(position[0]) - float(current_xz[0])) ** 2
            + (float(position[2]) - float(current_xz[1])) ** 2
        ) ** 0.5
        confidence = float(memory.get("confidence", track.get("confidence", 0.0)))
        freshness = float(memory.get("freshness", 1.0))
        views = int(track.get("views", 0))
        status = str(memory.get("status", "active"))
        kind = "target_object" if label == canonical_target else "support_surface"
        base_priority = (
            2.0 * confidence
            + 0.10 * min(8, views)
            + 0.35 * freshness
            - 0.05 * distance_m
            - {"active": 0.0, "stale": 0.55, "missing": 1.5}.get(status, 0.4)
            + (1.0 if kind == "target_object" else 0.0)
        )
        candidates.append(
            {
                "id": f"track_{track_id}",
                "track_id": track_id,
                "kind": kind,
                "label": label,
                "world_xz": [round(float(position[0]), 4), round(float(position[2]), 4)],
                "confidence": round(confidence, 4),
                "freshness": round(freshness, 4),
                "status": status,
                "independent_views": views,
                "negative_evidence_count": int(
                    memory.get("negative_evidence_count", 0)
                ),
                "distance_m": round(distance_m, 4),
                "local_priority": round(base_priority, 4),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["kind"] == "target_object",
            float(item["local_priority"]),
        ),
        reverse=True,
    )
    target_candidates = [
        item for item in candidates if item["kind"] == "target_object"
    ]
    support_candidates = _spatially_diverse_candidates(
        [
            item
            for item in candidates
            if item["kind"] == "support_surface"
        ],
        radius_m=float(support_merge_radius_m),
        limit=max(0, int(max_support_candidates)),
    )
    candidates = (target_candidates + support_candidates)[
        : max(1, int(max_candidates))
    ]
    return {
        "schema_version": "phase5a_online_task_request_v1",
        "task_text": str(task_text),
        "current_xz": [round(float(current_xz[0]), 4), round(float(current_xz[1]), 4)],
        "target_label": canonical_target,
        "target_labels": sorted(target_aliases),
        "candidate_landmarks": candidates,
        "required_output_schema": {
            "task_plan": [
                {
                    "step": 1,
                    "intent": "inspect_candidate",
                    "target_id": "track_id",
                }
            ],
            "ordered_candidate_ids": ["track_id"],
            "stop_probability": 0.0,
            "stop_condition": "short condition",
            "reason": "short audit explanation",
        },
        "constraints": [
            "Return JSON only.",
            "Use only candidate_landmarks ids.",
            "Do not output low-level movement actions.",
            "Inspect target_object candidates before support_surface fallbacks.",
            "Do not treat a stale or missing memory as a confirmed object.",
            "The traditional BEV/navmesh executor validates and executes each target.",
            "The robot must re-observe candidates before reporting success.",
        ],
    }


def plan_online_task(
    request_payload: dict[str, Any],
    mode: str,
    api_base: str,
    api_key: str,
    model: str,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode_requested = str(mode)
    mode_used = (
        "api"
        if mode_requested == "api" or (mode_requested == "auto" and bool(api_key))
        else "deterministic"
    )
    prompt = build_online_planner_prompt(request_payload)
    raw_response = None
    api_error = None
    started = time.perf_counter()
    if mode_used == "api":
        try:
            raw_response = call_openai_compatible_api(
                prompt=prompt,
                api_base=api_base,
                api_key=api_key,
                model=model,
                timeout_s=timeout_s,
            )
            output = extract_planner_output(raw_response)
        except Exception as exc:
            if mode_requested == "api":
                raise
            api_error = str(exc)
            mode_used = "deterministic"
            output = deterministic_online_plan(request_payload)
    else:
        output = deterministic_online_plan(request_payload)
    normalized = normalize_online_plan(output, request_payload)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    metadata = {
        "mode_requested": mode_requested,
        "mode_used": mode_used,
        "model": str(model),
        "api_base": str(api_base),
        "latency_ms": round(elapsed_ms, 3),
        "api_error": api_error,
        "prompt": prompt,
        "raw_response": raw_response,
    }
    return normalized, metadata


def build_online_planner_prompt(payload: dict[str, Any]) -> str:
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are the semantic task planner for an indoor robot. "
        "Order the supplied semantic-memory candidates for the user's task. "
        "Use scene_semantics when present to distinguish room types and support "
        "roles, such as a bathroom sink or dining table, and to prioritize likely "
        "search locations by grounded visual confidence. "
        "Direct target-object evidence normally precedes support-surface hypotheses. "
        "The robot will navigate, face, observe, update memory, and skip failed "
        "candidates using a separate traditional navigation executor. "
        "A semantic likelihood is a search prior and never proof that an object "
        "exists; final success requires independent re-observation. "
        "Return compact JSON only and never invent candidate ids.\n\n"
        f"INPUT_JSON:\n{compact}"
    )


def deterministic_online_plan(
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    candidates = list(request_payload.get("candidate_landmarks", []))
    candidates.sort(
        key=lambda item: (
            str(item.get("kind")) == "target_object",
            (
                0.65 * float(item.get("local_priority", 0.0))
                + 1.35
                * float(
                    item.get("scene_semantics", {}).get(
                        "target_likelihood",
                        0.0,
                    )
                )
                + 0.45
                * float(
                    item.get("scene_semantics", {}).get(
                        "visual_confidence",
                        0.0,
                    )
                )
            ),
        ),
        reverse=True,
    )
    ordered_ids = [str(item["id"]) for item in candidates]
    return {
        "task_plan": [
            {
                "step": index,
                "intent": (
                    "reobserve_target"
                    if item.get("kind") == "target_object"
                    else "inspect_support_surface"
                ),
                "target_id": str(item["id"]),
            }
            for index, item in enumerate(candidates, start=1)
        ],
        "ordered_candidate_ids": ordered_ids,
        "stop_probability": 0.05 if ordered_ids else 0.95,
        "stop_condition": (
            "all supplied target memories and support-surface fallbacks inspected"
        ),
        "reason": (
            "Targets are ranked by confidence, freshness, independent views, "
            "distance, memory status, and grounded scene-location priors; support "
            "surfaces remain unverified search hypotheses."
        ),
    }


def normalize_online_plan(
    output: dict[str, Any],
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    candidates = list(request_payload.get("candidate_landmarks", []))
    valid_ids = [str(item["id"]) for item in candidates]
    valid_id_set = set(valid_ids)
    model_order = [
        str(value)
        for value in output.get("ordered_candidate_ids", [])
        if str(value) in valid_id_set
    ]
    order = _unique(model_order)
    deterministic_order = deterministic_online_plan(request_payload)[
        "ordered_candidate_ids"
    ]
    order.extend(value for value in deterministic_order if value not in order)

    task_plan = []
    for item in output.get("task_plan", []):
        if not isinstance(item, dict):
            continue
        target_id = str(item.get("target_id") or item.get("target") or "")
        if target_id not in valid_id_set:
            continue
        task_plan.append(
            {
                "step": len(task_plan) + 1,
                "intent": str(item.get("intent", "inspect_candidate")),
                "target_id": target_id,
            }
        )
    planned_targets = {item["target_id"] for item in task_plan}
    for target_id in order:
        if target_id in planned_targets:
            continue
        candidate = next(item for item in candidates if str(item["id"]) == target_id)
        task_plan.append(
            {
                "step": len(task_plan) + 1,
                "intent": (
                    "reobserve_target"
                    if candidate.get("kind") == "target_object"
                    else "inspect_support_surface"
                ),
                "target_id": target_id,
            }
        )
    return {
        "task_plan": task_plan,
        "ordered_candidate_ids": order,
        "stop_probability": round(
            max(0.0, min(1.0, float(output.get("stop_probability", 0.0)))),
            4,
        ),
        "stop_condition": str(output.get("stop_condition", ""))[:1200],
        "reason": str(output.get("reason", ""))[:1200],
    }


def call_openai_compatible_api(
    prompt: str,
    api_base: str,
    api_key: str,
    model: str,
    timeout_s: float,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("API key is missing")
    body = {
        "model": str(model),
        "messages": [
            {
                "role": "system",
                "content": "Return valid compact JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    request = urllib.request.Request(
        str(api_base).rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {detail[:1000]}") from exc


def extract_planner_output(raw_response: dict[str, Any]) -> dict[str, Any]:
    choices = raw_response.get("choices") or []
    if not choices:
        raise RuntimeError("API response has no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("API response content is empty")
    cleaned = str(content).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        output = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        output = json.loads(match.group(0))
    if not isinstance(output, dict):
        raise ValueError("Planner output must be a JSON object")
    return output


def canonical_label(
    label: str,
    *,
    target_labels: Iterable[str] | None = None,
    target_label: str = "cup",
) -> str:
    normalized = normalize_label(label)
    aliases = {
        normalize_label(value)
        for value in (
            TARGET_LABELS if target_labels is None else target_labels
        )
    }
    return normalize_label(target_label) if normalized in aliases else normalized


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _spatially_diverse_candidates(
    candidates: Iterable[dict[str, Any]],
    radius_m: float,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    radius = max(0.0, float(radius_m))
    for candidate in candidates:
        position = candidate.get("world_xz") or []
        if len(position) < 2:
            continue
        if any(
            (
                (float(position[0]) - float(existing["world_xz"][0])) ** 2
                + (float(position[1]) - float(existing["world_xz"][1])) ** 2
            )
            ** 0.5
            < radius
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max(0, int(limit)):
            break
    return selected
