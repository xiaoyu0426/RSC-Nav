from __future__ import annotations

import base64
import json
import math
import mimetypes
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SUPPORT_PRIORS = {
    "sink": 0.78,
    "counter": 0.70,
    "table": 0.64,
}


def select_scene_keyframes(
    frames: Iterable[dict[str, Any]],
    detections: Iterable[dict[str, Any]],
    candidate_ids: Iterable[str],
    max_images: int = 8,
) -> list[dict[str, Any]]:
    """Select candidate-bearing views first, then fill with temporal scene context."""
    frame_items = [
        item
        for item in frames
        if item.get("rgb_path") and item.get("frame_index") is not None
    ]
    if not frame_items or int(max_images) <= 0:
        return []

    valid_candidates = {str(value) for value in candidate_ids}
    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    for detection in detections:
        track_id = detection.get("online_track_id")
        frame_index = detection.get("frame_index")
        if track_id is None or frame_index is None:
            continue
        candidate_id = f"track_{int(track_id)}"
        if candidate_id not in valid_candidates:
            continue
        detections_by_frame.setdefault(int(frame_index), []).append(detection)

    selected_indices: list[int] = []
    uncovered = set(valid_candidates)
    while uncovered and len(selected_indices) < int(max_images):
        best: tuple[int, float, int] | None = None
        for frame_index, frame_detections in detections_by_frame.items():
            if frame_index in selected_indices:
                continue
            visible = {
                f"track_{int(item['online_track_id'])}"
                for item in frame_detections
            }
            newly_covered = visible & uncovered
            if not newly_covered:
                continue
            quality = sum(_detection_view_quality(item) for item in frame_detections)
            score = (len(newly_covered), quality, -frame_index)
            if best is None or score > best:
                best = score
                selected_frame = frame_index
        if best is None:
            break
        selected_indices.append(selected_frame)
        uncovered.difference_update(
            f"track_{int(item['online_track_id'])}"
            for item in detections_by_frame.get(selected_frame, [])
        )

    remaining_slots = int(max_images) - len(selected_indices)
    for item in _pose_diverse_frames(
        frame_items,
        selected_indices=selected_indices,
        count=remaining_slots,
    ):
        selected_indices.append(int(item["frame_index"]))

    frame_by_index = {
        int(item["frame_index"]): item
        for item in frame_items
    }
    result = []
    for frame_index in sorted(set(selected_indices)):
        frame = frame_by_index.get(frame_index)
        if frame is None:
            continue
        frame_detections = detections_by_frame.get(frame_index, [])
        visible_ids = sorted(
            {
                f"track_{int(item['online_track_id'])}"
                for item in frame_detections
            }
        )
        result.append(
            {
                "frame_id": f"frame_{frame_index:04d}",
                "frame_index": frame_index,
                "rgb_path": str(frame["rgb_path"]),
                "visible_candidate_ids": visible_ids,
                "selection_reason": (
                    "candidate_evidence" if visible_ids else "scene_context"
                ),
            }
        )
    return result[: int(max_images)]


def understand_scene_with_vlm(
    task_text: str,
    candidates: Iterable[dict[str, Any]],
    keyframes: Iterable[dict[str, Any]],
    mode: str,
    api_base: str,
    api_key: str,
    model: str,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_items = list(candidates)
    keyframe_items = list(keyframes)
    mode_requested = str(mode)
    mode_used = (
        "api"
        if mode_requested == "api" or (mode_requested == "auto" and bool(api_key))
        else "deterministic"
    )
    prompt = build_scene_vlm_prompt(task_text, candidate_items, keyframe_items)
    raw_response = None
    api_error = None
    started = time.perf_counter()
    if mode_used == "api":
        try:
            raw_response = call_openai_compatible_vlm(
                prompt=prompt,
                keyframes=keyframe_items,
                api_base=api_base,
                api_key=api_key,
                model=model,
                timeout_s=timeout_s,
            )
            raw_output = extract_json_output(raw_response)
        except Exception as exc:
            if mode_requested == "api":
                raise
            api_error = str(exc)
            mode_used = "deterministic"
            raw_output = deterministic_scene_understanding(candidate_items)
    else:
        raw_output = deterministic_scene_understanding(candidate_items)
    normalized = normalize_scene_understanding(
        raw_output,
        candidates=candidate_items,
        valid_frame_ids=[item["frame_id"] for item in keyframe_items],
    )
    metadata = {
        "mode_requested": mode_requested,
        "mode_used": mode_used,
        "model": str(model),
        "api_base": str(api_base),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "api_error": api_error,
        "prompt": prompt,
        "raw_response": raw_response,
        "keyframes": keyframe_items,
    }
    return normalized, metadata


def build_scene_vlm_prompt(
    task_text: str,
    candidates: Iterable[dict[str, Any]],
    keyframes: Iterable[dict[str, Any]],
) -> str:
    candidate_items = list(candidates)
    frame_manifest = [
        {
            "frame_id": item["frame_id"],
            "visible_candidate_ids": item.get("visible_candidate_ids", []),
        }
        for item in keyframes
    ]
    payload = {
        "task_text": str(task_text),
        "candidate_landmarks": candidate_items,
        "frame_manifest": frame_manifest,
        "required_output_schema": {
            "scene_summary": "short description of observed room/zone layout",
            "regions": [
                {
                    "region_id": "short stable name",
                    "room_type": "bathroom, dining_room, living_room, kitchen, or unknown",
                    "confidence": 0.0,
                    "anchor_candidate_ids": ["track_id from candidate_landmarks"],
                    "evidence_frame_ids": ["frame_0000"],
                }
            ],
            "candidate_assessments": [
                {
                    "candidate_id": "track_id from candidate_landmarks",
                    "room_type": "observed room type or unknown",
                    "support_role": "sink, dining table, coffee table, counter, etc.",
                    "target_likelihood": 0.0,
                    "visual_confidence": 0.0,
                    "evidence_frame_ids": ["frame_0000"],
                    "reason": "short visual and commonsense justification",
                }
            ],
            "search_hypotheses": [
                {
                    "candidate_id": "track_id from candidate_landmarks",
                    "hypothesis": "what should be checked here",
                }
            ],
        },
    }
    return (
        "You are the visual scene-understanding module for an indoor search robot. "
        "Use the supplied chronological RGB keyframes and annotated candidate ids "
        "to infer room types, support-surface roles, and where the requested object "
        "is likely to be found. A likelihood is only a search prior, never proof that "
        "an object exists. Ground every assessment in supplied frames, use only given "
        "candidate ids and frame ids, and lower visual_confidence when context is "
        "ambiguous. Return compact JSON only.\n\n"
        f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def deterministic_scene_understanding(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    assessments = []
    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        label = str(candidate.get("label", "")).lower()
        kind = str(candidate.get("kind", ""))
        if kind == "target_object":
            likelihood = max(0.75, float(candidate.get("confidence", 0.0)))
            role = "direct_target_hypothesis"
        else:
            likelihood = DEFAULT_SUPPORT_PRIORS.get(label, 0.45)
            role = label or "unknown_support"
        assessments.append(
            {
                "candidate_id": candidate_id,
                "room_type": "unknown",
                "support_role": role,
                "target_likelihood": likelihood,
                "visual_confidence": 0.0,
                "evidence_frame_ids": [],
                "reason": (
                    "Deterministic label prior; no visual scene interpretation was used."
                ),
            }
        )
    return {
        "scene_summary": "Visual scene understanding unavailable; label priors only.",
        "regions": [],
        "candidate_assessments": assessments,
        "search_hypotheses": [],
    }


def normalize_scene_understanding(
    output: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    valid_frame_ids: Iterable[str],
) -> dict[str, Any]:
    candidate_items = list(candidates)
    valid_candidates = {
        str(item["id"]): item
        for item in candidate_items
        if item.get("id") is not None
    }
    valid_frames = {str(value) for value in valid_frame_ids}
    supplied_assessments = {}
    for assessment in output.get("candidate_assessments", []):
        if not isinstance(assessment, dict):
            continue
        candidate_id = str(
            assessment.get("candidate_id") or assessment.get("target_id") or ""
        )
        if candidate_id in valid_candidates and candidate_id not in supplied_assessments:
            supplied_assessments[candidate_id] = assessment

    fallback = deterministic_scene_understanding(candidate_items)
    fallback_by_id = {
        item["candidate_id"]: item
        for item in fallback["candidate_assessments"]
    }
    assessments = []
    for candidate_id, candidate in valid_candidates.items():
        raw = supplied_assessments.get(candidate_id, fallback_by_id[candidate_id])
        evidence_frames = _valid_unique_ids(
            raw.get("evidence_frame_ids", []),
            valid_frames,
        )
        assessments.append(
            {
                "candidate_id": candidate_id,
                "kind": str(candidate.get("kind", "")),
                "label": str(candidate.get("label", "")),
                "room_type": _short_text(raw.get("room_type", "unknown"), 80),
                "support_role": _short_text(
                    raw.get("support_role", candidate.get("label", "unknown")),
                    120,
                ),
                "target_likelihood": _probability(
                    raw.get(
                        "target_likelihood",
                        fallback_by_id[candidate_id]["target_likelihood"],
                    )
                ),
                "visual_confidence": _probability(
                    raw.get("visual_confidence", 0.0)
                ),
                "evidence_frame_ids": evidence_frames,
                "reason": _short_text(raw.get("reason", ""), 800),
            }
        )

    regions = []
    candidate_positions = {
        str(item["id"]): [
            float(item["world_xz"][0]),
            float(item["world_xz"][1]),
        ]
        for item in candidate_items
        if item.get("id") is not None
        and len(item.get("world_xz") or []) >= 2
    }
    for raw in output.get("regions", []):
        if not isinstance(raw, dict):
            continue
        region_id = _short_text(raw.get("region_id", ""), 80)
        if not region_id:
            continue
        anchor_candidate_ids = _valid_unique_ids(
            raw.get("anchor_candidate_ids", []),
            set(valid_candidates),
        )
        anchor_positions = [
            candidate_positions[candidate_id]
            for candidate_id in anchor_candidate_ids
            if candidate_id in candidate_positions
        ]
        anchor_xz = (
            [
                round(
                    sum(position[axis] for position in anchor_positions)
                    / len(anchor_positions),
                    4,
                )
                for axis in range(2)
            ]
            if anchor_positions
            else None
        )
        regions.append(
            {
                "region_id": region_id,
                "room_type": _short_text(raw.get("room_type", "unknown"), 80),
                "confidence": _probability(raw.get("confidence", 0.0)),
                "anchor_candidate_ids": anchor_candidate_ids,
                "anchor_xz": anchor_xz,
                "evidence_frame_ids": _valid_unique_ids(
                    raw.get("evidence_frame_ids", []),
                    valid_frames,
                ),
            }
        )
    if not regions:
        grouped_assessments: dict[str, list[dict[str, Any]]] = {}
        for assessment in assessments:
            room_type = str(assessment.get("room_type", "unknown"))
            if room_type and room_type != "unknown":
                grouped_assessments.setdefault(room_type, []).append(assessment)
        for room_type, room_items in grouped_assessments.items():
            anchor_candidate_ids = [
                str(item["candidate_id"])
                for item in room_items
                if str(item["candidate_id"]) in candidate_positions
            ]
            anchor_positions = [
                candidate_positions[candidate_id]
                for candidate_id in anchor_candidate_ids
            ]
            if not anchor_positions:
                continue
            regions.append(
                {
                    "region_id": f"{room_type}_derived",
                    "room_type": room_type,
                    "confidence": round(
                        sum(
                            float(item["visual_confidence"])
                            for item in room_items
                        )
                        / len(room_items),
                        4,
                    ),
                    "anchor_candidate_ids": anchor_candidate_ids,
                    "anchor_xz": [
                        round(
                            sum(
                                position[axis]
                                for position in anchor_positions
                            )
                            / len(anchor_positions),
                            4,
                        )
                        for axis in range(2)
                    ],
                    "evidence_frame_ids": _unique_strings(
                        frame_id
                        for item in room_items
                        for frame_id in item["evidence_frame_ids"]
                    ),
                }
            )

    hypotheses = []
    for raw in output.get("search_hypotheses", []):
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id", ""))
        if candidate_id not in valid_candidates:
            continue
        hypotheses.append(
            {
                "candidate_id": candidate_id,
                "hypothesis": _short_text(raw.get("hypothesis", ""), 500),
            }
        )
    return {
        "schema_version": "scene_semantic_search_v1",
        "scene_summary": _short_text(output.get("scene_summary", ""), 1600),
        "regions": regions,
        "candidate_assessments": assessments,
        "search_hypotheses": hypotheses,
        "contract": {
            "likelihood_is_search_prior_not_object_confirmation": True,
            "candidate_ids_restricted_to_observed_memory": True,
            "evidence_frames_restricted_to_selected_keyframes": True,
        },
    }


def enrich_planner_request(
    request_payload: dict[str, Any],
    scene_understanding: dict[str, Any],
) -> dict[str, Any]:
    enriched = json.loads(json.dumps(request_payload))
    assessments = {
        str(item["candidate_id"]): item
        for item in scene_understanding.get("candidate_assessments", [])
    }
    for candidate in enriched.get("candidate_landmarks", []):
        assessment = assessments.get(str(candidate.get("id")))
        if assessment is None:
            continue
        candidate["scene_semantics"] = {
            "room_type": assessment.get("room_type", "unknown"),
            "support_role": assessment.get("support_role", candidate.get("label")),
            "target_likelihood": assessment.get("target_likelihood", 0.0),
            "visual_confidence": assessment.get("visual_confidence", 0.0),
            "evidence_frame_ids": assessment.get("evidence_frame_ids", []),
            "reason": assessment.get("reason", ""),
        }
    enriched["schema_version"] = "phase5a_online_task_request_v2"
    enriched["scene_understanding"] = scene_understanding
    enriched.setdefault("constraints", []).extend(
        [
            "Use target_likelihood as a search prior, never as object confirmation.",
            "Prefer semantically plausible room/support combinations when direct target evidence is absent.",
            "Preserve uncertainty and exhaustive fallback coverage.",
        ]
    )
    return enriched


def rank_search_candidates(
    candidates: Iterable[dict[str, Any]],
    current_xz: tuple[float, float],
    scene_understanding: dict[str, Any] | None = None,
    planner_order: Iterable[str] = (),
    completed_ids: Iterable[str] = (),
    failed_ids: Iterable[str] = (),
    attempts: dict[str, int] | None = None,
    active_candidate_id: str | None = None,
    beliefs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    assessments = {
        str(item.get("candidate_id")): item
        for item in (scene_understanding or {}).get("candidate_assessments", [])
    }
    planner_ids = [str(value) for value in planner_order]
    planner_rank = {
        candidate_id: index
        for index, candidate_id in enumerate(planner_ids)
    }
    completed = {str(value) for value in completed_ids}
    failed = {str(value) for value in failed_ids}
    attempt_counts = attempts or {}
    ranked = []
    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        if not candidate_id or candidate_id in completed or candidate_id in failed:
            continue
        kind = str(candidate.get("kind", "support_surface"))
        label = str(candidate.get("label", "")).lower()
        assessment = assessments.get(candidate_id, {})
        confidence = _probability(candidate.get("confidence", 0.0))
        freshness = _probability(candidate.get("freshness", 1.0))
        views = max(0, int(candidate.get("independent_views", candidate.get("views", 0))))
        world_xz = candidate.get("world_xz") or []
        distance_m = (
            math.hypot(
                float(world_xz[0]) - float(current_xz[0]),
                float(world_xz[1]) - float(current_xz[1]),
            )
            if len(world_xz) >= 2
            else float(candidate.get("distance_m", 0.0))
        )
        belief = (beliefs or {}).get(candidate_id, {})
        target_likelihood = _probability(
            belief.get(
                "posterior",
                assessment.get(
                    "target_likelihood",
                    (
                        max(0.75, confidence)
                        if kind == "target_object"
                        else DEFAULT_SUPPORT_PRIORS.get(label, 0.45)
                    ),
                ),
            )
        )
        visual_confidence = _probability(
            assessment.get("visual_confidence", 0.0)
        )
        negative_evidence = max(
            0,
            int(candidate.get("negative_evidence_count", 0)),
        )
        attempt_count = max(0, int(attempt_counts.get(candidate_id, 0)))
        semantic_reliability = 0.65 + 0.35 * visual_confidence
        breakdown = {
            "direct_target_bonus": (
                max(0.0, 2.0 - 0.90 * attempt_count)
                if kind == "target_object"
                else 0.0
            ),
            "semantic_location_posterior": (
                2.20 * target_likelihood * semantic_reliability
            ),
            "detector_or_memory_confidence": 0.85 * confidence,
            "view_stability": 0.45 * min(1.0, views / 6.0),
            "freshness": 0.30 * freshness,
            "distance_cost": -0.08 * max(0.0, distance_m),
            "negative_evidence_cost": -0.22 * min(6, negative_evidence),
            "retry_cost": -0.45 * attempt_count,
            "active_target_commitment": (
                0.75 if candidate_id == str(active_candidate_id) else 0.0
            ),
        }
        score = sum(breakdown.values())
        ranked.append(
            {
                "candidate_id": candidate_id,
                "kind": kind,
                "label": label,
                "score": round(score, 4),
                "distance_m": round(distance_m, 4),
                "target_likelihood": round(target_likelihood, 4),
                "prior": round(
                    _probability(
                        belief.get("prior", target_likelihood)
                    ),
                    4,
                ),
                "posterior": round(target_likelihood, 4),
                "visual_confidence": round(visual_confidence, 4),
                "room_type": str(assessment.get("room_type", "unknown")),
                "support_role": str(assessment.get("support_role", label)),
                "reason": str(assessment.get("reason", ""))[:800],
                "score_breakdown": {
                    key: round(value, 4)
                    for key, value in breakdown.items()
                },
            }
        )
    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            float(item["distance_m"]),
            planner_rank.get(str(item["candidate_id"]), len(planner_rank)),
            str(item["candidate_id"]),
        )
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def initialize_search_beliefs(
    candidates: Iterable[dict[str, Any]],
    scene_understanding: dict[str, Any] | None,
    step: int,
) -> dict[str, dict[str, Any]]:
    assessments = {
        str(item.get("candidate_id")): item
        for item in (scene_understanding or {}).get("candidate_assessments", [])
    }
    beliefs = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        if not candidate_id:
            continue
        kind = str(candidate.get("kind", "support_surface"))
        label = str(candidate.get("label", "")).lower()
        confidence = _probability(candidate.get("confidence", 0.0))
        prior = _probability(
            assessments.get(candidate_id, {}).get(
                "target_likelihood",
                (
                    max(0.75, confidence)
                    if kind == "target_object"
                    else DEFAULT_SUPPORT_PRIORS.get(label, 0.45)
                ),
            )
        )
        beliefs[candidate_id] = {
            "candidate_id": candidate_id,
            "kind": kind,
            "label": label,
            "prior": round(prior, 6),
            "posterior": round(prior, 6),
            "status": "pending",
            "attempts": 0,
            "positive_evidence_count": 0,
            "negative_evidence_count": 0,
            "evidence_event_ids": [],
            "last_outcome": "initialized",
            "last_update_step": int(step),
        }
    return beliefs


def apply_search_evidence(
    beliefs: dict[str, dict[str, Any]],
    candidate_id: str,
    event_id: str,
    outcome: str,
    step: int,
    observable: bool = True,
) -> dict[str, Any]:
    belief = beliefs.get(str(candidate_id))
    if belief is None:
        return {
            "candidate_id": str(candidate_id),
            "event_id": str(event_id),
            "applied": False,
            "reason": "unknown_candidate",
        }
    event_ids = belief.setdefault("evidence_event_ids", [])
    if str(event_id) in event_ids:
        return {
            "candidate_id": str(candidate_id),
            "event_id": str(event_id),
            "applied": False,
            "reason": "duplicate_event",
            "posterior": belief["posterior"],
        }

    prior_posterior = _probability(belief.get("posterior", belief.get("prior", 0.5)))
    normalized_outcome = str(outcome)
    posterior = prior_posterior
    if normalized_outcome in {"target_evidence_observed", "verified"}:
        if normalized_outcome == "verified":
            posterior = 0.995
            belief["status"] = "verified"
        else:
            posterior = _bayes_positive(prior_posterior, 0.85, 0.15)
            belief["status"] = "evidence_observed"
        belief["positive_evidence_count"] = (
            int(belief.get("positive_evidence_count", 0)) + 1
        )
    elif (
        normalized_outcome == "no_target_evidence_observed"
        or normalized_outcome.startswith("rejected_")
    ):
        if observable:
            posterior = (
                0.01
                if normalized_outcome.startswith("rejected_")
                else _bayes_negative(prior_posterior, 0.90, 0.90)
            )
            belief["negative_evidence_count"] = (
                int(belief.get("negative_evidence_count", 0)) + 1
            )
            belief["status"] = (
                "rejected"
                if normalized_outcome.startswith("rejected_")
                else "searched_no_evidence"
            )
        else:
            normalized_outcome = "inconclusive_not_observable"
            belief["status"] = "pending"
    elif normalized_outcome.startswith("inconclusive"):
        posterior = max(0.01, 0.82 * prior_posterior)
        belief["status"] = "inconclusive"

    event_ids.append(str(event_id))
    belief["posterior"] = round(_probability(posterior), 6)
    belief["attempts"] = int(belief.get("attempts", 0)) + 1
    belief["last_outcome"] = normalized_outcome
    belief["last_update_step"] = int(step)
    return {
        "candidate_id": str(candidate_id),
        "event_id": str(event_id),
        "applied": True,
        "outcome": normalized_outcome,
        "observable": bool(observable),
        "prior_posterior": round(prior_posterior, 6),
        "posterior": belief["posterior"],
        "belief": dict(belief),
    }


def call_openai_compatible_vlm(
    prompt: str,
    keyframes: Iterable[dict[str, Any]],
    api_base: str,
    api_key: str,
    model: str,
    timeout_s: float,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("API key is missing")
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for keyframe in keyframes:
        content.append(
            {
                "type": "text",
                "text": (
                    f"IMAGE {keyframe['frame_id']} "
                    f"(visible candidates: "
                    f"{', '.join(keyframe.get('visible_candidate_ids', [])) or 'none'})"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_uri(keyframe["rgb_path"])},
            }
        )
    body = {
        "model": str(model),
        "messages": [
            {
                "role": "system",
                "content": "Return valid compact JSON only.",
            },
            {"role": "user", "content": content},
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
        raise RuntimeError(f"VLM API HTTP {exc.code}: {detail[:1000]}") from exc


def extract_json_output(raw_response: dict[str, Any]) -> dict[str, Any]:
    choices = raw_response.get("choices") or []
    if not choices:
        raise RuntimeError("VLM API response has no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("VLM API response content is empty")
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
        raise ValueError("VLM output must be a JSON object")
    return output


def _detection_view_quality(detection: dict[str, Any]) -> float:
    box = detection.get("box") or []
    area = 0.0
    if len(box) >= 4:
        area = max(0.0, float(box[2]) - float(box[0])) * max(
            0.0,
            float(box[3]) - float(box[1]),
        )
    return max(0.0, float(detection.get("score", 0.0))) * math.sqrt(area + 1.0)


def _evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    indices = {
        int(round(index * (len(items) - 1) / max(1, count - 1)))
        for index in range(count)
    }
    return [items[index] for index in sorted(indices)]


def _pose_diverse_frames(
    frames: list[dict[str, Any]],
    selected_indices: Iterable[int],
    count: int,
) -> list[dict[str, Any]]:
    selected_set = {int(value) for value in selected_indices}
    available = [
        item
        for item in frames
        if int(item["frame_index"]) not in selected_set
    ]
    if count <= 0 or not available:
        return []
    selected = [
        item
        for item in frames
        if int(item["frame_index"]) in selected_set
    ]
    result = []
    while available and len(result) < int(count):
        references = selected + result
        if not references:
            best = available[0]
        else:
            best = max(
                available,
                key=lambda item: min(
                    _frame_pose_distance(item, reference)
                    for reference in references
                ),
            )
        result.append(best)
        available.remove(best)
    return result


def _frame_pose_distance(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    first_position = first.get("agent_position_xyz") or []
    second_position = second.get("agent_position_xyz") or []
    if len(first_position) >= 3 and len(second_position) >= 3:
        translation_m = math.hypot(
            float(first_position[0]) - float(second_position[0]),
            float(first_position[2]) - float(second_position[2]),
        )
    else:
        translation_m = (
            abs(int(first["frame_index"]) - int(second["frame_index"])) / 50.0
        )
    first_yaw = _frame_yaw_deg(first)
    second_yaw = _frame_yaw_deg(second)
    yaw_distance = (
        abs((first_yaw - second_yaw + 180.0) % 360.0 - 180.0) / 90.0
        if first_yaw is not None and second_yaw is not None
        else 0.0
    )
    temporal_distance = (
        abs(int(first["frame_index"]) - int(second["frame_index"])) / 200.0
    )
    return translation_m / 1.5 + yaw_distance + 0.15 * temporal_distance


def _frame_yaw_deg(frame: dict[str, Any]) -> float | None:
    matrix = frame.get("agent_rotation_matrix") or []
    if (
        not isinstance(matrix, list)
        or len(matrix) < 3
        or not isinstance(matrix[0], list)
        or len(matrix[0]) < 3
        or not isinstance(matrix[2], list)
        or len(matrix[2]) < 3
    ):
        return None
    return math.degrees(
        math.atan2(float(matrix[0][2]), float(matrix[2][2]))
    )


def _bayes_positive(prior: float, sensitivity: float, false_positive: float) -> float:
    numerator = prior * sensitivity
    denominator = numerator + (1.0 - prior) * false_positive
    return numerator / max(1e-9, denominator)


def _bayes_negative(prior: float, sensitivity: float, specificity: float) -> float:
    numerator = prior * (1.0 - sensitivity)
    denominator = numerator + (1.0 - prior) * specificity
    return numerator / max(1e-9, denominator)


def _image_data_uri(path_value: str | Path) -> str:
    path = Path(path_value).expanduser().resolve()
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _probability(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _short_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[: max(0, int(limit))]


def _valid_unique_ids(values: Any, valid_ids: set[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        normalized = str(value)
        if normalized not in valid_ids or normalized in result:
            continue
        result.append(normalized)
    return result


def _unique_strings(values: Iterable[Any]) -> list[str]:
    result = []
    for value in values:
        normalized = str(value)
        if normalized and normalized not in result:
            result.append(normalized)
    return result
