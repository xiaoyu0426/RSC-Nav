from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

try:
    from scene_semantic_search import call_openai_compatible_vlm
except ModuleNotFoundError:  # Package import in tests.
    from .scene_semantic_search import call_openai_compatible_vlm


def should_request_target_vlm(
    confirmation_result: dict[str, Any],
    *,
    min_task_views: int,
) -> bool:
    return (
        int(confirmation_result.get("task_independent_views", 0))
        >= int(min_task_views)
        and int(confirmation_result.get("geometry_inlier_views", 0))
        >= int(min_task_views)
        and str(confirmation_result.get("status", ""))
        not in {
            "rejected_geometry_inconsistent",
            "rejected_planar_surface",
        }
        and len(confirmation_result.get("crop_paths") or []) >= int(min_task_views)
    )


def verify_target_crops_with_vlm(
    *,
    target_label: str,
    candidate_id: str,
    crop_paths: Iterable[str | Path],
    api_base: str,
    api_key: str,
    model: str,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = [Path(value).expanduser().resolve() for value in crop_paths]
    keyframes = [
        {
            "frame_id": f"{candidate_id}_view_{index}",
            "rgb_path": str(path),
            "visible_candidate_ids": [candidate_id],
        }
        for index, path in enumerate(paths, start=1)
        if path.exists()
    ]
    if len(keyframes) < 2:
        raise ValueError("target VLM verification requires at least two crop views")

    prompt = build_target_vlm_prompt(
        target_label=target_label,
        candidate_id=candidate_id,
        view_ids=[item["frame_id"] for item in keyframes],
    )
    started = time.perf_counter()
    raw_response = call_openai_compatible_vlm(
        prompt=prompt,
        keyframes=keyframes,
        api_base=api_base,
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
    )
    result = extract_target_vlm_result(
        raw_response,
        candidate_id=candidate_id,
        target_label=target_label,
    )
    metadata = {
        "mode_used": "api",
        "model": str(model),
        "api_base": str(api_base),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "view_ids": [item["frame_id"] for item in keyframes],
    }
    return result, metadata


def build_target_vlm_prompt(
    *,
    target_label: str,
    candidate_id: str,
    view_ids: Iterable[str],
) -> str:
    target = str(target_label).strip().lower()
    views = ", ".join(str(value) for value in view_ids)
    if target == "door":
        definition = (
            "A door or doorway is a human-passable architectural opening, including "
            "an open door, hinged door, sliding glass door, or exterior entrance "
            "door. A normal window, cabinet or shelf, mirror, appliance door, wall "
            "panel, or fireplace is not a door. Adjacent windows may dominate a "
            "crop; decide whether the candidate itself provides a floor-level human "
            "passage. Interpret a partial view together with the clearer view."
        )
    else:
        definition = (
            f"Decide whether the physical object shared by the views is a real "
            f"{target}. Reject lookalikes, pictures, parts, and unrelated surfaces."
        )
    return (
        "You are the final visual verifier for an indoor navigation robot. "
        f"Candidate {candidate_id} has independent crop views {views}. {definition} "
        "Return compact JSON only with schema: "
        '{"candidate_id":"'
        + str(candidate_id)
        + '","verdict":"target|not_target|unclear","confidence":0.0,'
        '"observed_type":"short noun","reason":"short visual reason",'
        '"supporting_view_ids":["view id"]}.'
    )


def extract_target_vlm_result(
    raw_response: dict[str, Any],
    *,
    candidate_id: str,
    target_label: str,
) -> dict[str, Any]:
    choices = raw_response.get("choices") or []
    if not choices:
        raise RuntimeError("target VLM response has no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("target VLM response content is empty")
    cleaned = str(content).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
        items = parsed["candidates"]
    elif isinstance(parsed, dict):
        items = [parsed]
    else:
        raise ValueError("target VLM output must be an object or list")

    selected = next(
        (
            item
            for item in items
            if isinstance(item, dict)
            and str(item.get("candidate_id", "")) == str(candidate_id)
        ),
        items[0] if items and isinstance(items[0], dict) else None,
    )
    if selected is None:
        raise ValueError("target VLM output has no candidate result")

    verdict = _normalize_verdict(
        selected.get("verdict"),
        target_label=target_label,
    )
    confidence = _probability(selected.get("confidence", 0.0))
    return {
        "candidate_id": str(candidate_id),
        "verdict": verdict,
        "confidence": confidence,
        "observed_type": str(selected.get("observed_type", "")).strip(),
        "reason": str(selected.get("reason", "")).strip(),
        "supporting_view_ids": [
            str(value) for value in selected.get("supporting_view_ids", [])
        ],
    }


def apply_target_vlm_verdict(
    confirmation_result: dict[str, Any],
    vlm_result: dict[str, Any] | None,
    *,
    min_confidence: float,
) -> dict[str, Any]:
    result = dict(confirmation_result)
    if not vlm_result:
        return result
    verdict = str(vlm_result.get("verdict", "unclear"))
    confidence = _probability(vlm_result.get("confidence", 0.0))
    result["vlm_verification"] = dict(vlm_result)
    result["verification_source"] = "grounding_crop_plus_qwen_vlm"
    if confidence < float(min_confidence) or verdict == "unclear":
        result["status"] = "insufficient_vlm_evidence"
        result["verified"] = False
        return result
    if verdict == "target":
        result["status"] = "verified"
        result["verified"] = True
    elif verdict == "not_target":
        result["status"] = "rejected_vlm_verifier"
        result["verified"] = False
    return result


def _normalize_verdict(value: Any, *, target_label: str) -> str:
    normalized = " ".join(
        str(value).strip().lower().replace("-", " ").replace("_", " ").split()
    )
    target = str(target_label).strip().lower()
    if normalized in {"target", "yes", "true", target}:
        return "target"
    if normalized in {
        "not target",
        "no",
        "false",
        f"not {target}",
        f"not a {target}",
    }:
        return "not_target"
    return "unclear"


def _probability(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric > 1.0 and numeric <= 100.0:
        numeric /= 100.0
    return max(0.0, min(1.0, numeric))
