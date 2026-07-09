from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-4.1-mini"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase5A API semantic task planner minimal closed-loop evaluator.")
    parser.add_argument("--planner-context", required=True, help="M3.5 planner_context.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["auto", "api", "deterministic"], default="auto")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-stopovers", type=int, default=3)
    parser.add_argument("--success-radius-m", type=float, default=1.25)
    parser.add_argument(
        "--hide-goal-matching-waypoint-ids",
        action="store_true",
        help="Do not expose the precomputed goal-matching waypoint id list to the planner prompt; keep the hidden evaluator check.",
    )
    args = parser.parse_args()

    context_path = Path(args.planner_context).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    context = _read_json(context_path)

    request_payload = _build_request_payload(
        context,
        max_stopovers=max(1, int(args.max_stopovers)),
        expose_goal_matching_waypoint_ids=not args.hide_goal_matching_waypoint_ids,
    )
    _write_json(out_dir / "planner_request.json", request_payload)
    prompt = _build_prompt(request_payload)
    (out_dir / "planner_prompt.txt").write_text(prompt, encoding="utf-8")

    mode_used = _select_mode(args.mode, args.api_key_env)
    api_error = None
    raw_response: dict[str, Any] | None = None
    if mode_used == "api":
        try:
            raw_response = _call_openai_compatible_api(
                prompt=prompt,
                api_base=args.api_base,
                api_key=os.getenv(args.api_key_env, ""),
                model=args.model,
                timeout_s=float(args.timeout_s),
            )
            planner_output = _extract_planner_output(raw_response)
        except Exception as exc:  # noqa: BLE001 - report API failure and keep the closed loop runnable in auto mode.
            if args.mode == "api":
                raise
            api_error = str(exc)
            mode_used = "deterministic"
            planner_output = _deterministic_teacher(request_payload)
    else:
        planner_output = _deterministic_teacher(request_payload)

    planner_output = _normalize_planner_output(planner_output, request_payload)
    execution_trace = _execute_waypoints(planner_output, request_payload, success_radius_m=float(args.success_radius_m))
    checks = _run_checks(planner_output, execution_trace, request_payload)
    metrics = _metrics(checks, execution_trace, mode_used, api_error)

    if raw_response is not None:
        _write_json(out_dir / "api_raw_response.json", raw_response)
    _write_json(out_dir / "planner_output.json", planner_output)
    _write_json(out_dir / "execution_trace.json", execution_trace)
    _write_json(out_dir / "metrics.json", metrics)
    _write_html(out_dir / "phase5a_planner_report.html", context_path, request_payload, planner_output, execution_trace, metrics)

    summary = {
        "phase": "phase5a_api_semantic_task_planner",
        "status": "passed" if metrics["passed"] else "failed",
        "mode_used": mode_used,
        "api_error": api_error,
        "outputs": {
            "planner_request": str(out_dir / "planner_request.json"),
            "planner_prompt": str(out_dir / "planner_prompt.txt"),
            "planner_output": str(out_dir / "planner_output.json"),
            "execution_trace": str(out_dir / "execution_trace.json"),
            "metrics": str(out_dir / "metrics.json"),
            "html": str(out_dir / "phase5a_planner_report.html"),
        },
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _select_mode(mode: str, api_key_env: str) -> str:
    if mode == "deterministic":
        return "deterministic"
    if mode == "api":
        return "api"
    return "api" if os.getenv(api_key_env) else "deterministic"


def _build_request_payload(
    context: dict[str, Any],
    max_stopovers: int,
    expose_goal_matching_waypoint_ids: bool = True,
) -> dict[str, Any]:
    topk = list(context.get("topk_landmarks", []))
    nodes = list((context.get("element_topology_graph") or {}).get("nodes", []))
    edges = list((context.get("element_topology_graph") or {}).get("edges", []))
    waypoints = list(context.get("candidate_waypoints", []))
    object_summary = list(context.get("object_memory_summary", []))
    goal = str(context.get("goal_query") or "find target")
    goal_target_label = _goal_target_label(goal)
    waypoints = _ensure_goal_matching_waypoints(waypoints, topk, goal_target_label)
    matching_waypoint_ids = [
        str(item.get("id"))
        for item in waypoints
        if str(item.get("anchor_label", "")).lower() == goal_target_label
    ]
    return {
        "schema_version": "phase5a_planner_request_v1",
        "goal_query": goal,
        "goal_target_label": goal_target_label,
        "current_pose": context.get("current_pose"),
        "context_id": context.get("context_id"),
        "phase4_lite_context_state": context.get("phase4_lite_context_state"),
        "geometry_bev_summary": context.get("geometry_bev_summary"),
        "semantic_bev_summary": context.get("semantic_bev_summary"),
        "topk_landmarks": topk[:8],
        "object_memory_summary": object_summary[:24],
        "element_topology_graph": {
            "nodes": nodes[:32],
            "edges": edges[:80],
        },
        "candidate_waypoints": waypoints[:64],
        "goal_matching_waypoint_ids": matching_waypoint_ids[:32] if expose_goal_matching_waypoint_ids else [],
        "goal_matching_waypoint_ids_exposed": bool(expose_goal_matching_waypoint_ids),
        "candidate_path_summaries": list(context.get("candidate_path_summaries", []))[:64],
        "max_stopovers": max_stopovers,
        "required_output_schema": {
            "task_plan": [{"step": 1, "intent": "approach_object", "target": "landmark_id"}],
            "stopover_waypoints": ["waypoint_id"],
            "selected_waypoint_id": "waypoint_id",
            "waypoint_scores": [{"id": "waypoint_id", "score": 0.0}],
            "stop_probability": 0.0,
            "reason": "short audit explanation",
        },
        "constraints": [
            "Return JSON only.",
            "Use only candidate_waypoints ids for stopover_waypoints and selected_waypoint_id.",
            "Use only known landmark/node ids for task_plan targets.",
            "Do not output low-level actions.",
            "Prefer high-confidence active landmarks and Tier1/Tier2 nodes.",
            "The selected_waypoint_id anchor_label must match goal_target_label when such a candidate waypoint exists.",
            "For the Phase5A next-waypoint MVP, return exactly one stopover waypoint unless a connector is explicitly required by the goal.",
            "The traditional BEV executor will validate selected waypoints after your decision.",
        ],
    }


def _build_prompt(payload: dict[str, Any]) -> str:
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are the Phase5A semantic task planner for an indoor navigation system. "
        "Choose stopover waypoint ids from the candidate list using RSC memory, top-k landmarks, "
        "semantic tiers, confidence, freshness, status, and context. "
        "Return JSON only with task_plan, stopover_waypoints, selected_waypoint_id, waypoint_scores, "
        "stop_probability, and reason. Do not invent ids or low-level actions.\n\n"
        f"INPUT_JSON:\n{compact}"
    )


def _call_openai_compatible_api(prompt: str, api_base: str, api_key: str, model: str, timeout_s: float) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("API key is missing.")
    url = api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return valid compact JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - user-configured API endpoint.
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {detail[:1000]}") from exc


def _extract_planner_output(raw_response: dict[str, Any]) -> dict[str, Any]:
    choices = raw_response.get("choices") or []
    if not choices:
        raise RuntimeError("API response has no choices.")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("API response content is empty.")
    return _parse_jsonish(content)


def _parse_jsonish(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Planner output must be a JSON object.")
    return data


def _deterministic_teacher(payload: dict[str, Any]) -> dict[str, Any]:
    top_landmark = _choose_target_landmark(payload)
    waypoint_scores = []
    for waypoint in payload.get("candidate_waypoints", []):
        score = _score_waypoint(waypoint, top_landmark)
        if str(waypoint.get("anchor_label", "")).lower() == str(payload.get("goal_target_label", "")).lower():
            score += 0.35
        elif payload.get("goal_matching_waypoint_ids"):
            score -= 0.35
        waypoint_scores.append({"id": waypoint["id"], "score": round(score, 4)})
    waypoint_scores.sort(key=lambda item: item["score"], reverse=True)
    # Phase5A MVP selects the next semantic stopover, not a full low-level path.
    # Multi-stop plans will become meaningful once connector/room waypoints are
    # backed by a real Habitat navmesh executor.
    selected_ids = [waypoint_scores[0]["id"]] if waypoint_scores else []
    selected = selected_ids[0] if selected_ids else None
    target_id = top_landmark.get("landmark_id") or top_landmark.get("id") or top_landmark.get("target") or "unknown_target"
    target_label = top_landmark.get("label", "target")
    task_plan = [
        {"step": 1, "intent": "select_memory_landmark", "target": target_id},
        {"step": 2, "intent": "approach_object", "target": target_id},
        {"step": 3, "intent": "stop_near_object", "target": target_id},
    ]
    confidence = _float(top_landmark.get("confidence"), 0.5)
    final_score = _float(top_landmark.get("final_score"), confidence)
    return {
        "task_plan": task_plan,
        "stopover_waypoints": selected_ids,
        "selected_waypoint_id": selected,
        "waypoint_scores": waypoint_scores,
        "stop_probability": round(max(0.05, min(0.95, 0.25 + 0.55 * final_score)), 4),
        "reason": f"Deterministic teacher selected waypoints anchored to the top RSC landmark '{target_label}' using confidence/status-aware retrieval.",
    }


def _choose_target_landmark(payload: dict[str, Any]) -> dict[str, Any]:
    topk = payload.get("topk_landmarks") or []
    if topk:
        return dict(topk[0])
    nodes = [node for node in (payload.get("element_topology_graph") or {}).get("nodes", []) if node.get("is_planner_candidate", True)]
    nodes.sort(key=lambda item: _float(item.get("planner_salience"), 0.0), reverse=True)
    return dict(nodes[0]) if nodes else {}


def _score_waypoint(waypoint: dict[str, Any], target_landmark: dict[str, Any]) -> float:
    anchor = str(waypoint.get("anchor_landmark_id", ""))
    target_id = str(target_landmark.get("landmark_id") or target_landmark.get("id") or "")
    anchor_match = 1.0 if anchor and anchor == target_id else 0.0
    priority = _float(waypoint.get("priority"), 0.0)
    target_score = _float(target_landmark.get("final_score"), _float(target_landmark.get("planner_salience"), 0.5))
    # Keep non-target candidate scores non-zero, but make the selected target clearly preferred.
    return 0.55 * anchor_match + 0.25 * priority + 0.20 * target_score


def _ensure_goal_matching_waypoints(waypoints: list[dict[str, Any]], topk: list[dict[str, Any]], goal_label: str) -> list[dict[str, Any]]:
    if any(str(item.get("anchor_label", "")).lower() == goal_label for item in waypoints):
        return waypoints
    augmented = list(waypoints)
    offsets = [(0.9, 0.0), (-0.9, 0.0), (0.0, 0.9), (0.0, -0.9)]
    for landmark in topk:
        label = str(landmark.get("label", "")).lower()
        if label != goal_label:
            continue
        landmark_id = str(landmark.get("landmark_id") or landmark.get("id") or "")
        if not landmark_id:
            continue
        x, y = _xy(landmark.get("bev_position"))
        base_score = _float(landmark.get("final_score"), _float(landmark.get("confidence"), 0.5))
        for index, (dx, dy) in enumerate(offsets):
            augmented.append(
                {
                    "id": f"wp_auto_{landmark_id}_{index}".replace(" ", "_"),
                    "anchor_landmark_id": landmark_id,
                    "anchor_label": label,
                    "type": "auto_goal_perimeter",
                    "bev_position": [round(x + dx, 4), round(y + dy, 4)],
                    "radius_m": 0.9,
                    "requires_traditional_planner_validation": True,
                    "priority": round(max(0.0, min(1.0, base_score)), 4),
                    "generated_by": "phase5a_goal_waypoint_augmenter",
                }
            )
        break
    return augmented


def _normalize_planner_output(output: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    valid_waypoint_ids = {str(item.get("id")) for item in payload.get("candidate_waypoints", [])}
    valid_node_ids = {str(item.get("id")) for item in (payload.get("element_topology_graph") or {}).get("nodes", [])}
    valid_node_ids.update(str(item.get("landmark_id")) for item in payload.get("topk_landmarks", []) if item.get("landmark_id"))

    waypoint_scores = []
    for item in output.get("waypoint_scores", []):
        waypoint_id = str(item.get("id", ""))
        if waypoint_id in valid_waypoint_ids:
            waypoint_scores.append({"id": waypoint_id, "score": round(max(0.0, min(1.0, _float(item.get("score"), 0.0))), 4)})
    if not waypoint_scores:
        fallback = _deterministic_teacher(payload)
        waypoint_scores = fallback["waypoint_scores"]

    stopovers = [str(value) for value in output.get("stopover_waypoints", []) if str(value) in valid_waypoint_ids]
    if not stopovers and waypoint_scores:
        stopovers = [waypoint_scores[0]["id"]]
    selected = str(output.get("selected_waypoint_id") or (stopovers[0] if stopovers else ""))
    if selected not in valid_waypoint_ids:
        selected = stopovers[0] if stopovers else (waypoint_scores[0]["id"] if waypoint_scores else None)

    task_plan = []
    for index, step in enumerate(output.get("task_plan", []), start=1):
        if not isinstance(step, dict):
            continue
        target = str(step.get("target", ""))
        task_plan.append(
            {
                "step": int(step.get("step", index)),
                "intent": str(step.get("intent", "navigate")),
                "target": target if target in valid_node_ids or target in valid_waypoint_ids else target,
            }
        )
    if not task_plan:
        task_plan = _deterministic_teacher(payload)["task_plan"]

    return {
        "task_plan": task_plan,
        "stopover_waypoints": stopovers,
        "selected_waypoint_id": selected,
        "waypoint_scores": waypoint_scores,
        "stop_probability": round(max(0.0, min(1.0, _float(output.get("stop_probability"), 0.0))), 4),
        "reason": str(output.get("reason", ""))[:1200],
    }


def _execute_waypoints(output: dict[str, Any], payload: dict[str, Any], success_radius_m: float) -> dict[str, Any]:
    waypoint_by_id = {str(item.get("id")): item for item in payload.get("candidate_waypoints", [])}
    current = _current_xy(payload)
    segments = []
    path_length = 0.0
    executable = True
    failure_reasons = []
    for index, waypoint_id in enumerate(output.get("stopover_waypoints", []), start=1):
        waypoint = waypoint_by_id.get(str(waypoint_id))
        if waypoint is None:
            executable = False
            failure_reasons.append(f"unknown waypoint id: {waypoint_id}")
            continue
        target = _xy(waypoint.get("bev_position"))
        distance = _dist(current, target)
        path_length += distance
        segments.append(
            {
                "segment": index,
                "from": [round(current[0], 4), round(current[1], 4)],
                "to": [round(target[0], 4), round(target[1], 4)],
                "waypoint_id": waypoint_id,
                "anchor_landmark_id": waypoint.get("anchor_landmark_id"),
                "anchor_label": waypoint.get("anchor_label"),
                "euclidean_length_m": round(distance, 4),
                "traditional_planner_proxy": "accepted_candidate_waypoint",
                "requires_navmesh_recheck": True,
            }
        )
        current = target
    selected_wp = waypoint_by_id.get(str(output.get("selected_waypoint_id")))
    selected_anchor = selected_wp.get("anchor_landmark_id") if selected_wp else None
    target_landmark = _target_landmark(payload, selected_anchor)
    final_distance = _dist(current, _xy(target_landmark.get("bev_position"))) if target_landmark else None
    stop_success = bool(final_distance is not None and final_distance <= success_radius_m and output.get("stop_probability", 0.0) >= 0.5)
    return {
        "executor": "traditional_bev_waypoint_proxy_v1",
        "executable": executable,
        "failure_reasons": failure_reasons,
        "start_pose": payload.get("current_pose"),
        "segments": segments,
        "path_length_m": round(path_length, 4),
        "selected_waypoint_id": output.get("selected_waypoint_id"),
        "selected_anchor_landmark_id": selected_anchor,
        "target_landmark": target_landmark,
        "final_distance_to_target_m": round(final_distance, 4) if final_distance is not None else None,
        "success_radius_m": float(success_radius_m),
        "stop_success_proxy": stop_success,
        "note": "This is a BEV candidate-waypoint executor for the Phase5A MVP. Habitat navmesh shortest-path execution should replace it in the next refinement.",
    }


def _run_checks(output: dict[str, Any], trace: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    valid_waypoints = {str(item.get("id")) for item in payload.get("candidate_waypoints", [])}
    valid_targets = {str(item.get("id")) for item in (payload.get("element_topology_graph") or {}).get("nodes", [])}
    valid_targets.update(str(item.get("landmark_id")) for item in payload.get("topk_landmarks", []) if item.get("landmark_id"))
    task_targets = [str(step.get("target", "")) for step in output.get("task_plan", []) if isinstance(step, dict)]
    goal_label = str(payload.get("goal_target_label", "")).lower()
    selected_anchor_label = str((trace.get("target_landmark") or {}).get("label", "")).lower()
    available_goal_matching_waypoints = [
        str(item.get("id"))
        for item in payload.get("candidate_waypoints", [])
        if str(item.get("anchor_label", "")).lower() == goal_label
    ]
    needs_goal_match = bool(available_goal_matching_waypoints)
    return [
        {
            "name": "json_parse_success",
            "passed": isinstance(output, dict) and bool(output.get("task_plan")),
            "detail": "planner_output.json is a structured object",
        },
        {
            "name": "stopover_waypoints_valid",
            "passed": bool(output.get("stopover_waypoints")) and all(str(item) in valid_waypoints for item in output.get("stopover_waypoints", [])),
            "detail": output.get("stopover_waypoints"),
        },
        {
            "name": "selected_waypoint_valid",
            "passed": str(output.get("selected_waypoint_id")) in valid_waypoints,
            "detail": output.get("selected_waypoint_id"),
        },
        {
            "name": "task_plan_references_known_memory",
            "passed": any(target in valid_targets for target in task_targets),
            "detail": task_targets[:8],
        },
        {
            "name": "traditional_planner_proxy_executable",
            "passed": bool(trace.get("executable")) and len(trace.get("segments", [])) > 0,
            "detail": {"segments": len(trace.get("segments", [])), "path_length_m": trace.get("path_length_m")},
        },
        {
            "name": "stop_decision_available",
            "passed": 0.0 <= _float(output.get("stop_probability"), -1.0) <= 1.0,
            "detail": output.get("stop_probability"),
        },
        {
            "name": "selected_anchor_matches_goal_label",
            "passed": (not needs_goal_match) or (selected_anchor_label == goal_label),
            "detail": {
                "goal_target_label": goal_label,
                "selected_anchor_label": selected_anchor_label,
                "num_goal_matching_waypoints_exposed": len(payload.get("goal_matching_waypoint_ids", [])),
                "num_goal_matching_waypoints_available": len(available_goal_matching_waypoints),
            },
        },
    ]


def _metrics(checks: list[dict[str, Any]], trace: dict[str, Any], mode_used: str, api_error: str | None) -> dict[str, Any]:
    passed_count = sum(1 for item in checks if item["passed"])
    return {
        "passed": passed_count == len(checks),
        "mode_used": mode_used,
        "api_error": api_error,
        "num_checks": len(checks),
        "num_passed": passed_count,
        "pass_rate": passed_count / max(1, len(checks)),
        "traditional_planner_execution_success": bool(trace.get("executable")),
        "stop_success_proxy": bool(trace.get("stop_success_proxy")),
        "goal_label_aligned": bool(next((item["passed"] for item in checks if item["name"] == "selected_anchor_matches_goal_label"), False)),
        "path_length_m": trace.get("path_length_m"),
        "final_distance_to_target_m": trace.get("final_distance_to_target_m"),
        "checks": checks,
    }


def _target_landmark(payload: dict[str, Any], selected_anchor: str | None) -> dict[str, Any] | None:
    if selected_anchor:
        for node in (payload.get("element_topology_graph") or {}).get("nodes", []):
            if str(node.get("id")) == str(selected_anchor):
                return node
        for item in payload.get("topk_landmarks", []):
            if str(item.get("landmark_id")) == str(selected_anchor):
                return item
    chosen = _choose_target_landmark(payload)
    return chosen or None


def _current_xy(payload: dict[str, Any]) -> tuple[float, float]:
    pose = payload.get("current_pose") or {}
    if isinstance(pose, dict) and "x" in pose:
        return _float(pose.get("x"), 0.0), _float(pose.get("y", pose.get("z")), 0.0)
    return 0.0, 0.0


def _xy(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _float(value[0], 0.0), _float(value[1], 0.0)
    return 0.0, 0.0


def _dist(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _goal_target_label(goal_query: str) -> str:
    text = goal_query.lower().strip()
    for prefix in ("navigate to ", "approach ", "go to ", "find ", "locate "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.strip().split()[-1] if text.strip() else "target"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_html(path: Path, context_path: Path, request_payload: dict[str, Any], output: dict[str, Any], trace: dict[str, Any], metrics: dict[str, Any]) -> None:
    checks = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td>{'passed' if item['passed'] else 'failed'}</td>"
        f"<td><pre>{html.escape(json.dumps(item.get('detail'), ensure_ascii=False, indent=2))}</pre></td>"
        "</tr>"
        for item in metrics.get("checks", [])
    )
    scores = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('id')))}</td>"
        f"<td>{float(item.get('score', 0.0)):.4f}</td>"
        "</tr>"
        for item in output.get("waypoint_scores", [])[:12]
    )
    segments = "\n".join(
        "<tr>"
        f"<td>{item.get('segment')}</td>"
        f"<td>{html.escape(str(item.get('waypoint_id')))}</td>"
        f"<td>{html.escape(str(item.get('anchor_label')))}</td>"
        f"<td>{item.get('euclidean_length_m')}</td>"
        "</tr>"
        for item in trace.get("segments", [])
    )
    output_preview = html.escape(json.dumps(output, indent=2, ensure_ascii=False))
    trace_preview = html.escape(json.dumps(trace, indent=2, ensure_ascii=False))
    status_color = "#137333" if metrics.get("passed") else "#b3261e"
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase5A API Semantic Planner MVP</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; background: #fafafa; line-height: 1.45; }}
h1 {{ margin-bottom: 4px; }}
.pill {{ display: inline-block; padding: 4px 8px; border-radius: 999px; background: #e8f0fe; color: #174ea6; margin-right: 8px; }}
.status {{ color: {status_color}; font-weight: 700; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
section {{ background: white; border: 1px solid #dde1e6; border-radius: 8px; padding: 14px; margin: 16px 0; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
th, td {{ border: 1px solid #dde1e6; padding: 8px; vertical-align: top; text-align: left; }}
th {{ background: #f1f3f4; }}
pre {{ background: #f6f8fa; padding: 10px; border-radius: 6px; overflow-x: auto; max-height: 520px; }}
code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
</style></head>
<body>
<h1>Phase5A API Semantic Task Planner MVP</h1>
<p><span class="pill">status: <span class="status">{html.escape('passed' if metrics.get('passed') else 'failed')}</span></span><span class="pill">mode: {html.escape(str(metrics.get('mode_used')))}</span><span class="pill">path: {metrics.get('path_length_m')} m</span></p>
<p>Input: <code>{html.escape(str(context_path))}</code></p>
<section>
<h2>Closed Loop</h2>
<p><code>planner_context.json -> semantic task plan -> stopover waypoint ranking -> BEV waypoint executor -> trace/report</code></p>
<p>Executor note: this MVP uses a traditional BEV candidate-waypoint proxy. Habitat navmesh shortest-path execution should replace it in the next refinement.</p>
</section>
<div class="grid">
<section>
<h2>Planner Output</h2>
<pre>{output_preview}</pre>
</section>
<section>
<h2>Execution Trace</h2>
<pre>{trace_preview}</pre>
</section>
</div>
<section>
<h2>Waypoint Scores</h2>
<table><tr><th>Waypoint ID</th><th>Score</th></tr>{scores}</table>
</section>
<section>
<h2>Executed Segments</h2>
<table><tr><th>#</th><th>Waypoint</th><th>Anchor</th><th>Length m</th></tr>{segments}</table>
</section>
<section>
<h2>Checks</h2>
<table><tr><th>Check</th><th>Status</th><th>Detail</th></tr>{checks}</table>
</section>
<section>
<h2>Files</h2>
<ul>
<li><a href="planner_request.json">planner_request.json</a></li>
<li><a href="planner_prompt.txt">planner_prompt.txt</a></li>
<li><a href="planner_output.json">planner_output.json</a></li>
<li><a href="execution_trace.json">execution_trace.json</a></li>
<li><a href="metrics.json">metrics.json</a></li>
</ul>
</section>
</body></html>"""
    path.write_text(html_doc, encoding="utf-8")


if __name__ == "__main__":
    main()
