from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_agent_caption(
    *,
    task: str | None,
    interest: Mapping[str, Any] | None,
    task_plan_events: Sequence[Mapping[str, Any]] | None = None,
    confirmed_cup_track_ids: Sequence[int] | None = None,
    target_label: str = "cup",
) -> dict[str, Any]:
    """Build a concise, trace-grounded explanation of the current decision."""
    state = dict(interest or {})
    events = [dict(event) for event in (task_plan_events or [])]
    confirmed = [int(track_id) for track_id in (confirmed_cup_track_ids or [])]
    target = str(target_label or "target").strip().lower()
    targets = _plural(target)
    sources = ["interest.mode"]

    if not task:
        return {
            "stage": "FAMILIARIZE",
            "plan": "Build reusable spatial memory",
            "why": (
                "The task is still hidden. Update geometry, object tracks, and "
                "room anchors from the current RGB-D observation."
            ),
            "next": "Move to the next high-information reachable view.",
            "evidence": "No task-specific evidence is available to the agent.",
            "source_fields": ["task", "interest.mode"],
        }

    ranking = [
        dict(item)
        for item in state.get("task_search_ranking", [])
        if isinstance(item, Mapping)
    ]
    by_id = {
        str(item.get("candidate_id")): item
        for item in ranking
        if item.get("candidate_id") is not None
    }
    active_value = state.get("task_active_candidate_id")
    active_id = (
        _candidate_id(active_value)
        if active_value is not None and str(active_value).strip()
        else ""
    )
    active = by_id.get(active_id, {})
    event = _most_informative_event(events)

    if event is not None:
        name = str(event.get("event", ""))
        candidate_id = _candidate_id(event.get("candidate_id"))
        candidate = by_id.get(candidate_id, {})
        sources.append(f"task_plan_events.{name}")

        if name == "task_injected_and_planned":
            planner = str(state.get("task_planner_model") or "API planner")
            return _caption(
                stage="PLAN",
                plan=f"Task received: find and report all {targets}",
                why=(
                    "Use grounded object memory and scene context to rank direct "
                    f"{target} hypotheses before likely support surfaces."
                ),
                next=(
                    _ranking_next(ranking)
                    if ranking
                    else "Initialize search beliefs from grounded planner candidates."
                ),
                evidence=(
                    _ranking_evidence(ranking)
                    if ranking
                    else f"Planner response: {planner}; grounded ranking is initializing."
                ),
                sources=sources + ["task", "interest.task_search_ranking"],
            )

        if name in {
            "cup_candidate_reobservation_started",
            "target_candidate_reobservation_started",
        }:
            return _caption(
                stage="VERIFY",
                plan=f"Verify {target} hypothesis {candidate_id}",
                why=(
                    "A detector candidate is not a result. Independent RGB-D "
                    "views must confirm visual identity and 3D relief."
                ),
                next="Hold the target in view and collect translated observations.",
                evidence=_candidate_evidence(candidate_id, candidate),
                sources=sources + ["interest.task_search_ranking"],
            )

        if name in {
            "cup_confirmation_observation",
            "target_confirmation_observation",
        }:
            status = str(event.get("status", "collecting"))
            return _caption(
                stage="VERIFY",
                plan=f"Gather evidence for {candidate_id}",
                why=(
                    f"Current gate status is {status}; "
                    f"{int(event.get('task_independent_views', 0))} independent "
                    f"views and {int(event.get('visual_passes', 0))} visual passes "
                    "are recorded."
                ),
                next="Continue until the strict confirmation gate resolves.",
                evidence=_candidate_evidence(candidate_id, candidate),
                sources=sources + ["interest.task_search_ranking"],
            )

        if name in {
            "cup_confirmation_completed",
            "cup_confirmation_deferred",
            "cup_confirmation_retry_scheduled",
            "target_confirmation_completed",
            "target_confirmation_deferred",
            "target_confirmation_retry_scheduled",
        }:
            return _target_result_caption(
                event,
                candidate,
                sources,
                target_label=target,
            )

        if name == "support_surface_inspection_started":
            hypothesis = event.get("search_hypothesis")
            if isinstance(hypothesis, Mapping):
                candidate = dict(hypothesis)
            return _caption(
                stage="SEARCH",
                plan=f"Inspect likely support {candidate_id}",
                why=(
                    f"Scene context predicts {targets} may occur here, but this is a "
                    "search prior rather than object confirmation."
                ),
                next=(
                    f"Sweep the surface and look for newly grounded {target} evidence."
                ),
                evidence=_candidate_evidence(candidate_id, candidate),
                sources=sources + ["task_plan_events.search_hypothesis"],
            )

        if name == "support_surface_inspection_completed":
            return _support_result_caption(event, candidate, sources)

        if name in {
            "candidate_unreachable",
            "candidate_navigation_failed",
            "candidate_navigation_failure",
        }:
            return _caption(
                stage="REPLAN",
                plan=f"Skip unreachable hypothesis {candidate_id}",
                why=(
                    "The current execution route could not reach its inspection "
                    "waypoint, so this candidate cannot be evaluated now."
                ),
                next="Select the next reachable high-posterior hypothesis.",
                evidence=_candidate_evidence(candidate_id, candidate),
                sources=sources + ["interest.task_search_ranking"],
            )

        if name == "online_memory_candidates_appended":
            appended = [
                _candidate_id(item)
                for item in event.get("candidate_ids", [])
            ]
            return _caption(
                stage="REPLAN",
                plan="Add newly observed hypotheses",
                why=(
                    "Online grounding produced candidates that were absent when "
                    "the task was first planned."
                ),
                next=(
                    f"Re-rank direct {targets} and support surfaces with current evidence."
                ),
                evidence=(
                    "New candidates: " + ", ".join(appended[:4])
                    if appended
                    else _ranking_evidence(ranking)
                ),
                sources=sources + ["task_plan_events.candidate_ids"],
            )

    mode = str(state.get("mode", ""))
    kind = str(active.get("kind", state.get("target_kind", "")))
    label = str(active.get("label", kind or "candidate"))
    sources.extend(
        [
            "interest.task_active_candidate_id",
            "interest.task_search_ranking",
        ]
    )

    if active_id and (
        kind in {"cup", "target", "target_object"}
        or label.lower() == target
    ):
        if "scan" in mode or "confirm" in mode:
            return _caption(
                stage="VERIFY",
                plan=f"Verify {target} hypothesis {active_id}",
                why=(
                    "The candidate is visually plausible, but only independent "
                    "RGB-D evidence can promote it to a reported result."
                ),
                next="Collect enough stable views to resolve the strict gate.",
                evidence=_candidate_evidence(active_id, active),
                sources=sources,
            )
        return _caption(
            stage="APPROACH",
            plan=f"Approach {target} hypothesis {active_id}",
            why="Direct target hypotheses are inspected before contextual supports.",
            next="Reach a clear viewpoint, then start strict confirmation.",
            evidence=_candidate_evidence(active_id, active),
            sources=sources,
        )

    if active_id:
        if "scan" in mode:
            return _caption(
                stage="SEARCH",
                plan=f"Inspect {label} {active_id}",
                why=(
                    "Its current posterior makes this a useful place to search "
                    f"for {target} evidence."
                ),
                next="Complete the observable sweep, then update its belief.",
                evidence=_candidate_evidence(active_id, active),
                sources=sources,
            )
        return _caption(
            stage="APPROACH",
            plan=f"Approach {label} {active_id}",
            why=(
                "This is the highest-ranked reachable hypothesis under the "
                "current semantic search belief."
            ),
            next="Navigate to an inspection waypoint and gather evidence.",
            evidence=_candidate_evidence(active_id, active),
            sources=sources,
        )

    if confirmed:
        ids = ", ".join(f"track_{track_id}" for track_id in confirmed)
        return _caption(
            stage="REPORT",
            plan=f"Preserve verified {target} results",
            why="Only candidates that passed the strict gate may be reported.",
            next="Continue searching untested high-posterior locations.",
            evidence=f"Verified: {ids}",
            sources=sources + ["confirmed_target_track_ids"],
        )

    return _caption(
        stage="REPLAN",
        plan="Choose the next search hypothesis",
        why="No active candidate is currently assigned to the executor.",
        next=_ranking_next(ranking),
        evidence=_ranking_evidence(ranking),
        sources=sources,
    )


def _caption(
    *,
    stage: str,
    plan: str,
    why: str,
    next: str,
    evidence: str,
    sources: Sequence[str],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "plan": plan,
        "why": why,
        "next": next,
        "evidence": evidence,
        "source_fields": list(dict.fromkeys(str(item) for item in sources)),
    }


def _candidate_id(value: Any) -> str:
    text = str(value or "unknown")
    if text.isdigit():
        return f"track_{text}"
    return text


def _plural(label: str) -> str:
    normalized = str(label).strip()
    if normalized.endswith("s"):
        return normalized
    if normalized.endswith(("ch", "sh", "x", "z")):
        return normalized + "es"
    return normalized + "s"


def _posterior(candidate: Mapping[str, Any]) -> float | None:
    for key in ("posterior", "target_posterior", "target_likelihood", "prior"):
        value = candidate.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _candidate_evidence(
    candidate_id: str,
    candidate: Mapping[str, Any],
) -> str:
    label = str(candidate.get("label", candidate.get("kind", "hypothesis")))
    posterior = _posterior(candidate)
    room = str(candidate.get("room_type", "")).strip()
    parts = [f"{candidate_id}: {label}"]
    if posterior is not None:
        parts.append(f"posterior {posterior:.2f}")
    if room and room.lower() not in {"none", "unknown"}:
        parts.append(f"room {room}")
    return " | ".join(parts)


def _ranking_evidence(ranking: Sequence[Mapping[str, Any]]) -> str:
    if not ranking:
        return "No ranked hypothesis is available."
    items = []
    for candidate in ranking[:3]:
        candidate_id = _candidate_id(candidate.get("candidate_id"))
        label = str(candidate.get("label", candidate.get("kind", "candidate")))
        posterior = _posterior(candidate)
        score = f" p={posterior:.2f}" if posterior is not None else ""
        items.append(f"{candidate_id} {label}{score}")
    return "Top hypotheses: " + "; ".join(items)


def _ranking_next(ranking: Sequence[Mapping[str, Any]]) -> str:
    if not ranking:
        return "Wait for grounded candidates, then replan."
    candidate = ranking[0]
    return (
        f"Inspect {_candidate_id(candidate.get('candidate_id'))} "
        f"({candidate.get('label', candidate.get('kind', 'candidate'))}) first."
    )


def _most_informative_event(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    priority = {
        "cup_confirmation_completed": 100,
        "cup_confirmation_deferred": 99,
        "cup_confirmation_retry_scheduled": 98,
        "target_confirmation_completed": 100,
        "target_confirmation_deferred": 99,
        "target_confirmation_retry_scheduled": 98,
        "support_surface_inspection_completed": 95,
        "candidate_unreachable": 90,
        "candidate_navigation_failed": 90,
        "candidate_navigation_failure": 90,
        "support_surface_inspection_started": 80,
        "cup_candidate_reobservation_started": 80,
        "cup_confirmation_observation": 70,
        "target_candidate_reobservation_started": 80,
        "target_confirmation_observation": 70,
        "online_memory_candidates_appended": 60,
        "task_injected_and_planned": 50,
    }
    relevant = [
        dict(event)
        for event in events
        if str(event.get("event", "")) in priority
    ]
    if not relevant:
        return None
    return max(
        relevant,
        key=lambda item: priority[str(item.get("event", ""))],
    )


def _target_result_caption(
    event: Mapping[str, Any],
    candidate: Mapping[str, Any],
    sources: Sequence[str],
    *,
    target_label: str,
) -> dict[str, Any]:
    candidate_id = _candidate_id(event.get("candidate_id"))
    status = str(event.get("status", "inconclusive"))
    views = int(event.get("task_independent_views", 0))
    passes = int(event.get("visual_passes", 0))
    negatives = int(event.get("visual_negatives", 0))
    evidence = (
        f"{candidate_id}: {status} | views {views} | "
        f"visual +{passes}/-{negatives}"
    )
    if status == "verified":
        return _caption(
            stage="CONFIRMED",
            plan=f"Verify {target_label} {candidate_id}",
            why="The candidate passed the independent-view, visual, and 3D gate.",
            next=(
                "Store the verified location and continue searching for other "
                f"{_plural(target_label)}."
            ),
            evidence=evidence,
            sources=sources + ["task_plan_events.belief_update"],
        )
    if status.startswith("rejected_"):
        reason = status.removeprefix("rejected_").replace("_", " ")
        return _caption(
            stage="REPLAN",
            plan=f"Reject {target_label} hypothesis {candidate_id}",
            why=f"Strict confirmation resolved it as {reason}.",
            next="Remove it from the result set and inspect the next hypothesis.",
            evidence=evidence,
            sources=sources + ["task_plan_events.belief_update"],
        )
    return _caption(
        stage="REPLAN",
        plan=f"Keep {candidate_id} inconclusive",
        why="The bounded verification attempt did not produce enough evidence.",
        next="Deprioritize it for now and continue with the next hypothesis.",
        evidence=evidence,
        sources=sources + ["task_plan_events.belief_update"],
    )


def _support_result_caption(
    event: Mapping[str, Any],
    candidate: Mapping[str, Any],
    sources: Sequence[str],
) -> dict[str, Any]:
    candidate_id = _candidate_id(event.get("candidate_id"))
    outcome = str(event.get("outcome", "unknown"))
    observable = bool(event.get("observable_scan"))
    update = event.get("belief_update")
    belief = dict(update) if isinstance(update, Mapping) else {}
    before = belief.get("prior_posterior")
    after = belief.get("posterior")
    transition = ""
    if before is not None and after is not None:
        transition = f" | posterior {float(before):.2f} -> {float(after):.2f}"
    if not observable:
        return _caption(
            stage="REPLAN",
            plan=f"Defer evidence update for {candidate_id}",
            why="The support sweep was not observable enough for negative evidence.",
            next="Seek another viewpoint or inspect the next reachable hypothesis.",
            evidence=_candidate_evidence(candidate_id, candidate) + transition,
            sources=sources + ["task_plan_events.belief_update"],
        )
    if outcome == "target_evidence_observed":
        observed = [
            _candidate_id(item)
            for item in event.get("observed_target_candidate_ids", [])
        ]
        return _caption(
            stage="REPLAN",
            plan=f"Cup evidence found near {candidate_id}",
            why=(
                "The observable support sweep produced grounded cup hypotheses; "
                "they are candidates, not yet verified results."
            ),
            next="Re-rank the new cup hypotheses and verify them independently.",
            evidence=(
                f"Observed: {', '.join(observed[:4]) or 'new cup candidates'}"
                f"{transition}"
            ),
            sources=sources + ["task_plan_events.belief_update"],
        )
    return _caption(
        stage="REPLAN",
        plan=f"No cup evidence at {candidate_id}",
        why="An observable sweep found no target evidence at this support.",
        next="Lower its search belief and move to the next reachable hypothesis.",
        evidence=_candidate_evidence(candidate_id, candidate) + transition,
        sources=sources + ["task_plan_events.belief_update"],
    )
