from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any


TIER0_BACKGROUND = {"wall", "floor", "ceiling", "ceiling light"}
TIER1_CONNECTOR = {"door", "corridor", "hallway", "entrance", "exit", "stairs", "staircase", "frontier"}
TIER2_LANDMARK = {
    "bed",
    "sofa",
    "chair",
    "table",
    "cabinet",
    "tv",
    "television",
    "sink",
    "toilet",
    "counter",
    "shelf",
    "bookshelf",
}
TIER3_TASK_OBJECT = {"cup", "mug", "book", "remote", "laptop", "bottle", "plant"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the M3.5 G/S/O/L representation bundle for Phase5A planner input.")
    parser.add_argument("--run-dir", required=True, help="M2/M2.5 validation directory containing bridge/ and grounding metrics.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default="m35_representation_bundle")
    parser.add_argument("--goal-query", default="find bed")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--near-radius-m", type=float, default=1.8)
    parser.add_argument("--waypoint-radius-m", type=float, default=0.9)
    parser.add_argument("--max-nodes", type=int, default=80)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    bridge_dir = run_dir / "bridge"
    retrieval_dir = bridge_dir / "phase3_retrieval"
    out_dir = Path(args.out_dir).expanduser().resolve()
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    bridge_metadata = _read_json(bridge_dir / "bridge_metadata.json")
    rsc_memory = _read_json(bridge_dir / "rsc_memory_init.json")
    landmark_nodes = _read_json(retrieval_dir / "landmark_nodes.json")
    topk_retrieval = _read_json(retrieval_dir / "topk_retrieval.json")
    retrieval_metrics = _read_json(retrieval_dir / "metrics.json")
    grounding_metrics = _read_json(run_dir / "grounding_metrics.json", required=False) or {}
    candidates_payload = _read_json(run_dir / "grounding_candidates.json", required=False) or {}
    mvp_metadata = _read_json(run_dir / "rgb_to_semantic_bev_mvp" / "rgb_to_semantic_bev_mvp_metadata.json", required=False) or {}

    copied_assets = _copy_assets(run_dir, bridge_dir, retrieval_dir, assets_dir)
    objects = list(rsc_memory.get("items", []))
    object_by_id = {str(item.get("id") or item.get("object_id")): item for item in objects}
    enriched_nodes = _enrich_landmarks(
        landmark_nodes,
        object_by_id=object_by_id,
        max_nodes=max(1, int(args.max_nodes)),
    )
    edges = _build_edges(enriched_nodes, near_radius_m=float(args.near_radius_m))
    candidate_waypoints = _candidate_waypoints(enriched_nodes, radius_m=float(args.waypoint_radius_m))
    selected_goal = str(args.goal_query)
    topk_for_goal = _select_topk(topk_retrieval, selected_goal, top_k=int(args.top_k))

    geometry_summary = {
        "representation": "G: Geometry BEV",
        "purpose": "traditional navigation, reachability, and collision constraints",
        "grid": bridge_metadata.get("grid", {}),
        "occupancy_cells": bridge_metadata.get("occupancy_cells", 0),
        "point_count": bridge_metadata.get("point_count", 0),
        "assets": {
            "occupancy_bev_png": copied_assets.get("occupancy_bev.png"),
            "traditional_bev_phase2_style_png": copied_assets.get("traditional_bev_phase2_style.png"),
        },
        "planner_role": "hard geometry constraint; Phase5A should validate candidate waypoints with this layer.",
    }
    semantic_summary = {
        "representation": "S: Semantic Evidence BEV",
        "purpose": "class evidence and confidence projected into BEV; PNG is visualization only",
        "label_to_id": bridge_metadata.get("label_to_id", {}),
        "semantic_cells": bridge_metadata.get("semantic_cells", 0),
        "num_labels": bridge_metadata.get("num_labels", 0),
        "validation": (grounding_metrics.get("validation") or {}),
        "assets": {
            "semantic_bev_png": copied_assets.get("semantic_bev.png"),
            "semantic_bev_mvp_png": copied_assets.get("semantic_bev_from_owlv2_mvp.png"),
            "semantic_confidence_png": copied_assets.get("semantic_evidence_confidence.png"),
        },
        "planner_role": "compact semantic cue; planner should prefer O/L over dense pixels for task planning.",
    }
    object_memory = {
        "representation": "O: Object Memory",
        "scene_id": rsc_memory.get("scene_id", bridge_metadata.get("scene_id")),
        "context_id": bridge_metadata.get("context_id"),
        "source": rsc_memory.get("source"),
        "items": [_enrich_object(item) for item in objects],
        "planner_policy": {
            "tier0_background": "do not use as goal landmarks; keep as geometry/background cue",
            "tier1_connector": "high priority for stopover and room transition planning",
            "tier2_landmark": "high priority for goal anchoring and semantic memory reuse",
            "tier3_task_object": "query-dependent priority; use confidence and freshness conservatively",
        },
    }
    landmark_graph = {
        "representation": "L: Landmark / Element Topology Graph",
        "context_id": bridge_metadata.get("context_id"),
        "nodes": enriched_nodes,
        "edges": edges,
        "retrieval": {
            "topk_retrieval": topk_retrieval,
            "selected_goal_query": selected_goal,
            "selected_topk": topk_for_goal,
            "metrics": retrieval_metrics.get("metrics", {}),
        },
        "candidate_waypoints": candidate_waypoints,
        "graph_notes": [
            "near edges are metric proximity hints, not a full route graph.",
            "reachable_anchor waypoints must be checked by the traditional BEV/navmesh planner before execution.",
            "context_id is preserved for Phase4 remapping and cross-context isolation.",
        ],
    }
    planner_context = {
        "schema_version": "m35_planner_context_v1",
        "goal_query": selected_goal,
        "scene_id": bridge_metadata.get("scene_id"),
        "current_pose": None,
        "context_id": bridge_metadata.get("context_id"),
        "phase4_lite_context_state": {
            "current_context_id": bridge_metadata.get("context_id"),
            "context_confidence": 1.0,
            "remap_triggered": False,
            "mismatch_signals": [],
            "note": "Phase4 full remapping is not implemented yet; fields are reserved so Phase5A input will not need a schema rewrite.",
        },
        "geometry_bev_summary": _compact_geometry(geometry_summary),
        "semantic_bev_summary": _compact_semantic(semantic_summary),
        "topk_landmarks": topk_for_goal,
        "object_memory_summary": _planner_objects(object_memory["items"]),
        "element_topology_graph": {
            "nodes": _planner_nodes(enriched_nodes),
            "edges": _planner_edges(edges),
        },
        "candidate_waypoints": candidate_waypoints,
        "candidate_path_summaries": [
            {
                "waypoint_id": item["id"],
                "anchor_landmark_id": item["anchor_landmark_id"],
                "requires_traditional_planner_validation": True,
            }
            for item in candidate_waypoints
        ],
        "output_contract": {
            "task_plan": "ordered high-level intents referencing node ids or waypoint ids",
            "stopover_waypoints": "waypoint ids selected from candidate_waypoints",
            "selected_waypoint_id": "single next waypoint id",
            "waypoint_scores": "score per candidate waypoint",
            "stop_probability": "0..1 probability that the agent should stop near the current/selected landmark",
        },
        "constraints": [
            "Do not output low-level actions.",
            "Do not invent unreachable free-form coordinates.",
            "Prefer Tier1 connectors and Tier2 landmarks over Tier0 background classes for task planning.",
            "Validate all selected waypoint ids with traditional BEV/navmesh before execution.",
        ],
    }
    manifest = {
        "phase": "m35_semantic_representation_alignment",
        "label": args.label,
        "status": "passed" if _bundle_passed(geometry_summary, semantic_summary, object_memory, landmark_graph, planner_context) else "failed",
        "run_dir": str(run_dir),
        "context_id": bridge_metadata.get("context_id"),
        "scene_id": bridge_metadata.get("scene_id"),
        "counts": {
            "objects": len(object_memory["items"]),
            "landmark_nodes": len(enriched_nodes),
            "edges": len(edges),
            "candidate_waypoints": len(candidate_waypoints),
            "topk_for_goal": len(topk_for_goal),
        },
        "quality": {
            "grounding_validation": semantic_summary["validation"],
            "retrieval_metrics": retrieval_metrics.get("metrics", {}),
        },
        "outputs": {
            "geometry_bev_summary": "geometry_bev_summary.json",
            "semantic_evidence_summary": "semantic_evidence_summary.json",
            "object_memory": "object_memory.json",
            "landmark_graph": "landmark_graph.json",
            "planner_context": "planner_context.json",
            "html": "representation_bundle_report.html",
        },
        "assets": copied_assets,
        "notes": [
            "This bundle fixes the G/S/O/L planner input contract before Phase5A.",
            "Phase4 remapping is represented as reserved context fields only; full mismatch scoring is a later phase.",
            "The bundle uses existing best96 GroundingDINO evidence and does not rerun perception.",
        ],
    }

    _write_json(out_dir / "geometry_bev_summary.json", geometry_summary)
    _write_json(out_dir / "semantic_evidence_summary.json", semantic_summary)
    _write_json(out_dir / "object_memory.json", object_memory)
    _write_json(out_dir / "landmark_graph.json", landmark_graph)
    _write_json(out_dir / "planner_context.json", planner_context)
    _write_json(out_dir / "bundle_manifest.json", manifest)
    _write_html(out_dir / "representation_bundle_report.html", manifest, geometry_summary, semantic_summary, object_memory, landmark_graph, planner_context, mvp_metadata)
    print(json.dumps(manifest, indent=2))


def _read_json(path: Path, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_assets(run_dir: Path, bridge_dir: Path, retrieval_dir: Path, assets_dir: Path) -> dict[str, str]:
    candidates = [
        bridge_dir / "occupancy_bev.png",
        bridge_dir / "semantic_bev.png",
        retrieval_dir / "retrieval_bev.png",
        retrieval_dir / "retrieval_bev.svg",
        run_dir / "rgb_to_semantic_bev_mvp" / "traditional_bev_phase2_style.png",
        run_dir / "rgb_to_semantic_bev_mvp" / "object_inventory_projection_evidence.png",
        run_dir / "rgb_to_semantic_bev_mvp" / "semantic_bev_from_owlv2_mvp.png",
        run_dir / "rgb_to_semantic_bev_mvp" / "semantic_evidence_confidence.png",
        run_dir / "rgb_to_semantic_bev_mvp" / "rgb_to_semantic_bev_mvp_pipeline.png",
    ]
    copied = {}
    for source in candidates:
        if not source.exists():
            continue
        target = assets_dir / source.name
        shutil.copy2(source, target)
        copied[source.name] = f"assets/{source.name}"
    return copied


def _enrich_object(item: dict[str, Any]) -> dict[str, Any]:
    label = _label(item)
    tier, role, base_salience = _tier(label)
    confidence = _float(item.get("confidence"), 0.5)
    freshness = _float(item.get("freshness"), 1.0)
    status = str(item.get("status", "active"))
    return {
        **item,
        "semantic_tier": tier,
        "planner_role": role,
        "planner_salience": round(_salience(base_salience, confidence, freshness, status), 4),
        "planner_use": _planner_use(tier),
    }


def _enrich_landmarks(nodes: list[dict[str, Any]], object_by_id: dict[str, dict[str, Any]], max_nodes: int) -> list[dict[str, Any]]:
    enriched = []
    for node in nodes:
        label = _label(node)
        tier, role, base_salience = _tier(label)
        confidence = _float(node.get("confidence"), 0.5)
        freshness = _float(node.get("freshness"), 1.0)
        status = str(node.get("status", "active"))
        source_view_ids: list[str] = []
        for object_id in node.get("source_object_ids", []):
            source_item = object_by_id.get(str(object_id))
            if source_item:
                source_view_ids.extend([str(value) for value in source_item.get("source_view_ids", [])])
        planner_salience = _salience(base_salience, confidence, freshness, status)
        enriched.append(
            {
                **node,
                "semantic_tier": tier,
                "planner_role": role,
                "planner_salience": round(planner_salience, 4),
                "planner_use": _planner_use(tier),
                "source_view_ids": sorted(set(source_view_ids)),
                "is_planner_candidate": tier != "tier0_background" and status != "missing",
            }
        )
    enriched.sort(key=lambda item: (item.get("is_planner_candidate", False), _float(item.get("planner_salience"), 0.0)), reverse=True)
    return enriched[:max_nodes]


def _build_edges(nodes: list[dict[str, Any]], near_radius_m: float) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    planner_nodes = [item for item in nodes if item.get("is_planner_candidate")]
    for left_index, left in enumerate(planner_nodes):
        for right in planner_nodes[left_index + 1 :]:
            distance = _distance(_bev(left), _bev(right))
            if distance <= near_radius_m:
                edges.append(
                    {
                        "id": f"edge_near_{len(edges):04d}",
                        "type": "near",
                        "source": left["id"],
                        "target": right["id"],
                        "distance_m": round(distance, 3),
                        "confidence": round(min(_float(left.get("confidence"), 0.5), _float(right.get("confidence"), 0.5)), 4),
                    }
                )
    for node in planner_nodes:
        for view_id in node.get("source_view_ids", [])[:6]:
            edges.append(
                {
                    "id": f"edge_visible_{len(edges):04d}",
                    "type": "visible_from_keyframe",
                    "source": str(view_id),
                    "target": node["id"],
                    "confidence": _float(node.get("confidence"), 0.5),
                }
            )
    return edges[:240]


def _candidate_waypoints(nodes: list[dict[str, Any]], radius_m: float) -> list[dict[str, Any]]:
    waypoints = []
    offsets = [(radius_m, 0.0), (-radius_m, 0.0), (0.0, radius_m), (0.0, -radius_m)]
    candidates = [node for node in nodes if node.get("is_planner_candidate")]
    candidates.sort(key=lambda item: _float(item.get("planner_salience"), 0.0), reverse=True)
    for node in candidates[:12]:
        x, y = _bev(node)
        for offset_index, (dx, dy) in enumerate(offsets):
            waypoints.append(
                {
                    "id": f"wp_{node['id']}_{offset_index}".replace(" ", "_"),
                    "anchor_landmark_id": node["id"],
                    "anchor_label": node.get("label"),
                    "type": "landmark_perimeter",
                    "bev_position": [round(x + dx, 4), round(y + dy, 4)],
                    "radius_m": round(radius_m, 3),
                    "requires_traditional_planner_validation": True,
                    "priority": round(_float(node.get("planner_salience"), 0.0), 4),
                }
            )
    return waypoints


def _select_topk(topk_retrieval: list[dict[str, Any]], goal_query: str, top_k: int) -> list[dict[str, Any]]:
    if not topk_retrieval:
        return []
    normalized_goal = goal_query.lower().replace("find ", "").replace("go to ", "").strip()
    selected = None
    for item in topk_retrieval:
        if str(item.get("query", "")).lower() == normalized_goal:
            selected = item
            break
    if selected is None:
        selected = topk_retrieval[0]
    return list(selected.get("results", []))[:top_k]


def _planner_objects(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [item for item in items if item.get("planner_use") != "background_only"]
    candidates.sort(key=lambda item: _float(item.get("planner_salience"), 0.0), reverse=True)
    fields = ["id", "label", "bev_position", "confidence", "freshness", "status", "context_id", "semantic_tier", "planner_role", "planner_salience"]
    return [{field: item.get(field) for field in fields if field in item} for item in candidates[:24]]


def _planner_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "id",
        "label",
        "aliases",
        "bev_position",
        "confidence",
        "freshness",
        "status",
        "context_id",
        "semantic_tier",
        "planner_role",
        "planner_salience",
        "is_planner_candidate",
    ]
    return [{field: item.get(field) for field in fields if field in item} for item in nodes[:32]]


def _planner_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    near_edges = [edge for edge in edges if edge.get("type") == "near"]
    visible_edges = [edge for edge in edges if edge.get("type") == "visible_from_keyframe"]
    return near_edges[:48] + visible_edges[:48]


def _compact_geometry(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "grid": summary.get("grid", {}),
        "occupancy_cells": summary.get("occupancy_cells", 0),
        "point_count": summary.get("point_count", 0),
        "planner_role": summary.get("planner_role"),
    }


def _compact_semantic(summary: dict[str, Any]) -> dict[str, Any]:
    validation = summary.get("validation") or {}
    return {
        "label_to_id": summary.get("label_to_id", {}),
        "semantic_cells": summary.get("semantic_cells", 0),
        "num_labels": summary.get("num_labels", 0),
        "precision": validation.get("precision"),
        "recall": validation.get("recall"),
        "f1": validation.get("f1"),
        "planner_role": summary.get("planner_role"),
    }


def _bundle_passed(geometry: dict[str, Any], semantic: dict[str, Any], objects: dict[str, Any], graph: dict[str, Any], context: dict[str, Any]) -> bool:
    return all(
        [
            bool(geometry.get("grid")),
            int(semantic.get("num_labels") or 0) > 0,
            len(objects.get("items", [])) > 0,
            len(graph.get("nodes", [])) > 0,
            len(context.get("candidate_waypoints", [])) > 0,
        ]
    )


def _label(item: dict[str, Any]) -> str:
    return str(item.get("label") or item.get("category") or "").strip().lower()


def _tier(label: str) -> tuple[str, str, float]:
    if label in TIER0_BACKGROUND:
        return "tier0_background", "geometry/background cue", 0.2
    if label in TIER1_CONNECTOR:
        return "tier1_connector", "connector / transition landmark", 0.95
    if label in TIER2_LANDMARK:
        return "tier2_stable_landmark", "stable navigation landmark", 0.85
    if label in TIER3_TASK_OBJECT:
        return "tier3_task_object", "query-dependent movable object", 0.65
    return "tier2_stable_landmark", "open-vocabulary landmark candidate", 0.7


def _planner_use(tier: str) -> str:
    if tier == "tier0_background":
        return "background_only"
    if tier == "tier1_connector":
        return "stopover_and_transition"
    if tier == "tier2_stable_landmark":
        return "goal_anchor"
    return "query_dependent_anchor"


def _salience(base: float, confidence: float, freshness: float, status: str) -> float:
    status_score = {"active": 1.0, "stale": 0.55, "relocated": 0.45, "missing": 0.0}.get(status, 0.5)
    return max(0.0, min(1.0, 0.45 * base + 0.28 * confidence + 0.17 * freshness + 0.10 * status_score))


def _float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _bev(item: dict[str, Any]) -> tuple[float, float]:
    value = item.get("bev_position") or item.get("centroid_xz") or [0.0, 0.0]
    return float(value[0]), float(value[1])


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _write_html(
    path: Path,
    manifest: dict[str, Any],
    geometry: dict[str, Any],
    semantic: dict[str, Any],
    objects: dict[str, Any],
    graph: dict[str, Any],
    planner_context: dict[str, Any],
    mvp_metadata: dict[str, Any],
) -> None:
    asset = manifest.get("assets", {})
    validation = semantic.get("validation") or {}
    cards = [
        ("G. Traditional BEV", asset.get("traditional_bev_phase2_style.png") or asset.get("occupancy_bev.png")),
        ("S. Semantic BEV", asset.get("semantic_bev_from_owlv2_mvp.png") or asset.get("semantic_bev.png")),
        ("O. Object Evidence", asset.get("object_inventory_projection_evidence.png")),
        ("L. Retrieval BEV", asset.get("retrieval_bev.png") or asset.get("retrieval_bev.svg")),
    ]
    card_html = "\n".join(
        f"""<section><h2>{html.escape(title)}</h2>{f'<img src="{html.escape(src)}">' if src else '<p>asset not available</p>'}</section>"""
        for title, src in cards
    )
    top_nodes = graph["nodes"][:12]
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(node.get('label')))}</td>"
        f"<td>{html.escape(str(node.get('semantic_tier')))}</td>"
        f"<td>{float(node.get('planner_salience', 0.0)):.3f}</td>"
        f"<td>{float(node.get('confidence', 0.0)):.3f}</td>"
        f"<td>{html.escape(str(node.get('status')))}</td>"
        f"<td>{html.escape(str(node.get('bev_position')))}</td>"
        "</tr>"
        for node in top_nodes
    )
    context_preview = json.dumps(
        {
            "schema_version": planner_context["schema_version"],
            "goal_query": planner_context["goal_query"],
            "context_id": planner_context["context_id"],
            "topk_landmarks": planner_context["topk_landmarks"][:3],
            "candidate_waypoints": planner_context["candidate_waypoints"][:4],
            "phase4_lite_context_state": planner_context["phase4_lite_context_state"],
        },
        indent=2,
        ensure_ascii=False,
    )
    metrics_line = (
        f"P/R/F1={validation.get('precision', 0):.3f}/{validation.get('recall', 0):.3f}/{validation.get('f1', 0):.3f}"
        if validation
        else "grounding metrics unavailable"
    )
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>M3.5 Representation Bundle</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; background: #fafafa; }}
h1 {{ margin-bottom: 4px; }}
.muted {{ color: #5f6368; }}
.pill {{ display: inline-block; padding: 4px 8px; background: #e8f0fe; color: #174ea6; border-radius: 999px; margin-right: 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; margin: 18px 0; }}
section {{ background: white; border: 1px solid #dde1e6; border-radius: 8px; padding: 14px; }}
section img {{ width: 100%; max-height: 620px; object-fit: contain; background: white; border: 1px solid #eceff3; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
th, td {{ border: 1px solid #dde1e6; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f1f3f4; }}
pre {{ background: #202124; color: #f1f3f4; padding: 14px; border-radius: 8px; overflow-x: auto; }}
a {{ color: #174ea6; }}
</style></head>
<body>
<h1>M3.5 G/S/O/L Representation Bundle</h1>
<p class="muted">Purpose: freeze the planner input contract before Phase5A API waypoint planning.</p>
<p><span class="pill">status: {html.escape(manifest['status'])}</span><span class="pill">context: {html.escape(str(manifest.get('context_id')))}</span><span class="pill">{html.escape(metrics_line)}</span></p>
<div class="grid">{card_html}</div>
<section>
<h2>Planner Contract</h2>
<p><code>planner_context.json</code> is the Phase5A input. PNGs are diagnostic views; the planner should consume structured G/S/O/L fields.</p>
<pre>{html.escape(context_preview)}</pre>
</section>
<section>
<h2>Top Planner Landmarks</h2>
<table><tr><th>Label</th><th>Tier</th><th>Salience</th><th>Confidence</th><th>Status</th><th>BEV</th></tr>{rows}</table>
</section>
<section>
<h2>Bundle Files</h2>
<ul>
<li><a href="bundle_manifest.json">bundle_manifest.json</a></li>
<li><a href="geometry_bev_summary.json">geometry_bev_summary.json</a></li>
<li><a href="semantic_evidence_summary.json">semantic_evidence_summary.json</a></li>
<li><a href="object_memory.json">object_memory.json</a></li>
<li><a href="landmark_graph.json">landmark_graph.json</a></li>
<li><a href="planner_context.json">planner_context.json</a></li>
</ul>
</section>
<section>
<h2>Notes</h2>
<ul>
<li>Wall/floor/ceiling are retained as background or geometry signals, not goal landmarks.</li>
<li>Door and connector-like nodes are high-priority transition landmarks.</li>
<li>Phase4 remapping is only represented by reserved context fields here; full mismatch scoring is next after Phase5A minimal closed loop.</li>
<li>{html.escape(str(mvp_metadata.get('caveat', 'Open-vocabulary evidence remains detector-dependent.')))}</li>
</ul>
</section>
</body></html>"""
    path.write_text(html_doc, encoding="utf-8")


if __name__ == "__main__":
    main()
