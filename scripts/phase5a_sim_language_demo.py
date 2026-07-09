from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phase23_habitat_control_server import HabitatControlSession, _rgb_array, ensure_conda_nvidia_egl_vendor  # noqa: E402


DEFAULT_TABLE_CASE = "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_table_20260703"
DEFAULT_BED_CASE = "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_bed_20260703"
DEFAULT_CASE_DIRS = [
    "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_bed_20260703",
    "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_chair_20260703",
    "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_door_20260703",
    "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_sofa_20260703",
    "outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_find_table_20260703",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a Habitat first-person MVP demo from natural language API planning.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dataset-config")
    parser.add_argument("--out-dir", default="outputs/phase5a_sim_demo/water_then_owner_bed_20260704")
    parser.add_argument("--case-dir", action="append", dest="case_dirs", help="Planner request dir used to collect semantic waypoints. Defaults to 5 no-leak find cases.")
    parser.add_argument("--table-case-dir", default=DEFAULT_TABLE_CASE, help=argparse.SUPPRESS)
    parser.add_argument("--bed-case-dir", default=DEFAULT_BED_CASE, help=argparse.SUPPRESS)
    parser.add_argument("--goal", default="去找到有水的地方，然后回到主人（在床上）身边")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "qwen3-max"))
    parser.add_argument("--mode", choices=["api", "deterministic", "auto"], default="auto")
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--start-xz", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--step-m", type=float, default=0.18)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=260)
    parser.add_argument("--wait-seconds", type=float, default=2.0, help="Observation wait time after each reached stopover.")
    args = parser.parse_args()

    ensure_conda_nvidia_egl_vendor()
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = [Path(item).expanduser().resolve() for item in (args.case_dirs or DEFAULT_CASE_DIRS)]
    planner_requests = [_read_json(case_dir / "planner_request.json") for case_dir in case_dirs]
    demo_request = _build_demo_request(args.goal, planner_requests)
    (out_dir / "demo_planner_request.json").write_text(json.dumps(demo_request, ensure_ascii=False, indent=2), encoding="utf-8")

    mode = _select_mode(args.mode, args.api_key_env)
    api_error = None
    if mode == "api":
        try:
            demo_plan = _call_api_plan(demo_request, args.api_base, os.getenv(args.api_key_env, ""), args.model, float(args.timeout_s))
        except Exception as exc:  # noqa: BLE001 - keep the demo runnable and record the fallback.
            if args.mode == "api":
                raise
            api_error = str(exc)
            mode = "deterministic"
            demo_plan = _deterministic_plan(demo_request)
    else:
        demo_plan = _deterministic_plan(demo_request)
    demo_plan = _normalize_plan(demo_plan, demo_request)
    (out_dir / "demo_planner_output.json").write_text(json.dumps(demo_plan, ensure_ascii=False, indent=2), encoding="utf-8")

    session = HabitatControlSession(
        scene=Path(args.scene).expanduser().resolve(),
        scene_dataset_config=Path(args.scene_dataset_config).expanduser().resolve() if args.scene_dataset_config else None,
        resolution=args.resolution,
        move_amount=0.25,
        turn_amount=15.0,
        semantic_categories=["wall", "door", "table", "chair", "bed", "sofa"],
    )
    try:
        trace = _record_demo(
            session=session,
            demo_request=demo_request,
            demo_plan=demo_plan,
            frames_dir=frames_dir,
            start_xz=tuple(args.start_xz),
            step_m=float(args.step_m),
            fps=int(args.fps),
            max_frames=int(args.max_frames),
            wait_seconds=float(args.wait_seconds),
        )
    finally:
        session.close()

    trace["mode_used"] = mode
    trace["api_error"] = api_error
    trace["goal"] = args.goal
    trace["model"] = args.model if mode == "api" else None
    (out_dir / "demo_execution_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(out_dir, trace, demo_plan)
    print(json.dumps({"status": "passed" if trace["reachable"] else "failed", "out_dir": str(out_dir), **trace["summary"]}, ensure_ascii=False, indent=2))


def _build_demo_request(goal: str, planner_requests: list[dict[str, Any]]) -> dict[str, Any]:
    waypoints = _merge_by_id(
        item
        for request in planner_requests
        for item in request.get("candidate_waypoints", [])
    )
    landmarks = _merge_by_id(
        item
        for request in planner_requests
        for item in request.get("topk_landmarks", [])
    )
    objects = _merge_by_id(
        item
        for request in planner_requests
        for item in request.get("object_memory_summary", [])
    )
    label_counts: dict[str, int] = {}
    for item in objects:
        label = str(item.get("label", "unknown")).lower()
        label_counts[label] = label_counts.get(label, 0) + 1
    return {
        "schema_version": "phase5a_sim_language_demo_request_v1",
        "goal_query": goal,
        "affordance_note": (
            "The planner must infer where water is likely to be from the semantic map. "
            "Possible real-world anchors include bathroom, kitchen, sink, faucet, fridge, water dispenser, cup, bottle, counter, and table. "
            "If explicit water-related anchors are absent from the current map, choose the best available search region and state the uncertainty."
        ),
        "semantic_map_summary": {
            "available_labels": sorted(label_counts),
            "label_counts": label_counts,
            "known_limitation": "Current MVP scene labels may not include bottle/cup/sink/fridge/bathroom/kitchen, so water-place planning may be uncertain.",
        },
        "required_sequence": [
            {"intent": "find_water_place", "preferred_anchor_label": "planner_infers_from_semantic_map"},
            {"intent": "return_to_owner", "preferred_anchor_label": "bed"},
        ],
        "candidate_waypoints": waypoints[:80],
        "topk_landmarks": landmarks[:24],
        "object_memory_summary": objects[:40],
        "required_output_schema": {
            "task_plan": [{"step": 1, "intent": "find_water_place", "target": "landmark_id"}],
            "water_waypoint_id": "waypoint_id",
            "owner_waypoint_id": "waypoint_id",
            "stopover_waypoints": ["water_waypoint_id", "owner_waypoint_id"],
            "water_place_reasoning": "why this semantic map element is the best available water-place candidate",
            "reason": "short explanation",
        },
        "constraints": [
            "Return JSON only.",
            "Infer the water-place waypoint from semantic map objects and landmarks; do not assume a fixed label if better evidence exists.",
            "Use a bed waypoint for returning to the owner.",
            "Use only candidate_waypoints ids.",
        ],
    }


def _merge_by_id(items) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or item.get("landmark_id") or "")
        if item_id and item_id not in out:
            out[item_id] = item
    return list(out.values())


def _filter_waypoints(request: dict[str, Any], label: str) -> list[dict[str, Any]]:
    out = []
    for item in request.get("candidate_waypoints", []):
        if str(item.get("anchor_label", "")).lower() == label:
            out.append(item)
    return out


def _call_api_plan(payload: dict[str, Any], api_base: str, api_key: str, model: str, timeout_s: float) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("API key is missing.")
    prompt = (
        "You are the RSC-Nav Phase5A semantic task planner. The user goal is Chinese natural language. "
        "Infer likely water-place candidates from the semantic map, then return to the owner on the bed. "
        "Do not hard-code table: bathrooms, kitchens, sinks, fridges, cups, bottles, dispensers, counters, or tables may be relevant depending on available map evidence. "
        "Return JSON only and use only candidate waypoint ids. If the map lacks explicit water-related objects, state uncertainty in water_place_reasoning.\n\n"
        f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    endpoint = api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - user-configured API endpoint.
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {detail[:1000]}") from exc
    content = raw["choices"][0]["message"]["content"]
    return _extract_json(content)


def _extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _deterministic_plan(payload: dict[str, Any]) -> dict[str, Any]:
    table = _best_available_water_waypoint(payload)
    bed = next(item for item in payload["candidate_waypoints"] if item.get("anchor_label") == "bed")
    return {
        "task_plan": [
            {"step": 1, "intent": "find_water_place", "target": table.get("anchor_landmark_id")},
            {"step": 2, "intent": "return_to_owner", "target": bed.get("anchor_landmark_id")},
        ],
        "water_waypoint_id": table["id"],
        "owner_waypoint_id": bed["id"],
        "stopover_waypoints": [table["id"], bed["id"]],
        "water_place_reasoning": "Deterministic fallback ranked explicit water-related labels first; none were available, so it selected the best available surface/room proxy.",
        "reason": "Deterministic MVP fallback: infer water-place from available labels, then return to bed.",
    }


def _normalize_plan(plan: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    valid = {str(item.get("id")): item for item in payload.get("candidate_waypoints", [])}
    water = str(plan.get("water_waypoint_id", ""))
    owner = str(plan.get("owner_waypoint_id", ""))
    if water not in valid or valid[water].get("anchor_label") == "bed":
        water = _best_available_water_waypoint(payload)["id"]
    if owner not in valid or valid[owner].get("anchor_label") != "bed":
        owner = next(item["id"] for item in payload["candidate_waypoints"] if item.get("anchor_label") == "bed")
    return {
        "task_plan": plan.get("task_plan") or _deterministic_plan(payload)["task_plan"],
        "water_waypoint_id": water,
        "owner_waypoint_id": owner,
        "stopover_waypoints": [water, owner],
        "water_place_reasoning": str(plan.get("water_place_reasoning", ""))[:1200],
        "reason": str(plan.get("reason", ""))[:1200],
    }


def _best_available_water_waypoint(payload: dict[str, Any]) -> dict[str, Any]:
    priority = {
        "sink": 100,
        "faucet": 98,
        "water dispenser": 96,
        "dispenser": 95,
        "fridge": 90,
        "refrigerator": 90,
        "bottle": 88,
        "cup": 86,
        "kitchen": 82,
        "bathroom": 80,
        "counter": 72,
        "table": 65,
        "door": 25,
        "chair": 10,
        "sofa": 8,
        "bed": -100,
    }
    candidates = []
    for item in payload["candidate_waypoints"]:
        label = str(item.get("anchor_label", "")).lower()
        score = priority.get(label, 30)
        score += 0.1 * float(item.get("priority", 0.0) or 0.0)
        candidates.append((score, item))
    candidates.sort(key=lambda row: row[0], reverse=True)
    return candidates[0][1]


def _record_demo(
    session: HabitatControlSession,
    demo_request: dict[str, Any],
    demo_plan: dict[str, Any],
    frames_dir: Path,
    start_xz: tuple[float, float],
    step_m: float,
    fps: int,
    max_frames: int,
    wait_seconds: float,
) -> dict[str, Any]:
    import habitat_sim

    pathfinder = session.sim.pathfinder
    waypoint_by_id = {str(item.get("id")): item for item in demo_request["candidate_waypoints"]}
    landmark_by_id = {
        str(item.get("landmark_id") or item.get("id")): item
        for item in demo_request.get("topk_landmarks", [])
    }
    waypoints = [waypoint_by_id[item] for item in demo_plan["stopover_waypoints"]]
    ref_y = float(np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)[1])
    current = _snap(pathfinder, [float(start_xz[0]), ref_y, float(start_xz[1])])
    all_route_points = [current]
    arrival_events: dict[int, dict[str, Any]] = {}
    segments = []
    reachable = True
    for index, waypoint in enumerate(waypoints, start=1):
        requested = [float(waypoint["bev_position"][0]), ref_y, float(waypoint["bev_position"][1])]
        target = _snap(pathfinder, requested)
        shortest_path = habitat_sim.ShortestPath()
        shortest_path.requested_start = current
        shortest_path.requested_end = target
        ok = bool(pathfinder.find_path(shortest_path))
        reachable = reachable and ok
        points = [np.asarray(point, dtype=np.float32) for point in shortest_path.points] if ok else [current, target]
        sampled = _resample_polyline(points, step_m)
        if all_route_points and sampled:
            sampled = sampled[1:]
        all_route_points.extend(sampled)
        arrival_index = len(all_route_points) - 1
        intent = "find_water_place" if index == 1 else "return_to_owner"
        anchor_look_at = _anchor_look_at(waypoint, landmark_by_id, ref_y, target)
        arrival_events[arrival_index] = {
            "segment": index,
            "intent": intent,
            "anchor_label": waypoint.get("anchor_label"),
            "waypoint_id": waypoint["id"],
            "look_at": anchor_look_at,
            "stage": (
                f"observe target: likely water place ({waypoint.get('anchor_label')})"
                if index == 1
                else f"observe target: owner location near {waypoint.get('anchor_label')}"
            ),
        }
        segments.append(
            {
                "segment": index,
                "intent": intent,
                "waypoint_id": waypoint["id"],
                "anchor_label": waypoint.get("anchor_label"),
                "requested_xyz": requested,
                "snapped_target_xyz": _point_list(target),
                "arrival_observation_wait_seconds": float(wait_seconds),
                "arrival_look_at_xyz": _point_list(anchor_look_at),
                "reachable": ok,
                "geodesic_distance_m": round(float(shortest_path.geodesic_distance), 4) if ok else None,
                "num_path_points": len(points),
            }
        )
        current = target

    wait_frames = max(0, int(round(float(wait_seconds) * max(1, int(fps)))))
    frame_specs = []
    for point_index, point in enumerate(all_route_points):
        look_at = all_route_points[min(point_index + 1, len(all_route_points) - 1)]
        if np.linalg.norm(np.asarray(look_at) - np.asarray(point)) < 1e-4 and point_index > 0:
            look_at = all_route_points[point_index - 1]
        frame_specs.append(
            {
                "point": np.asarray(point, dtype=np.float32),
                "look_at": np.asarray(look_at, dtype=np.float32),
                "stage": _stage_label(point_index, len(all_route_points)),
                "kind": "move",
            }
        )
        if point_index in arrival_events:
            event = arrival_events[point_index]
            for _ in range(wait_frames):
                frame_specs.append(
                    {
                        "point": np.asarray(point, dtype=np.float32),
                        "look_at": np.asarray(event["look_at"], dtype=np.float32),
                        "stage": event["stage"],
                        "kind": "wait",
                        "segment": event["segment"],
                    }
                )

    if len(frame_specs) > max_frames:
        idxs = np.linspace(0, len(frame_specs) - 1, max_frames).round().astype(int)
        frame_specs = [frame_specs[int(i)] for i in idxs]

    frame_paths = []
    for frame_index, spec in enumerate(frame_specs):
        point = spec["point"]
        look_at = spec["look_at"]
        if np.linalg.norm(np.asarray(look_at) - np.asarray(point)) < 1e-4:
            look_at = np.asarray(point, dtype=np.float32) + np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        session._set_agent_pose(np.asarray(point, dtype=np.float32), np.asarray(look_at, dtype=np.float32))
        observations = session.sim.get_sensor_observations()
        rgb = Image.fromarray(_rgb_array(observations.get("rgb"))).convert("RGB")
        stage = str(spec["stage"])
        goal_text = str(demo_request.get("goal_query") or "execute natural-language task")
        rgb = _overlay(rgb, f"Goal: {goal_text}", stage)
        frame_path = frames_dir / f"frame_{frame_index:04d}.jpg"
        rgb.save(frame_path, quality=94)
        frame_paths.append(frame_path)

    video_path, video_kind = _write_video(frames_dir.parent, frame_paths, fps)
    summary = {
        "num_frames": len(frame_paths),
        "video": str(video_path),
        "video_kind": video_kind,
        "reachable_segments": sum(1 for item in segments if item["reachable"]),
        "total_segments": len(segments),
        "wait_seconds_per_stopover": float(wait_seconds),
        "wait_frames_per_stopover": int(wait_frames),
    }
    return {
        "reachable": bool(reachable),
        "segments": segments,
        "route_points_xyz": [_point_list(point) for point in all_route_points],
        "num_wait_frames": sum(1 for item in frame_specs if item.get("kind") == "wait"),
        "frame_paths": [str(path) for path in frame_paths],
        "summary": summary,
    }


def _anchor_look_at(waypoint: dict[str, Any], landmark_by_id: dict[str, dict[str, Any]], ref_y: float, fallback_target: np.ndarray) -> np.ndarray:
    anchor_id = str(waypoint.get("anchor_landmark_id", ""))
    landmark = landmark_by_id.get(anchor_id)
    if landmark and isinstance(landmark.get("bev_position"), list) and len(landmark["bev_position"]) >= 2:
        return np.asarray([float(landmark["bev_position"][0]), float(ref_y), float(landmark["bev_position"][1])], dtype=np.float32)
    if isinstance(waypoint.get("bev_position"), list) and len(waypoint["bev_position"]) >= 2:
        return np.asarray([float(waypoint["bev_position"][0]), float(ref_y), float(waypoint["bev_position"][1])], dtype=np.float32)
    return np.asarray(fallback_target, dtype=np.float32)


def _stage_label(index: int, total: int) -> str:
    if total <= 1:
        return "stage: initialize"
    ratio = index / max(1, total - 1)
    if ratio < 0.48:
        return "stage 1/2: navigate to API-selected likely water place"
    return "stage 2/2: return to owner near bed"


def _overlay(image: Image.Image, line1: str, line2: str) -> Image.Image:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_ui_font(13)
    pad = 8
    box_h = 44
    draw.rectangle((0, 0, image.width, box_h), fill=(0, 0, 0, 150))
    draw.text((pad, 7), line1, fill=(255, 255, 255, 255), font=font)
    draw.text((pad, 25), line2, fill=(170, 225, 255, 255), font=font)
    return image


def _load_ui_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _write_video(out_dir: Path, frame_paths: list[Path], fps: int) -> tuple[Path, str]:
    try:
        import imageio.v2 as imageio

        mp4 = out_dir / "water_then_owner_bed_first_person.mp4"
        frames = [imageio.imread(path) for path in frame_paths]
        imageio.mimsave(mp4, frames, fps=max(1, int(fps)), macro_block_size=16)
        return mp4, "mp4"
    except Exception:
        gif = out_dir / "water_then_owner_bed_first_person.gif"
        frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(1000 / max(1, fps)), loop=0)
        return gif, "gif"


def _write_html(out_dir: Path, trace: dict[str, Any], plan: dict[str, Any]) -> None:
    video = Path(trace["summary"]["video"]).name
    if trace["summary"]["video_kind"] == "mp4":
        media = f'<video controls autoplay muted loop src="{html.escape(video)}" style="max-width:100%;border:1px solid #d8e0ea"></video>'
    else:
        media = f'<img src="{html.escape(video)}" style="max-width:100%;border:1px solid #d8e0ea">'
    rows = "".join(
        "<tr>"
        f"<td>{item['segment']}</td><td>{html.escape(item['intent'])}</td><td>{html.escape(item['anchor_label'] or '')}</td>"
        f"<td>{html.escape(item['waypoint_id'])}</td><td>{'yes' if item['reachable'] else 'no'}</td><td>{item.get('geodesic_distance_m')}</td>"
        "</tr>"
        for item in trace["segments"]
    )
    html_doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>RSC-Nav Water Demo MVP</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202124;line-height:1.5}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8e0ea;padding:8px;text-align:left}}th{{background:#eef3f8}}pre{{background:#f6f8fa;padding:12px;overflow:auto}}</style></head>
<body><h1>RSC-Nav MVP Demo: Find Water Place, Return to Owner</h1>
<p>自然语言输入：<b>去找到有水的地方，然后回到主人（在床上）身边</b></p>
<p>当前 MVP 边界：planner 必须基于当前语义 map 自行判断哪里可能有水；如果 map 中没有 bottle/cup/sink/fridge/bathroom/kitchen 等显式水相关标签，planner 会在现有语义元素里选择最可能的搜索点并说明不确定性。主人位置映射为 bed。</p>
{media}
<h2>API / Planner Output</h2><pre>{html.escape(json.dumps(plan, ensure_ascii=False, indent=2))}</pre>
<h2>Navmesh Execution</h2><table><tr><th>#</th><th>Intent</th><th>Anchor</th><th>Waypoint</th><th>Reachable</th><th>Geodesic m</th></tr>{rows}</table>
<p>Trace JSON: <a href="demo_execution_trace.json">demo_execution_trace.json</a></p>
</body></html>"""
    (out_dir / "demo_report.html").write_text(html_doc, encoding="utf-8")


def _select_mode(mode: str, api_key_env: str) -> str:
    if mode == "deterministic":
        return "deterministic"
    if mode == "api":
        return "api"
    return "api" if os.getenv(api_key_env) else "deterministic"


def _snap(pathfinder, xyz: list[float]) -> np.ndarray:
    return np.asarray(pathfinder.snap_point(np.asarray(xyz, dtype=np.float32)), dtype=np.float32)


def _resample_polyline(points: list[np.ndarray], step_m: float) -> list[np.ndarray]:
    if len(points) <= 1:
        return points
    cumulative = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(np.asarray(b) - np.asarray(a))))
    total = cumulative[-1]
    if total <= 1e-6:
        return [points[0], points[-1]]
    count = max(2, int(math.ceil(total / max(0.05, step_m))) + 1)
    targets = np.linspace(0.0, total, count)
    out = []
    seg = 0
    for target in targets:
        while seg + 1 < len(cumulative) and cumulative[seg + 1] < target:
            seg += 1
        if seg + 1 >= len(points):
            out.append(np.asarray(points[-1], dtype=np.float32))
            continue
        span = cumulative[seg + 1] - cumulative[seg]
        alpha = 0.0 if span <= 0 else (target - cumulative[seg]) / span
        out.append((1.0 - alpha) * np.asarray(points[seg]) + alpha * np.asarray(points[seg + 1]))
    return [np.asarray(point, dtype=np.float32) for point in out]


def _point_list(point: Any) -> list[float]:
    arr = np.asarray(point, dtype=np.float32)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
