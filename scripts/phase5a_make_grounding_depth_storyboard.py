from __future__ import annotations

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phase23_habitat_control_server import HabitatControlSession, ensure_conda_nvidia_egl_vendor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Phase5A demos as RGB+grounding/depth/BEV GIFs.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dataset-config")
    parser.add_argument("--out-dir", action="append", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument(
        "--semantic-panel-image",
        default="outputs/m35_semantic_representation_alignment/representation_bundle_best96_20260703/assets/object_inventory_projection_evidence.png",
    )
    args = parser.parse_args()

    ensure_conda_nvidia_egl_vendor()
    semantic_panel_image = resolve_path(args.semantic_panel_image)
    session = HabitatControlSession(
        scene=Path(args.scene).expanduser().resolve(),
        scene_dataset_config=Path(args.scene_dataset_config).expanduser().resolve() if args.scene_dataset_config else None,
        resolution=int(args.resolution),
        move_amount=0.25,
        turn_amount=15.0,
        semantic_categories=["wall", "door", "table", "chair", "bed", "sofa"],
        enable_oracle_metrics=False,
    )
    try:
        summaries = [
            build_demo_gif(
                session,
                resolve_path(out_dir),
                semantic_panel_image,
                int(args.fps),
                int(args.max_frames),
                int(args.panel_size),
            )
            for out_dir in args.out_dir
        ]
    finally:
        session.close()
    print(json.dumps({"generated": summaries}, ensure_ascii=False, indent=2))


def build_demo_gif(
    session: HabitatControlSession,
    out_dir: Path,
    semantic_panel_image: Path,
    fps: int,
    max_frames: int,
    panel_size: int,
) -> dict[str, Any]:
    trace = read_json(out_dir / "demo_execution_trace.json")
    request = read_json(out_dir / "demo_planner_request.json")
    plan = read_json(out_dir / "demo_planner_output.json")
    route = [np.asarray(point, dtype=np.float32) for point in trace.get("route_points_xyz", [])]
    if len(route) < 2:
        raise RuntimeError(f"Missing route points in {out_dir}")
    if not semantic_panel_image.exists():
        raise FileNotFoundError(semantic_panel_image)
    goal = str(request.get("goal_query") or trace.get("goal") or "natural-language task")
    replay_specs = build_replay_specs(route, trace.get("segments", []), fps)
    replay_specs = sample_items(replay_specs, max(2, max_frames))
    goal_timeline = build_goal_timeline(replay_specs, trace.get("segments", []), plan)
    semantic_panel = Image.open(semantic_panel_image).convert("RGB")

    session.step_count = 0
    session.memory_step_count = 0
    session.memory_origin_world_xz = None
    session._reset_memory()

    frames: list[Image.Image] = []
    for index, spec in enumerate(replay_specs):
        point = np.asarray(spec["point"], dtype=np.float32)
        look_at = np.asarray(spec["look_at"], dtype=np.float32)
        session._set_agent_pose(point, look_at)
        session.step_count = index
        session.memory_step_count = index
        payload = session._state_payload()
        goal_state = goal_timeline[min(index, len(goal_timeline) - 1)]
        if spec.get("kind") == "wait":
            goal_state = dict(goal_state)
            goal_state["status_line"] = f"arrived: stop and look at {goal_state.get('anchor_label', 'target')}"
        frames.append(compose_frame(payload, goal, goal_state, semantic_panel, index + 1, len(replay_specs), panel_size))

    gif_path = out_dir / "grounding_depth_demo.gif"
    poster_path = out_dir / "grounding_depth_demo_poster.png"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
        optimize=False,
    )
    frames[-1].save(poster_path)

    metadata = {
        "gif": gif_path.name,
        "poster": poster_path.name,
        "source_trace": "demo_execution_trace.json",
        "frames": len(frames),
        "fps": int(fps),
        "stopover_wait_frames": sum(1 for item in replay_specs if item.get("kind") == "wait"),
        "semantic_panel_source": str(semantic_panel_image),
        "panels": ["RGB + grounding boxes + API goal", "Depth", "Semantic evidence", "API planner step"],
        "pose_note": "Replay uses Habitat-Sim exact pose; real-world deployment still needs SLAM/VIO/relocalization.",
    }
    (out_dir / "grounding_depth_demo_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out_dir": str(out_dir), **metadata}


def compose_frame(
    payload: dict[str, Any],
    goal: str,
    goal_state: dict[str, Any],
    semantic_panel: Image.Image,
    frame_index: int,
    frame_total: int,
    panel_size: int,
) -> Image.Image:
    margin = 16
    gap = 12
    header_h = 70
    label_h = 24
    width = margin * 2 + panel_size * 2 + gap
    height = margin * 2 + header_h + (panel_size + label_h) * 2 + gap
    canvas = Image.new("RGB", (width, height), (246, 248, 251))
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(13)
    small = load_font(11)
    draw.text((margin, margin), "RSC-Nav map-then-task demo with live grounding/depth", fill=(24, 34, 48), font=font)
    draw.text((margin, margin + 22), truncate(f"Goal: {goal}", 120), fill=(71, 84, 103), font=small)
    draw.text((width - margin - 92, margin + 22), f"{frame_index}/{frame_total}", fill=(71, 84, 103), font=small)

    rgb = decode_image(payload["rgb_jpeg"]).convert("RGB")
    draw_grounding_boxes(rgb, payload.get("grounding_boxes", []))
    draw_rgb_goal_banner(rgb, goal_state)
    depth = decode_image(payload["depth_png"]).convert("RGB")
    planner_panel = planner_panel_image(goal_state, panel_size)
    panels = [
        ("RGB + grounding box + API goal", rgb),
        ("Depth", depth),
        ("Semantic evidence / object memory", semantic_panel),
        ("API planner current stopover", planner_panel),
    ]
    for idx, (label, image) in enumerate(panels):
        row = idx // 2
        col = idx % 2
        x = margin + col * (panel_size + gap)
        y = margin + header_h + row * (panel_size + label_h + gap)
        draw.rounded_rectangle((x, y, x + panel_size, y + panel_size), radius=6, fill=(255, 255, 255), outline=(206, 216, 226))
        canvas.paste(fit_image(image, panel_size), (x, y))
        draw.text((x, y + panel_size + 7), label, fill=(37, 48, 64), font=small)
    return canvas


def draw_grounding_boxes(image: Image.Image, boxes: list[dict[str, Any]]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(max(10, image.width // 22))
    line_w = max(2, image.width // 160)
    for box in boxes[:10]:
        x = int(box.get("x", 0))
        y = int(box.get("y", 0))
        w = int(box.get("w", 0))
        h = int(box.get("h", 0))
        if w <= 2 or h <= 2:
            continue
        color = hex_to_rgba(str(box.get("color") or "#61c6a7"), 230)
        label = f"{box.get('category', 'obj')} {float(box.get('score', 0.0)):.2f}"
        draw.rectangle((x, y, x + w, y + h), outline=color, width=line_w)
        text_box = draw.textbbox((0, 0), label, font=font)
        label_w = text_box[2] - text_box[0] + 8
        label_h = text_box[3] - text_box[1] + 7
        label_y = max(0, y - label_h)
        draw.rectangle((x, label_y, x + label_w, label_y + label_h), fill=color)
        draw.text((x + 4, label_y + 2), label, fill=(5, 18, 14, 255), font=font)


def draw_rgb_goal_banner(image: Image.Image, goal_state: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(max(11, image.width // 23))
    small = load_font(max(9, image.width // 30))
    banner_h = max(46, image.height // 5)
    draw.rectangle((0, 0, image.width, banner_h), fill=(0, 0, 0, 158))
    step = int(goal_state.get("step", 1))
    total = int(goal_state.get("total_steps", 1))
    intent = str(goal_state.get("intent_label", "API goal"))
    anchor = str(goal_state.get("anchor_label", "unknown"))
    waypoint = str(goal_state.get("waypoint_id", ""))
    line1 = f"API goal {step}/{total}: {intent} -> {anchor}"
    status = str(goal_state.get("status_line") or f"waypoint: {short_id(waypoint)}")
    line2 = status
    draw.text((8, 7), truncate(line1, 58), fill=(255, 255, 255, 255), font=font)
    draw.text((8, 27), truncate(line2, 72), fill=(180, 230, 255, 255), font=small)


def planner_panel_image(goal_state: dict[str, Any], panel_size: int) -> Image.Image:
    image = Image.new("RGB", (panel_size, panel_size), (253, 254, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(max(13, panel_size // 20))
    font = load_font(max(11, panel_size // 25))
    small = load_font(max(10, panel_size // 29))
    accent = (31, 119, 180, 255)
    draw.rectangle((0, 0, panel_size, 42), fill=(238, 245, 255, 255))
    draw.text((12, 12), "API planner sequence", fill=(24, 34, 48), font=title_font)
    step = int(goal_state.get("step", 1))
    total = int(goal_state.get("total_steps", 1))
    y = 58
    for item in goal_state.get("all_goals", []):
        active = int(item.get("step", 0)) == step
        color = accent if active else (130, 144, 164, 255)
        fill = (229, 243, 255, 255) if active else (247, 249, 252, 255)
        draw.rounded_rectangle((12, y, panel_size - 12, y + 54), radius=7, fill=fill, outline=color, width=2 if active else 1)
        draw.text((24, y + 8), f"{item.get('step')}/{total} {item.get('intent_label')}", fill=(24, 34, 48), font=font)
        draw.text((24, y + 29), f"anchor: {item.get('anchor_label')}  {short_id(str(item.get('waypoint_id', '')))}", fill=(71, 84, 103), font=small)
        y += 64

    reason = str(goal_state.get("reason", "") or goal_state.get("water_place_reasoning", ""))
    status = str(goal_state.get("status_line") or "")
    if status:
        y += 2
        draw.rounded_rectangle((12, y, panel_size - 12, y + 30), radius=6, fill=(236, 253, 245, 255), outline=(52, 168, 83, 255))
        draw.text((22, y + 8), truncate(status, 48), fill=(22, 101, 52), font=small)
        y += 42
    if reason:
        y += 2
        draw.text((14, y), "Planner reasoning:", fill=(24, 34, 48), font=font)
        y += 22
        for line in wrap_text(reason, 42)[:6]:
            draw.text((14, y), line, fill=(71, 84, 103), font=small)
            y += 17
    return image


def build_goal_timeline(route: list[np.ndarray], segments: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    if route and isinstance(route[0], dict):
        specs = route
        route_points = [np.asarray(item["point"], dtype=np.float32) for item in specs]
    else:
        specs = []
        route_points = [np.asarray(item, dtype=np.float32) for item in route]
    if not segments:
        return [
            {
                "step": 1,
                "total_steps": 1,
                "intent_label": "execute task",
                "anchor_label": "target",
                "waypoint_id": "",
                "reason": str(plan.get("reason", "")),
                "all_goals": [],
            }
            for _ in route_points
        ]
    goals = []
    for idx, segment in enumerate(segments, start=1):
        intent = str(segment.get("intent", "goal"))
        label = {
            "find_water_place": "find likely water place",
            "return_to_owner": "return to owner",
        }.get(intent, intent.replace("_", " "))
        goals.append(
            {
                "step": idx,
                "total_steps": len(segments),
                "intent": intent,
                "intent_label": label,
                "anchor_label": str(segment.get("anchor_label") or "unknown"),
                "waypoint_id": str(segment.get("waypoint_id") or ""),
                "reason": str(plan.get("reason", "")),
                "water_place_reasoning": str(plan.get("water_place_reasoning", "")),
            }
        )

    segment_distances = [float(item.get("geodesic_distance_m") or 0.0) for item in segments]
    if not any(value > 0 for value in segment_distances):
        segment_distances = [1.0 for _ in segments]
    cumulative_limits = np.cumsum(segment_distances) / max(1e-6, float(sum(segment_distances)))
    route_progress = route_fraction(route_points)
    timeline = []
    for idx, progress in enumerate(route_progress):
        explicit_segment = int(specs[idx].get("segment", 0)) if specs else 0
        if explicit_segment > 0:
            goal_index = min(explicit_segment - 1, len(goals) - 1)
        else:
            goal_index = int(np.searchsorted(cumulative_limits, progress, side="right"))
            goal_index = min(goal_index, len(goals) - 1)
        item = dict(goals[goal_index])
        item["all_goals"] = goals
        timeline.append(item)
    return timeline


def build_replay_specs(route: list[np.ndarray], segments: list[dict[str, Any]], fps: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    arrival_by_index: dict[int, dict[str, Any]] = {}
    for segment in segments:
        target = segment.get("snapped_target_xyz")
        if not target:
            continue
        target_arr = np.asarray(target, dtype=np.float32)
        nearest = min(range(len(route)), key=lambda idx: float(np.linalg.norm(np.asarray(route[idx]) - target_arr)))
        arrival_by_index[nearest] = segment

    for index, point in enumerate(route):
        point = np.asarray(point, dtype=np.float32)
        look_at = next_look_at(route, index)
        specs.append({"point": point, "look_at": look_at, "kind": "move", "segment": 0})
        segment = arrival_by_index.get(index)
        if not segment:
            continue
        wait_seconds = float(segment.get("arrival_observation_wait_seconds") or 2.0)
        wait_frames = max(4, int(round(wait_seconds * max(1, int(fps)))))
        arrival_look_at = np.asarray(segment.get("arrival_look_at_xyz") or look_at, dtype=np.float32)
        for _ in range(wait_frames):
            specs.append(
                {
                    "point": point,
                    "look_at": arrival_look_at,
                    "kind": "wait",
                    "segment": int(segment.get("segment") or 0),
                }
            )
    return specs


def route_fraction(route: list[np.ndarray]) -> list[float]:
    if len(route) <= 1:
        return [0.0]
    distances = [0.0]
    for prev, cur in zip(route[:-1], route[1:]):
        distances.append(distances[-1] + float(np.linalg.norm(np.asarray(cur) - np.asarray(prev))))
    total = max(1e-6, distances[-1])
    return [float(value / total) for value in distances]


def decode_image(value: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(value)))


def fit_image(image: Image.Image, size: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (size, size), (250, 252, 254))
    out.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return out


def sample_items(items: list[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return items
    indexes = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[int(index)] for index in indexes]


def next_look_at(route: list[np.ndarray], index: int) -> np.ndarray:
    if index + 1 < len(route):
        return route[index + 1]
    if index > 0:
        return route[index - 1]
    return route[index] + np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ):
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def hex_to_rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        return (97, 198, 167, alpha)
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), alpha)


def truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "..."


def short_id(value: str) -> str:
    if len(value) <= 34:
        return value
    return value[:16] + "..." + value[-12:]


def wrap_text(value: str, width: int) -> list[str]:
    words = value.replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = word if not current else current + " " + word
        if len(test) <= width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    main()
