from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a linked GIF/HTML storyboard for Phase5A: map first, then task execution.")
    parser.add_argument("--out-dir", default="outputs/phase5a_sim_demo/water_then_owner_bed_20260704")
    parser.add_argument(
        "--exploration-images",
        default="outputs/phase213_episode_runs/20260625-170940_phase213_hd384_pitch_sweep_depth_coverage/images",
    )
    parser.add_argument("--execution-frames", default=None)
    parser.add_argument(
        "--semantic-panel-image",
        default="outputs/m35_semantic_representation_alignment/representation_bundle_best96_20260703/assets/object_inventory_projection_evidence.png",
        help="Object-centric semantic map/evidence image used in the linked storyboard.",
    )
    parser.add_argument("--map-frame-count", type=int, default=64)
    parser.add_argument("--task-frame-count", type=int, default=72)
    parser.add_argument("--panel-size", type=int, default=420)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    exploration_dir = (ROOT / args.exploration_images).resolve() if not Path(args.exploration_images).is_absolute() else Path(args.exploration_images)
    execution_dir = Path(args.execution_frames or out_dir / "frames")
    execution_dir = (ROOT / execution_dir).resolve() if not execution_dir.is_absolute() else execution_dir
    semantic_panel_image = Path(args.semantic_panel_image)
    semantic_panel_image = (ROOT / semantic_panel_image).resolve() if not semantic_panel_image.is_absolute() else semantic_panel_image
    out_dir.mkdir(parents=True, exist_ok=True)

    storyboard = build_storyboard(
        out_dir=out_dir,
        exploration_dir=exploration_dir,
        execution_dir=execution_dir,
        semantic_panel_image=semantic_panel_image,
        map_frame_count=max(4, args.map_frame_count),
        task_frame_count=max(4, args.task_frame_count),
        panel_size=max(220, args.panel_size),
        fps=max(1, args.fps),
    )
    write_story_html(out_dir, storyboard)
    print(json.dumps(storyboard, ensure_ascii=False, indent=2))


def build_storyboard(
    out_dir: Path,
    exploration_dir: Path,
    execution_dir: Path,
    semantic_panel_image: Path,
    map_frame_count: int,
    task_frame_count: int,
    panel_size: int,
    fps: int,
) -> dict:
    rgb_paths = sample_paths(sorted(exploration_dir.glob("step_*_rgb.jpg")), map_frame_count)
    bev_paths = sample_paths(sorted(exploration_dir.glob("step_*_bev.png")), map_frame_count)
    semantic_paths = sample_paths(sorted(exploration_dir.glob("step_*_semantic_bev.png")), map_frame_count)
    exec_paths = sample_paths(sorted(execution_dir.glob("frame_*.jpg")), task_frame_count)
    if not rgb_paths or not bev_paths:
        raise FileNotFoundError(f"Missing exploration RGB/BEV frames under {exploration_dir}")
    if not exec_paths:
        raise FileNotFoundError(f"Missing execution frames under {execution_dir}")
    if not semantic_panel_image.exists():
        raise FileNotFoundError(f"Missing semantic panel image: {semantic_panel_image}")

    final_rgb = rgb_paths[-1]
    final_bev = bev_paths[-1]
    final_semantic = semantic_panel_image
    frames: list[Image.Image] = []

    for index in range(map_frame_count):
        frames.append(
            compose_frame(
                rgb_paths[min(index, len(rgb_paths) - 1)],
                bev_paths[min(index, len(bev_paths) - 1)],
                semantic_panel_image,
                None,
                panel_size,
                title="Stage 1: enter new room, build reusable semantic map",
                phase="coverage-loop exploration MVP",
                frame_index=index + 1,
                frame_total=map_frame_count + task_frame_count,
            )
        )

    for index in range(task_frame_count):
        frames.append(
            compose_frame(
                final_rgb,
                final_bev,
                final_semantic,
                exec_paths[min(index, len(exec_paths) - 1)],
                panel_size,
                title="Stage 2: natural-language task planning and execution",
                phase="API planner -> navmesh path -> first-person replay",
                frame_index=map_frame_count + index + 1,
                frame_total=map_frame_count + task_frame_count,
            )
        )

    gif_path = out_dir / "semantic_map_then_task_linked.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )

    poster = out_dir / "semantic_map_then_task_poster.png"
    frames[-1].save(poster)
    return {
        "storyboard_gif": gif_path.name,
        "poster_png": poster.name,
        "exploration_source": str(exploration_dir),
        "execution_source": str(execution_dir),
        "semantic_panel_source": str(semantic_panel_image),
        "map_frame_count": map_frame_count,
        "task_frame_count": task_frame_count,
        "fps": fps,
        "panel_size": panel_size,
        "narrative": [
            "Stage 1 uses the existing coverage-loop traversal as the MVP substitute for future curiosity/interest-driven exploration.",
            "Stage 2 freezes the constructed map as reusable semantic memory, sends the natural-language command to the API planner, and visualizes the executable Habitat/navmesh route.",
        ],
    }


def sample_paths(paths: list[Path], count: int) -> list[Path]:
    if not paths:
        return []
    if len(paths) <= count:
        return paths
    if count <= 1:
        return [paths[-1]]
    indexes = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[int(index)] for index in indexes]


def compose_frame(
    rgb_path: Path,
    bev_path: Path,
    semantic_path: Path,
    execution_path: Path | None,
    panel_size: int,
    title: str,
    phase: str,
    frame_index: int,
    frame_total: int,
) -> Image.Image:
    margin = 18
    header_h = 74
    gap = 14
    label_h = 30
    width = margin * 2 + panel_size * 2 + gap
    height = margin * 2 + header_h + (panel_size + label_h) * 2 + gap
    canvas = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    font = load_ui_font(13)
    title_font = font
    draw.text((margin, margin), title, fill=(26, 36, 52), font=title_font)
    draw.text((margin, margin + 22), phase, fill=(82, 94, 111), font=font)
    draw.text((width - margin - 130, margin + 22), f"{frame_index}/{frame_total}", fill=(82, 94, 111), font=font)

    panels = [
        ("First-person RGB during mapping", rgb_path),
        ("Traditional BEV / occupancy", bev_path),
        ("Object-centric semantic evidence / centroids", semantic_path),
        ("Task execution after command", execution_path),
    ]
    origins = [
        (margin, margin + header_h),
        (margin + panel_size + gap, margin + header_h),
        (margin, margin + header_h + panel_size + label_h + gap),
        (margin + panel_size + gap, margin + header_h + panel_size + label_h + gap),
    ]
    for (label, path), origin in zip(panels, origins):
        x, y = origin
        draw.rounded_rectangle((x, y, x + panel_size, y + panel_size), radius=8, fill=(255, 255, 255), outline=(207, 216, 226))
        if path is None:
            draw_placeholder(canvas, (x, y), panel_size)
        else:
            image = fit_image(Image.open(path).convert("RGB"), panel_size)
            canvas.paste(image, (x, y))
        draw.text((x, y + panel_size + 8), label, fill=(37, 48, 64), font=font)
    return canvas


def fit_image(image: Image.Image, size: int) -> Image.Image:
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (size, size), (250, 252, 254))
    out.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return out


def draw_placeholder(canvas: Image.Image, origin: tuple[int, int], size: int) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = origin
    draw.rectangle((x + 1, y + 1, x + size - 1, y + size - 1), fill=(239, 243, 248))
    font = load_ui_font(13)
    lines = ["Waiting for user command", "", "Natural language", "task will start next"]
    yy = y + size // 2 - 34
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((x + (size - (bbox[2] - bbox[0])) // 2, yy), line, fill=(88, 101, 119), font=font)
        yy += 18


def write_story_html(out_dir: Path, storyboard: dict) -> None:
    report_path = out_dir / "demo_report.html"
    plan_path = out_dir / "demo_planner_output.json"
    trace_path = out_dir / "demo_execution_trace.json"
    plan = read_json(plan_path)
    trace = read_json(trace_path)
    request = read_json(out_dir / "demo_planner_request.json")
    goal = str(request.get("goal_query") or trace.get("goal") or "natural-language task")
    video = Path(trace.get("summary", {}).get("video") or "water_then_owner_bed_first_person.mp4").name
    rows = "".join(
        "<tr>"
        f"<td>{item['segment']}</td><td>{html.escape(str(item['intent']))}</td><td>{html.escape(str(item.get('anchor_label') or ''))}</td>"
        f"<td>{html.escape(str(item['waypoint_id']))}</td><td>{'yes' if item['reachable'] else 'no'}</td><td>{item.get('geodesic_distance_m')}</td>"
        "</tr>"
        for item in trace.get("segments", [])
    )
    html_doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>RSC-Nav Map-Then-Task Demo</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202124;line-height:1.55;background:#fbfcfe}}
.hero{{max-width:1160px;margin:auto}}.note{{background:#eef5ff;border:1px solid #cfe0f6;padding:12px 14px;border-radius:8px}}
img,video{{max-width:100%;border:1px solid #d8e0ea;background:white}}table{{border-collapse:collapse;width:100%;margin-top:8px}}
th,td{{border:1px solid #d8e0ea;padding:8px;text-align:left}}th{{background:#eef3f8}}pre{{background:#f6f8fa;padding:12px;overflow:auto;border:1px solid #e2e8f0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main class="hero">
<h1>RSC-Nav MVP Demo: 先建语义地图，再执行自然语言任务</h1>
<p class="note">MVP 叙事：机器人初入环境时先用 coverage-loop 遍历策略构建 semantic BEV / object memory。后续兴趣驱动探索策略会替代这个固定遍历；当前先用它验证“语义地图可复用后再接任务规划和执行”。</p>
<h2>Linked GIF: Mapping -> Planning -> Execution</h2>
<img src="{html.escape(storyboard['storyboard_gif'])}" alt="semantic map then task linked gif">
<div class="grid">
<section><h2>First-person execution video</h2><video controls autoplay muted loop src="{html.escape(video)}"></video></section>
<section><h2>Natural language command</h2><p><b>{html.escape(goal)}</b></p>
<p>Planner 只能使用当前 semantic map 里的候选 landmark/waypoint。因为当前 MVP map 只有有限语义标签，涉及水杯、餐桌、房间功能等细粒度概念时，API 会把自然语言目标映射到最接近的可执行语义锚点，并保留不确定性。</p></section>
</div>
<h2>API / Planner Output</h2><pre>{html.escape(json.dumps(plan, ensure_ascii=False, indent=2))}</pre>
<h2>Habitat/Navmesh Execution</h2><table><tr><th>#</th><th>Intent</th><th>Anchor</th><th>Waypoint</th><th>Reachable</th><th>Geodesic m</th></tr>{rows}</table>
<h2>Artifacts</h2>
<ul>
<li><a href="{html.escape(storyboard['storyboard_gif'])}">semantic_map_then_task_linked.gif</a></li>
<li><a href="{html.escape(video)}">{html.escape(video)}</a></li>
<li><a href="demo_execution_trace.json">demo_execution_trace.json</a></li>
<li><a href="demo_planner_request.json">demo_planner_request.json</a></li>
<li><a href="demo_planner_output.json">demo_planner_output.json</a></li>
</ul>
<p>Storyboard metadata: map frames={storyboard['map_frame_count']}, task frames={storyboard['task_frame_count']}, fps={storyboard['fps']}.</p>
</main></body></html>"""
    report_path.write_text(html_doc, encoding="utf-8")
    (out_dir / "storyboard_metadata.json").write_text(json.dumps(storyboard, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_ui_font(size: int) -> ImageFont.ImageFont:
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


if __name__ == "__main__":
    main()
