from __future__ import annotations

import argparse
import html
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


CANVAS_SIZE = (1280, 720)
RGB_BOX = (0, 0, 640, 640)
DEPTH_BOX = (640, 0, 960, 320)
BEV_BOX = (960, 0, 1280, 320)
STATUS_BOX = (640, 320, 1280, 640)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a visual audit for an online interest-exploration run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--gif-duration-ms", type=int, default=120)
    parser.add_argument("--gif-frame-stride", type=int, default=2)
    parser.add_argument("--gif-width", type=int, default=960)
    parser.add_argument("--normal-action-duration-ms", type=int, default=1000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "report"
    rendered_dir = out_dir / "rendered_frames"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_json(run_dir / "frames_metadata.json")
    summary = _read_json(run_dir / "online_summary.json")
    coverage_path = run_dir / "posthoc_coverage_metrics.json"
    if coverage_path.exists():
        summary["posthoc_coverage"] = _read_json(coverage_path)
    trace = [
        json.loads(line)
        for line in (run_dir / "online_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frames = metadata.get("frames", [])
    if len(frames) != len(trace):
        raise ValueError(f"Frame/trace length mismatch: {len(frames)} != {len(trace)}")

    bev_crop_box = _bev_crop_box(run_dir)
    base_frame_duration_ms = max(40, int(args.gif_duration_ms))
    normal_action_duration_ms = max(1, int(args.normal_action_duration_ms))
    playback_speed = normal_action_duration_ms / float(base_frame_duration_ms)
    rendered_paths = []
    for frame, record in zip(frames, trace):
        rendered = _compose_frame(
            run_dir,
            frame,
            record,
            bev_crop_box=bev_crop_box,
            playback_speed=playback_speed,
            normal_action_duration_ms=normal_action_duration_ms,
        )
        path = rendered_dir / f"frame_{int(record['step']):04d}.jpg"
        rendered.save(path, quality=92)
        rendered_paths.append(path)

    gif_path = out_dir / "online_interest_exploration.gif"
    gif_stride = max(1, int(args.gif_frame_stride))
    gif_width = max(480, int(args.gif_width))
    gif_frames = []
    for path in rendered_paths[::gif_stride]:
        image = Image.open(path).convert("RGB")
        gif_height = int(round(image.height * gif_width / image.width))
        gif_frames.append(
            image.resize((gif_width, gif_height), Image.Resampling.LANCZOS)
        )
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=base_frame_duration_ms * gif_stride,
        loop=0,
        optimize=True,
    )

    mp4_path = out_dir / "online_interest_exploration.mp4"
    _render_mp4(rendered_dir, mp4_path, fps=max(2, round(1000 / base_frame_duration_ms)))
    html_path = out_dir / "online_interest_exploration.html"
    html_path.write_text(
        _html_report(
            metadata,
            summary,
            trace,
            gif_path.name,
            mp4_path.name if mp4_path.exists() else None,
            playback_speed=playback_speed,
            normal_action_duration_ms=normal_action_duration_ms,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "html": str(html_path),
                "gif": str(gif_path),
                "mp4": str(mp4_path) if mp4_path.exists() else None,
                "num_frames": len(rendered_paths),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _compose_frame(
    run_dir: Path,
    frame: dict[str, Any],
    record: dict[str, Any],
    bev_crop_box: tuple[int, int, int, int] | None,
    playback_speed: float,
    normal_action_duration_ms: int,
) -> Image.Image:
    step = int(record["step"])
    canvas = Image.new("RGB", CANVAS_SIZE, "#11161b")
    overlay = run_dir / "overlays" / f"frame_{step:04d}_overlay.jpg"
    rgb_path = overlay if overlay.exists() else Path(frame["rgb_path"])
    depth_path = Path(frame["depth_png"])
    bev_path = Path(frame["bev_png"])
    _paste_fit(canvas, Image.open(rgb_path).convert("RGB"), RGB_BOX)
    _paste_fit(canvas, Image.open(depth_path).convert("RGB"), DEPTH_BOX)
    bev = Image.open(bev_path).convert("RGB")
    if bev_crop_box is not None:
        bev = bev.crop(bev_crop_box)
    _paste_fit(canvas, bev, BEV_BOX)

    draw = ImageDraw.Draw(canvas)
    font = _font(25)
    small = _font(20)
    tiny = _font(17)
    _label(draw, RGB_BOX, "ONLINE RGB + GROUNDING", font)
    _label(draw, DEPTH_BOX, "CURRENT DEPTH", small)
    _label(draw, BEV_BOX, "ONLINE BEV + MEMORY", small)

    interest = record.get("interest", {})
    timing = record.get("timing_ms", {})
    phase = str(interest.get("exploration_phase", "unknown"))
    phase_short = {
        "deep_familiarization": "deep_fam",
        "task_planning": "task_plan",
        "task_execution": "task_exec",
        "cup_search": "task_exec",
    }.get(phase, phase)
    guidance_status = str(interest.get("guided_correction_status", "disabled"))
    phase_display = (
        f"{phase_short} | GUIDE"
        if guidance_status in {"navigating", "scanning"}
        else phase_short
    )
    planner = str(interest.get("execution_planner", "scan"))
    planner_short = {
        "hybrid_navmesh": "hybrid",
        "observed_bev": "observed-BEV",
    }.get(planner, planner)
    geodesic = interest.get("navmesh_geodesic_distance_m")
    route_status = (
        f"{planner_short} | {float(geodesic):.1f} m"
        if geodesic is not None
        else planner_short
    )
    task_text = record.get("task")
    planner_mode = interest.get("task_planner_mode")
    planner_model = interest.get("task_planner_model")
    evidence = record.get("evidence_events", {})
    status_lines = [
        f"Step {step:03d}  |  {record.get('action')}  |  {phase_display}",
        f"Policy: {interest.get('mode')} | {route_status}",
        (
            f"Task: issued at step {record.get('task_injection_step')} | find all cups"
            if task_text
            else "Task: not issued (building reusable memory)"
        ),
        (
            f"Planner: {planner_model} ({planner_mode})"
            if planner_mode
            else "Planner: waiting for MemoryReady"
        ),
        f"Active candidate: {interest.get('task_active_candidate_id') or 'none'}",
        (
            f"Evidence +/NA/miss: "
            f"{len(evidence.get('positive_observation_ids', []))}/"
            f"{len(evidence.get('not_observable_ids', []))}/"
            f"{len(evidence.get('expected_visible_miss_ids', []))}"
        ),
        f"Focused-confirmed cups: {len(record.get('confirmed_cup_track_ids', []))}",
        f"Explored cells: {record.get('bev', {}).get('num_explored_cells', 0):,}",
    ]
    x0, y0, x1, y1 = STATUS_BOX
    draw.rectangle(STATUS_BOX, fill="#172028")
    y = y0 + 30
    for index, line in enumerate(status_lines):
        draw.text((x0 + 28, y), line, fill="#f2f5f7" if index == 0 else "#c6d2db", font=small)
        y += 34
    footer = (
        "OBSERVE -> GROUND -> MEMORY UPDATE -> PLAN / NAVIGATE"
        if task_text
        else "OBSERVE -> GROUND -> UPDATE -> INTEREST -> ACTION"
    )
    draw.rectangle((0, 640, 1280, 720), fill="#0a0e12")
    draw.text((34, 668), footer, fill="#67d5ae", font=small)
    draw.text(
        (720, 654),
        f"Playback {playback_speed:.1f}x normal",
        fill="#f0c768",
        font=tiny,
    )
    draw.text(
        (720, 681),
        f"Normal reference: 1 action = {normal_action_duration_ms / 1000.0:.1f} s",
        fill="#93a4b2",
        font=tiny,
    )
    return canvas


def _html_report(
    metadata: dict[str, Any],
    summary: dict[str, Any],
    trace: list[dict[str, Any]],
    gif_name: str,
    mp4_name: str | None,
    playback_speed: float,
    normal_action_duration_ms: int,
) -> str:
    action_counts = Counter(str(row.get("action")) for row in trace)
    mode_counts = Counter(str(row.get("interest", {}).get("mode")) for row in trace)
    loop_seconds = sum(float(row.get("timing_ms", {}).get("total", 0.0)) for row in trace) / 1000.0
    load_seconds = float(summary.get("worker", {}).get("load_ms", 0.0)) / 1000.0
    mean_loop_ms = float(summary.get("timing_ms", {}).get("total", {}).get("mean", 0.0))
    coverage = float(
        summary.get("posthoc_coverage", {}).get("navmesh_observation_coverage", 0.0)
    )
    confirmed = summary.get("confirmed_cups", [])
    confirmed_rows = "".join(
        "<tr>"
        f"<td>{int(item.get('track_id', -1))}</td>"
        f"<td>{html.escape(str(item.get('label', 'cup')))}</td>"
        f"<td>{float(item.get('confidence', 0.0)):.3f}</td>"
        f"<td>{int(item.get('views', 0))}</td>"
        f"<td>{', '.join(f'{float(v):.2f}' for v in item.get('position_3d', []))}</td>"
        "</tr>"
        for item in confirmed
    ) or '<tr><td colspan="5">本次运行没有达到检测器再观测门槛的 cup-labeled track。</td></tr>'
    mp4 = (
        f'<video controls loop muted playsinline src="{html.escape(mp4_name)}"></video>'
        if mp4_name
        else ""
    )
    contract = metadata.get("online_contract", {})
    familiarization_step = summary.get("familiarization_complete_step")
    guided = summary.get("guided_correction", {})
    task_injection_step = summary.get("task_injection_step")
    task_planner = summary.get("task_planner") or {}
    task_planner_output = summary.get("task_planner_output") or {}
    evidence_totals = summary.get("evidence_event_totals") or {}
    checkpoint_labels = [
        ("autonomous_before_guidance", "纠偏前自主记忆"),
        ("after_guidance", "一次纠偏后"),
        ("task_start", "任务注入时"),
        ("final", "任务执行结束"),
    ]
    checkpoint_cards = "".join(
        (
            '<div class="checkpoint">'
            f"<h3>{html.escape(label)}</h3>"
            f'<img src="../checkpoints/{html.escape(name)}/bev_memory.png" '
            f'alt="{html.escape(label)}">'
            f"<p><code>{html.escape(name)}</code></p>"
            "</div>"
        )
        for name, label in checkpoint_labels
        if name in (summary.get("checkpoints") or {})
    )
    geometry_description = (
        "LingBot-Map causal predicted depth + Habitat exact pose"
        if summary.get("geometry_source") == "lingbot_depth_exact_pose"
        else "Habitat depth + exact pose"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RSC-Nav 实时环境熟悉与寻杯任务</title>
<style>
:root{{--bg:#0d1217;--panel:#151d24;--line:#2a3741;--text:#eef3f5;--muted:#9aabb7;--accent:#65d2ad;--gold:#e5b74e}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 ui-sans-serif,system-ui,sans-serif}}
header,main{{width:min(1240px,calc(100% - 32px));margin:auto}} header{{padding:34px 0 20px;border-bottom:1px solid var(--line)}}
h1{{margin:0 0 8px;font-size:32px;letter-spacing:0}} h2{{font-size:19px;margin:0 0 14px}} p{{color:var(--muted);margin:6px 0}}
main{{padding:24px 0 50px}} .metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:20px}}
.metric,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:6px}} .metric{{padding:14px}} .metric b{{display:block;font-size:23px;color:var(--accent)}} .metric span{{color:var(--muted)}}
.panel{{padding:18px;margin:14px 0}} video,img{{display:block;width:100%;background:#070a0c;border:1px solid var(--line)}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
code{{color:var(--gold)}} table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:9px;border-bottom:1px solid var(--line)}} th{{color:var(--muted)}}
.ok{{color:var(--accent)}} @media(max-width:850px){{.metrics{{grid-template-columns:1fr 1fr}}.grid{{grid-template-columns:1fr}}}}
.checkpoints{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .checkpoint h3{{font-size:15px;margin:0 0 8px}} .checkpoint img{{aspect-ratio:1;object-fit:contain}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b1014;padding:14px;border:1px solid var(--line);color:#cfe0e9}}
@media(max-width:850px){{.checkpoints{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<header>
		  <h1>Real-time environment familiarization → Find and report all cups</h1>
	  <p>运行时任务：{html.escape(str(summary.get("task", "")))}</p>
	  <p>这不是先录制再批处理：每一步均按 observe → GroundingDINO → 3D/BEV → object memory → interest policy → Habitat action 顺序执行。</p>
		  <p>任务在 step <code>{html.escape(str(task_injection_step))}</code> 才注入；此前只做与任务无关的环境熟悉。Qwen3-Max 只排序语义候选，底层路径仍由传统 BEV/navmesh 执行。</p>
		  <p>本运行若出现 <code>GUIDE</code>，表示一次经用户授权、公开记录的人工航点纠偏；机器人仍通过离散动作抵达，没有瞬移。</p>
	  <p>candidate cups 是熟悉阶段形成的稳定候选；confirmed cups 还要求在搜索阶段再次观测到。两者均不等于 Habitat 实例真值，本页不宣称实例级完整召回。</p>
</header>
<main>
  <section class="metrics">
    <div class="metric"><b>{int(summary.get("num_steps", 0))}</b><span>在线动作步</span></div>
    <div class="metric"><b>{mean_loop_ms:.0f} ms</b><span>平均闭环延迟</span></div>
    <div class="metric"><b>{loop_seconds:.1f} s</b><span>闭环累计时间</span></div>
    <div class="metric"><b>{load_seconds:.1f} s</b><span>模型冷启动</span></div>
    <div class="metric"><b>{coverage * 100:.1f}%</b><span>跑后 navmesh 观察覆盖</span></div>
    <div class="metric"><b>{int(summary.get("num_detected_collisions", 0))}</b><span>检测到的碰撞</span></div>
    <div class="metric"><b>{int(summary.get("num_scanned_surface_regions", 0))}</b><span>独立台面巡视区域</span></div>
    <div class="metric"><b>{int(summary.get("num_candidate_cups", 0))}</b><span>稳定 cup 候选</span></div>
	    <div class="metric"><b>{int(summary.get("num_confirmed_cups", 0))}</b><span>检测器再观测 cup track</span></div>
	    <div class="metric"><b>{html.escape(str(familiarization_step if familiarization_step is not None else "N/A"))}</b><span>熟悉完成 step</span></div>
		    <div class="metric"><b>{int(summary.get("cup_search_steps", 0))}</b><span>水杯搜索步</span></div>
		    <div class="metric"><b>{int(guided.get("explored_cell_gain") or 0):,}</b><span>单次纠偏新增探索格</span></div>
	    <div class="metric"><b>{html.escape(str(task_planner.get("mode_used", "N/A")))}</b><span>任务 planner 模式</span></div>
	    <div class="metric"><b>{float(task_planner.get("latency_ms", 0.0)):.0f} ms</b><span>API 规划耗时</span></div>
	    <div class="metric"><b>{int(evidence_totals.get("expected_visible_miss", 0))}</b><span>应见未见事件</span></div>
	    <div class="metric"><b>{int(summary.get("task_execution_steps", 0))}</b><span>任务执行步</span></div>
	  </section>
  <section class="panel">
    <h2>同步执行回放</h2>
	    {mp4}
	    <p>回放速度：<code>{playback_speed:.1f}× normal</code>；
	    “正常速度”按每个离散动作 {normal_action_duration_ms / 1000.0:.1f} 秒定义。
	    这只是回放倍率，真实模型耗时见上方闭环延迟。</p>
	    <p>GIF 版本：</p>
    <img src="{html.escape(gif_name)}" alt="Online interest exploration GIF">
	  </section>
	  <section class="panel">
	    <h2>四阶段可复用记忆检查点</h2>
	    <div class="checkpoints">{checkpoint_cards}</div>
	    <p>四张图分别来自同一 episode 的在线状态，不是用最终地图倒推生成。</p>
	  </section>
	  <section class="grid">
    <div class="panel">
      <h2>因果与泄漏审计</h2>
      <p class="ok">✓ 当前/历史帧才可参与当前决策：{str(bool(summary.get("causal_invariants", {}).get("all_decisions_use_current_or_past_frames"))).lower()}</p>
      <p class="ok">✓ semantic sensor：{html.escape(str(contract.get("semantic_sensor_enabled")))}</p>
      <p class="ok">✓ semantic scene read：{html.escape(str(contract.get("semantic_scene_read")))}</p>
	      <p class="ok">✓ 预制路线：{html.escape(str(contract.get("precomputed_route")))}</p>
	      <p class="ok">✓ 任务在 MemoryReady 后才可见：{str(bool(summary.get("causal_invariants", {}).get("task_hidden_until_memory_ready"))).lower()}</p>
      <p>导航目标来源：<code>{html.escape(str(contract.get("navigation_target_source")))}</code></p>
      <p>navmesh 用途：<code>{html.escape(str(contract.get("navmesh_online_usage")))}</code></p>
	      <p>几何输入：<code>{html.escape(geometry_description)}</code>。</p>
    </div>
    <div class="panel">
      <h2>策略分布</h2>
      <p>动作：<code>{html.escape(json.dumps(dict(action_counts), ensure_ascii=False))}</code></p>
      <p>兴趣模式：<code>{html.escape(json.dumps(dict(mode_counts), ensure_ascii=False))}</code></p>
	      <p>停止原因：<code>{html.escape(str(summary.get("stop_reason")))}</code></p>
	      <p>人工纠偏：<code>{html.escape(json.dumps(guided, ensure_ascii=False))}</code></p>
      <p>已探索栅格：<code>{int(summary.get("bev", {}).get("num_explored_cells", 0)):,}</code></p>
      <p>覆盖确认：<code>{int(summary.get("coverage_confirmations", 0))}</code>；最终阶段：<code>{html.escape(str(summary.get("final_exploration_phase")))}</code></p>
    </div>
	  </section>
	  <section class="grid">
	    <div class="panel">
	      <h2>运行时 API 任务规划</h2>
	      <p>模型：<code>{html.escape(str(task_planner.get("model", "N/A")))}</code>；
	      模式：<code>{html.escape(str(task_planner.get("mode_used", "N/A")))}</code>；
	      注入 step：<code>{html.escape(str(task_injection_step))}</code>。</p>
	      <pre>{html.escape(json.dumps(task_planner_output, ensure_ascii=False, indent=2))}</pre>
	    </div>
	    <div class="panel">
	      <h2>三态证据累计</h2>
	      <p>positive observation：
	      <code>{int(evidence_totals.get("positive_observation", 0))}</code></p>
	      <p>not observable：
	      <code>{int(evidence_totals.get("not_observable", 0))}</code></p>
	      <p>expected-visible miss：
	      <code>{int(evidence_totals.get("expected_visible_miss", 0))}</code></p>
	      <p>Planner 只决定候选顺序；每次到达后的新观测仍会更新 confidence、freshness 和 negative evidence。</p>
	    </div>
	  </section>
  <section class="panel">
    <h2>任务阶段再次检测到的 cup-labeled tracks</h2>
    <table><thead><tr><th>Track</th><th>类别</th><th>置信度</th><th>独立视角</th><th>世界坐标 XYZ</th></tr></thead>
    <tbody>{confirmed_rows}</tbody></table>
  </section>
</main>
</body>
</html>"""


def _render_mp4(rendered_dir: Path, output: Path, fps: int) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(max(1, int(fps))),
                "-i",
                str(rendered_dir / "frame_%04d.jpg"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(output),
            ],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return


def _bev_crop_box(run_dir: Path) -> tuple[int, int, int, int] | None:
    state_path = run_dir / "online_bev_state.npz"
    if not state_path.exists():
        return None
    state = np.load(state_path)
    explored = np.asarray(state["explored"], dtype=bool)
    cells = np.argwhere(explored)
    if cells.size == 0:
        return None
    height = explored.shape[1]
    pad = 18
    left = max(0, int(cells[:, 0].min()) - pad)
    right = min(explored.shape[0], int(cells[:, 0].max()) + pad + 1)
    top = max(0, height - 1 - int(cells[:, 1].max()) - pad)
    bottom = min(height, height - int(cells[:, 1].min()) + pad)
    side = max(right - left, bottom - top)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    left = max(0, center_x - side // 2)
    top = max(0, center_y - side // 2)
    right = min(explored.shape[0], left + side)
    bottom = min(height, top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    return left, top, right, bottom


def _paste_fit(canvas: Image.Image, source: Image.Image, box: tuple[int, int, int, int]) -> None:
    width, height = box[2] - box[0], box[3] - box[1]
    fitted = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
    canvas.paste(fitted, (box[0], box[1]))


def _label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    x0, y0, _, _ = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.rectangle((x0 + 12, y0 + 12, x0 + width + 28, y0 + height + 25), fill="#10171d")
    draw.text((x0 + 20, y0 + 17), text, fill="#f4f7f8", font=font)


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
