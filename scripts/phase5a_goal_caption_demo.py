from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_caption import build_agent_caption  # noqa: E402


CANVAS_SIZE = (1280, 720)
RGB_BOX = (0, 0, 720, 540)
DEPTH_BOX = (720, 0, 1000, 260)
BEV_BOX = (1000, 0, 1280, 260)
BELIEF_BOX = (720, 260, 1280, 540)
CAPTION_BOX = (0, 540, 1280, 720)

KEY_EVENT_WEIGHTS_MS = {
    "task_injected_and_planned": 1500,
    "cup_candidate_reobservation_started": 900,
    "cup_confirmation_completed": 1700,
    "cup_confirmation_deferred": 1500,
    "cup_confirmation_retry_scheduled": 1000,
    "support_surface_inspection_started": 900,
    "support_surface_inspection_completed": 1700,
    "candidate_unreachable": 1100,
    "candidate_navigation_failed": 1100,
    "candidate_navigation_failure": 1100,
    "online_memory_candidates_appended": 900,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a goal-centric Habitat demo with trace-grounded agent captions."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--target-duration-s", type=float, default=30.0)
    parser.add_argument("--gif-width", type=int, default=960)
    parser.add_argument("--palette-colors", type=int, default=192)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else run_dir / "goal_caption_demo"
    )
    rendered_dir = out_dir / "rendered_frames"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_json(run_dir / "frames_metadata.json")
    summary = _read_json(run_dir / "online_summary.json")
    trace = _read_jsonl(run_dir / "online_trace.jsonl")
    frames = list(metadata.get("frames", []))
    if not frames or len(frames) != len(trace):
        raise ValueError(
            f"Frame/trace length mismatch: {len(frames)} != {len(trace)}"
        )

    selection = _select_demo_steps(trace)
    durations = _allocate_durations(
        selection,
        trace,
        target_duration_ms=max(5000, round(args.target_duration_s * 1000)),
    )
    bev_crop_box = _bev_crop_box(run_dir)
    rendered_paths: list[Path] = []
    captions: list[dict[str, Any]] = []
    for output_index, step in enumerate(selection):
        frame = frames[step]
        record = trace[step]
        caption = record.get("agent_caption")
        if not isinstance(caption, dict):
            caption = build_agent_caption(
                task=record.get("task"),
                interest=record.get("interest"),
                task_plan_events=record.get("task_plan_events"),
                confirmed_cup_track_ids=record.get(
                    "confirmed_cup_track_ids",
                    [],
                ),
            )
        rendered = _compose_frame(
            run_dir,
            frame,
            record,
            caption=caption,
            bev_crop_box=bev_crop_box,
        )
        path = rendered_dir / f"frame_{output_index:04d}.jpg"
        rendered.save(path, quality=91, subsampling=1)
        rendered_paths.append(path)
        captions.append(
            {
                "output_index": output_index,
                "step": int(record["step"]),
                "duration_ms": durations[output_index],
                **caption,
            }
        )

    closing_path = rendered_dir / f"frame_{len(rendered_paths):04d}.jpg"
    closing = _compose_closing_frame(
        run_dir,
        frames,
        trace,
        summary,
    )
    closing.save(closing_path, quality=92, subsampling=1)
    rendered_paths.append(closing_path)
    closing_duration_ms = 2400

    target_duration_ms = max(5000, round(args.target_duration_s * 1000))
    frame_budget_ms = target_duration_ms - closing_duration_ms
    durations = _rescale_durations(durations, frame_budget_ms)
    all_durations = durations + [closing_duration_ms]

    gif_path = out_dir / "rscnav_goal_caption_30s.gif"
    _write_gif(
        rendered_paths,
        all_durations,
        gif_path,
        width=max(640, int(args.gif_width)),
        palette_colors=min(256, max(32, int(args.palette_colors))),
    )
    mp4_path = out_dir / "rscnav_goal_caption_30s.mp4"
    _write_variable_mp4(rendered_paths, all_durations, mp4_path)
    poster_path = out_dir / "rscnav_goal_caption_result.png"
    closing.save(poster_path)
    metadata_path = out_dir / "rscnav_goal_caption_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_run": str(run_dir),
                "source_commit": summary.get("run_commit"),
                "task": summary.get("task"),
                "task_injection_step": summary.get("task_injection_step"),
                "source_steps": len(trace),
                "retained_trace_frames": len(selection),
                "output_frames": len(rendered_paths),
                "duration_s": sum(all_durations) / 1000.0,
                "timing_contract": (
                    "Navigation is curated at approximately 25x; key decisions "
                    "are held for readability."
                ),
                "caption_contract": {
                    "trace_grounded": True,
                    "stored_caption_preferred": True,
                    "reconstruction_uses_current_record_only": True,
                    "success_requires_verified_status": True,
                },
                "selected_steps": selection,
                "captions": captions,
                "closing_result": _result_summary(summary),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gif": str(gif_path),
                "mp4": str(mp4_path) if mp4_path.exists() else None,
                "poster": str(poster_path),
                "metadata": str(metadata_path),
                "duration_s": sum(all_durations) / 1000.0,
                "retained_trace_frames": len(selection),
            },
            indent=2,
        )
    )


def _select_demo_steps(trace: list[dict[str, Any]]) -> list[int]:
    task_step = next(
        (
            int(record["step"])
            for record in trace
            if record.get("task") is not None
        ),
        len(trace) // 2,
    )
    selected = set(range(0, task_step, 12))
    selected.update(range(task_step, len(trace), 6))
    selected.update({0, max(0, task_step - 1), task_step, len(trace) - 1})

    previous_candidate: str | None = None
    previous_phase: str | None = None
    for index, record in enumerate(trace):
        events = record.get("task_plan_events") or []
        if any(
            str(event.get("event", "")) in KEY_EVENT_WEIGHTS_MS
            for event in events
        ):
            selected.update(
                range(max(0, index - 4), min(len(trace), index + 5))
            )
        interest = record.get("interest") or {}
        candidate = str(interest.get("task_active_candidate_id") or "")
        phase = str(interest.get("exploration_phase") or "")
        if candidate and candidate != previous_candidate:
            selected.update(
                range(max(0, index - 2), min(len(trace), index + 3))
            )
        if phase and phase != previous_phase:
            selected.update(
                range(max(0, index - 2), min(len(trace), index + 3))
            )
        previous_candidate = candidate or previous_candidate
        previous_phase = phase or previous_phase
    return sorted(selected)


def _allocate_durations(
    selection: list[int],
    trace: list[dict[str, Any]],
    *,
    target_duration_ms: int,
) -> list[int]:
    weights: list[int] = []
    previous_candidate: str | None = None
    for step in selection:
        record = trace[step]
        weight = 90
        for event in record.get("task_plan_events") or []:
            weight = max(
                weight,
                KEY_EVENT_WEIGHTS_MS.get(str(event.get("event", "")), 0),
            )
        candidate = str(
            (record.get("interest") or {}).get("task_active_candidate_id")
            or ""
        )
        if candidate and candidate != previous_candidate:
            weight = max(weight, 420)
        if step == 0:
            weight = max(weight, 1500)
        weights.append(weight)
        previous_candidate = candidate or previous_candidate
    return _rescale_durations(weights, target_duration_ms)


def _rescale_durations(durations: list[int], target_ms: int) -> list[int]:
    if not durations:
        return []
    scale = target_ms / float(sum(durations))
    result = [max(40, int(round(value * scale / 10.0)) * 10) for value in durations]
    difference = target_ms - sum(result)
    index = len(result) - 1
    while difference != 0 and result:
        unit = 10 if difference > 0 else -10
        if unit < 0 and result[index] <= 40:
            index = (index - 1) % len(result)
            continue
        result[index] += unit
        difference -= unit
        index = (index - 1) % len(result)
    return result


def _compose_frame(
    run_dir: Path,
    frame: dict[str, Any],
    record: dict[str, Any],
    *,
    caption: dict[str, Any],
    bev_crop_box: tuple[int, int, int, int] | None,
) -> Image.Image:
    step = int(record["step"])
    canvas = Image.new("RGB", CANVAS_SIZE, "#0d1217")
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
    _panel_label(draw, RGB_BOX, "RGB + LIVE GROUNDING")
    _panel_label(draw, DEPTH_BOX, "DEPTH")
    _panel_label(draw, BEV_BOX, "SEMANTIC EVIDENCE")
    _draw_belief_panel(draw, record)
    _draw_caption_panel(draw, record, caption)
    return canvas


def _draw_belief_panel(
    draw: ImageDraw.ImageDraw,
    record: dict[str, Any],
) -> None:
    x0, y0, x1, y1 = BELIEF_BOX
    draw.rectangle(BELIEF_BOX, fill="#151d24")
    draw.text(
        (x0 + 22, y0 + 18),
        "SEARCH BELIEF",
        fill="#eef3f5",
        font=_font(21, bold=True),
    )
    interest = record.get("interest") or {}
    ranking = [
        item
        for item in interest.get("task_search_ranking", [])
        if isinstance(item, dict)
    ]
    active = str(interest.get("task_active_candidate_id") or "")
    if not ranking:
        draw.text(
            (x0 + 22, y0 + 65),
            (
                "Task received: initializing grounded ranking"
                if record.get("task") is not None
                else "Task hidden: building reusable memory"
            ),
            fill="#9fb0bd",
            font=_font(18),
        )
    for row_index, item in enumerate(ranking[:3]):
        candidate_id = str(item.get("candidate_id", "unknown"))
        label = str(item.get("label", item.get("kind", "candidate")))
        posterior = _candidate_probability(item)
        row_y = y0 + 58 + row_index * 54
        is_active = candidate_id == active
        draw.rectangle(
            (x0 + 18, row_y, x1 - 18, row_y + 44),
            fill="#203229" if is_active else "#1a252d",
            outline="#65d2ad" if is_active else "#2a3741",
            width=2 if is_active else 1,
        )
        draw.text(
            (x0 + 30, row_y + 10),
            f"{row_index + 1}. {candidate_id}  {label}",
            fill="#eef3f5",
            font=_font(17, bold=is_active),
        )
        probability_text = f"p {posterior:.2f}" if posterior is not None else "p --"
        draw.text(
            (x1 - 105, row_y + 10),
            probability_text,
            fill="#f0c768" if is_active else "#9fb0bd",
            font=_font(17, bold=is_active),
        )
    planner = str(interest.get("task_planner_model") or "waiting")
    mode = str(interest.get("mode") or "unknown")
    draw.text(
        (x0 + 22, y1 - 38),
        _fit_line(
            draw,
            f"Planner {planner} | policy {mode}",
            _font(15),
            x1 - x0 - 44,
        ),
        fill="#7f929f",
        font=_font(15),
    )


def _draw_caption_panel(
    draw: ImageDraw.ImageDraw,
    record: dict[str, Any],
    caption: dict[str, Any],
) -> None:
    x0, y0, x1, y1 = CAPTION_BOX
    draw.rectangle(CAPTION_BOX, fill="#090d11")
    stage = str(caption.get("stage", "PLAN"))
    stage_color = {
        "FAMILIARIZE": "#55b8e8",
        "PLAN": "#f0c768",
        "APPROACH": "#67d5ae",
        "SEARCH": "#67d5ae",
        "VERIFY": "#efb84f",
        "REPLAN": "#ef8b73",
        "CONFIRMED": "#67d5ae",
        "REPORT": "#67d5ae",
    }.get(stage, "#9fb0bd")
    draw.text(
        (x0 + 28, y0 + 13),
        'GOAL  Find and report all cups',
        fill="#eef3f5",
        font=_font(17, bold=True),
    )
    task_state = (
        f"task live from step {record.get('task_injection_step')}"
        if record.get("task") is not None
        else "task hidden from agent"
    )
    draw.text(
        (x1 - 340, y0 + 14),
        f"Step {int(record['step']):03d} | {task_state}",
        fill="#7f929f",
        font=_font(15),
    )
    draw.rectangle(
        (x0 + 28, y0 + 46, x0 + 165, y0 + 79),
        fill=stage_color,
    )
    stage_width = draw.textlength(stage, font=_font(16, bold=True))
    draw.text(
        (x0 + 96 - stage_width / 2, y0 + 53),
        stage,
        fill="#091016",
        font=_font(16, bold=True),
    )
    plan_font = _font(24, bold=True)
    draw.text(
        (x0 + 184, y0 + 49),
        _fit_line(
            draw,
            str(caption.get("plan", "")),
            plan_font,
            x1 - x0 - 215,
        ),
        fill="#f4f7f8",
        font=plan_font,
    )
    body_font = _font(17)
    why_lines = _wrap_lines(
        draw,
        "WHY  " + str(caption.get("why", "")),
        body_font,
        x1 - x0 - 56,
        max_lines=2,
    )
    for index, line in enumerate(why_lines):
        draw.text(
            (x0 + 28, y0 + 91 + index * 22),
            line,
            fill="#c7d2da",
            font=body_font,
        )
    next_text = _fit_line(
        draw,
        "NEXT  " + str(caption.get("next", "")),
        body_font,
        x1 - x0 - 56,
    )
    draw.text(
        (x0 + 28, y0 + 138),
        next_text,
        fill="#67d5ae",
        font=body_font,
    )
    evidence = _fit_line(
        draw,
        "EVIDENCE  " + str(caption.get("evidence", "")),
        _font(15),
        x1 - x0 - 400,
    )
    draw.text(
        (x0 + 28, y1 - 25),
        evidence,
        fill="#9fb0bd",
        font=_font(15),
    )
    draw.text(
        (x1 - 355, y1 - 25),
        "NAVIGATION ~25x | KEY DECISIONS HELD",
        fill="#f0c768",
        font=_font(14),
    )


def _compose_closing_frame(
    run_dir: Path,
    frames: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Image.Image:
    result = _result_summary(summary)
    evidence_step = result.get("positive_support_step")
    if evidence_step is None:
        evidence_step = len(trace) - 1
    evidence_step = max(0, min(len(trace) - 1, int(evidence_step)))
    frame = frames[evidence_step]
    overlay = run_dir / "overlays" / f"frame_{evidence_step:04d}_overlay.jpg"
    rgb_path = overlay if overlay.exists() else Path(frame["rgb_path"])

    canvas = Image.new("RGB", CANVAS_SIZE, "#0d1217")
    _paste_fit(canvas, Image.open(rgb_path).convert("RGB"), (0, 0, 660, 570))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 660, 58), fill="#10171d")
    draw.text(
        (22, 17),
        f"BEST POSITIVE SEARCH EVIDENCE | STEP {evidence_step}",
        fill="#eef3f5",
        font=_font(20, bold=True),
    )
    draw.rectangle((660, 0, 1280, 570), fill="#151d24")
    draw.text(
        (690, 32),
        "TASK RESULT",
        fill="#eef3f5",
        font=_font(24, bold=True),
    )
    verified = int(result["verified_cups"])
    draw.text(
        (690, 82),
        f"{verified} strictly verified cups",
        fill="#67d5ae" if verified else "#ef8b73",
        font=_font(34, bold=True),
    )
    result_lines = [
        (
            f"Support evidence: +{result['positive_supports']} / "
            f"-{result['negative_supports']}"
        ),
        f"Rejected cup hypotheses: {result['rejected_candidates']}",
        f"Inconclusive hypotheses: {result['inconclusive_candidates']}",
    ]
    y = 150
    for line in result_lines:
        draw.text((690, y), line, fill="#c7d2da", font=_font(20))
        y += 42
    draw.rectangle(
        (690, 304, 1248, 438),
        fill="#1a252d",
        outline="#2a3741",
        width=2,
    )
    conclusion = (
        "The planner found useful support evidence but did not claim task "
        "success. The remaining bottleneck is active viewpoint verification."
        if verified == 0
        else "Verified targets are reportable; search continues for additional cups."
    )
    for index, line in enumerate(
        _wrap_lines(
            draw,
            conclusion,
            _font(20),
            510,
            max_lines=4,
        )
    ):
        draw.text(
            (714, 328 + index * 29),
            line,
            fill="#eef3f5",
            font=_font(20),
        )
    draw.text(
        (690, 474),
        "Evidence image is a search observation, not a verified target crop.",
        fill="#f0c768",
        font=_font(16),
    )
    draw.rectangle((0, 570, 1280, 720), fill="#090d11")
    draw.text(
        (30, 592),
        "CONCLUSION",
        fill="#7f929f",
        font=_font(16, bold=True),
    )
    final_text = (
        "Goal not completed in this run: zero cup candidates passed the strict gate."
        if verified == 0
        else f"Goal partially completed: {verified} cups passed the strict gate."
    )
    draw.text(
        (30, 622),
        final_text,
        fill="#eef3f5",
        font=_font(27, bold=True),
    )
    draw.text(
        (30, 670),
        "Next iteration: choose informative confirmation viewpoints before spending budget on lower-priority supports.",
        fill="#67d5ae",
        font=_font(19),
    )
    return canvas


def _result_summary(summary: dict[str, Any]) -> dict[str, Any]:
    events = list(summary.get("task_plan_events") or [])
    support_events = [
        event
        for event in events
        if event.get("event") == "support_surface_inspection_completed"
    ]
    confirmation_events = [
        event
        for event in events
        if event.get("event")
        in {"cup_confirmation_completed", "cup_confirmation_deferred"}
    ]
    positive = [
        event
        for event in support_events
        if event.get("outcome") == "target_evidence_observed"
    ]
    negative = [
        event
        for event in support_events
        if event.get("outcome") == "no_target_evidence_observed"
        and event.get("observable_scan")
    ]
    rejected = [
        event
        for event in confirmation_events
        if str(event.get("status", "")).startswith("rejected_")
    ]
    inconclusive = [
        event
        for event in confirmation_events
        if str(event.get("status", "")).startswith("inconclusive")
    ]
    return {
        "verified_cups": int(summary.get("num_confirmed_cups", 0)),
        "positive_supports": len(positive),
        "negative_supports": len(negative),
        "rejected_candidates": len(rejected),
        "inconclusive_candidates": len(inconclusive),
        "positive_support_step": (
            int(positive[0]["step"]) if positive else None
        ),
    }


def _candidate_probability(item: dict[str, Any]) -> float | None:
    for key in ("posterior", "target_posterior", "target_likelihood", "prior"):
        if item.get(key) is not None:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                continue
    return None


def _write_gif(
    paths: list[Path],
    durations: list[int],
    output: Path,
    *,
    width: int,
    palette_colors: int,
) -> None:
    frames: list[Image.Image] = []
    for path in paths:
        source = Image.open(path).convert("RGB")
        height = round(source.height * width / source.width)
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
        frames.append(
            resized.quantize(
                colors=palette_colors,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.FLOYDSTEINBERG,
            )
        )
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )


def _write_variable_mp4(
    paths: list[Path],
    durations: list[int],
    output: Path,
) -> None:
    concat_path = output.with_suffix(".concat.txt")
    lines = []
    for path, duration in zip(paths, durations):
        safe_path = str(path).replace("'", "'\\''")
        lines.append(f"file '{safe_path}'")
        lines.append(f"duration {duration / 1000.0:.3f}")
    safe_last = str(paths[-1]).replace("'", "'\\''")
    lines.append(f"file '{safe_last}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-vf",
                "fps=25",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        output.unlink(missing_ok=True)
    finally:
        concat_path.unlink(missing_ok=True)


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


def _paste_fit(
    canvas: Image.Image,
    source: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    size = (box[2] - box[0], box[3] - box[1])
    fitted = ImageOps.fit(source, size, method=Image.Resampling.LANCZOS)
    canvas.paste(fitted, (box[0], box[1]))


def _panel_label(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
) -> None:
    font = _font(17, bold=True)
    width = draw.textlength(text, font=font)
    x0, y0, _, _ = box
    draw.rectangle(
        (x0 + 12, y0 + 12, x0 + width + 30, y0 + 43),
        fill="#10171d",
    )
    draw.text((x0 + 21, y0 + 19), text, fill="#f4f7f8", font=font)


def _fit_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: float,
) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    shortened = text
    while shortened and draw.textlength(shortened + suffix, font=font) > max_width:
        shortened = shortened[:-1]
    return shortened.rstrip() + suffix


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: float,
    *,
    max_lines: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    consumed = " ".join(lines)
    if len(consumed) < len(text) and lines:
        lines[-1] = _fit_line(
            draw,
            lines[-1] + "...",
            font,
            max_width,
        )
    return lines


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / names[0],
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
