from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retime a curated online demo and render its confirmed targets."
    )
    parser.add_argument("--input-gif", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-gif", required=True)
    parser.add_argument("--output-showcase", required=True)
    parser.add_argument("--output-metadata")
    parser.add_argument("--playback-speed", type=float, default=25.0)
    parser.add_argument("--source-step-stride", type=int, default=3)
    parser.add_argument("--palette-colors", type=int, default=192)
    parser.add_argument(
        "--target-step",
        action="append",
        default=[],
        metavar="TRACK_ID:STEP",
        help="Override the showcase evidence step for a confirmed target.",
    )
    args = parser.parse_args()

    input_gif = Path(args.input_gif).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()
    output_gif = Path(args.output_gif).expanduser().resolve()
    output_showcase = Path(args.output_showcase).expanduser().resolve()
    output_metadata = (
        Path(args.output_metadata).expanduser().resolve()
        if args.output_metadata
        else output_gif.with_suffix(".json")
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target_step_overrides = _target_step_overrides(args.target_step)
    with Image.open(input_gif) as source:
        frame_count = int(source.n_frames)
        source_size = source.size
        source_duration_ms = sum(
            _frame_duration_ms(source, frame_index) for frame_index in range(frame_count)
        )

    step_stride = max(1, int(args.source_step_stride))
    playback_speed = max(0.1, float(args.playback_speed))
    desired_frame_duration_ms = 1000.0 * step_stride / playback_speed
    desired_duration_ms = desired_frame_duration_ms * frame_count
    pts_scale = desired_duration_ms / max(1.0, float(source_duration_ms))

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    output_showcase.parent.mkdir(parents=True, exist_ok=True)
    palette_colors = min(256, max(32, int(args.palette_colors)))
    with tempfile.TemporaryDirectory(prefix="rscnav-public-demo-") as temp_dir:
        speed_overlay = Path(temp_dir) / "speed_overlay.png"
        _render_speed_overlay(
            size=source_size,
            playback_speed=playback_speed,
            output_path=speed_overlay,
        )
        filter_graph = (
            f"[0:v]setpts=PTS*{pts_scale:.10f}[base];"
            "[base][1:v]overlay=0:0:eof_action=repeat[stamped];"
            "[stamped]split[s0][s1];"
            f"[s0]palettegen=max_colors={palette_colors}[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(input_gif),
                "-i",
                str(speed_overlay),
                "-filter_complex",
                filter_graph,
                "-fps_mode",
                "vfr",
                "-loop",
                "0",
                str(output_gif),
            ],
            check=True,
        )
    with Image.open(output_gif) as rendered_gif:
        encoded_duration_ms = sum(
            _frame_duration_ms(rendered_gif, frame_index)
            for frame_index in range(int(rendered_gif.n_frames))
        )

    selections = _render_target_showcase(
        input_gif=input_gif,
        summary=summary,
        step_stride=step_stride,
        target_step_overrides=target_step_overrides,
        output_path=output_showcase,
    )
    output_metadata.write_text(
        json.dumps(
            {
                "input_gif": input_gif.name,
                "output_gif": output_gif.name,
                "output_showcase": output_showcase.name,
                "playback_speed": playback_speed,
                "source_step_stride": step_stride,
                "frame_count": frame_count,
                "source_duration_s": source_duration_ms / 1000.0,
                "planned_output_duration_s": desired_duration_ms / 1000.0,
                "encoded_output_duration_s": encoded_duration_ms / 1000.0,
                "target_selections": selections,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _frame_duration_ms(image: Image.Image, frame_index: int) -> int:
    image.seek(frame_index)
    return max(10, int(image.info.get("duration", 100)))


def _render_speed_overlay(
    size: tuple[int, int],
    playback_speed: float,
    output_path: Path,
) -> None:
    width, height = size
    scale = width / 960.0
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box = (
        round(528 * scale),
        round(482 * scale),
        round(744 * scale),
        round(510 * scale),
    )
    draw.rectangle(box, fill="#0d1217")
    speed_text = (
        f"{int(playback_speed)}x"
        if playback_speed.is_integer()
        else f"{playback_speed:g}x"
    )
    draw.text(
        (round(540 * scale), round(486 * scale)),
        f"Playback {speed_text} normal",
        fill="#e5b74e",
        font=_font(max(10, round(15 * scale))),
    )
    overlay.save(output_path)


def _render_target_showcase(
    input_gif: Path,
    summary: dict[str, Any],
    step_stride: int,
    target_step_overrides: dict[int, int],
    output_path: Path,
) -> list[dict[str, Any]]:
    confirmed = list(summary.get("confirmed_cups", []))
    task_step = int(summary.get("task_injection_step") or 0)
    canvas = Image.new("RGB", (1280, 720), "#0d1217")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(34, bold=True)
    subtitle_font = _font(20)
    track_font = _font(23, bold=True)
    body_font = _font(18)
    tiny_font = _font(15)

    draw.text((34, 24), "任务目标确认结果", fill="#eef3f5", font=title_font)
    draw.text(
        (34, 70),
        f"任务：{summary.get('task', '寻找任务目标')}  |  在线再次确认 {len(confirmed)} 个 cup track",
        fill="#9fb0bd",
        font=subtitle_font,
    )

    card_positions = ((28, 112), (650, 112), (28, 404), (650, 404))
    selections: list[dict[str, Any]] = []
    with Image.open(input_gif) as source:
        for item, (card_x, card_y) in zip(confirmed[:4], card_positions):
            visible_steps = [
                int(step)
                for step in item.get("visible_steps", [])
                if int(step) >= task_step
            ]
            if not visible_steps:
                visible_steps = [int(step) for step in item.get("visible_steps", [])]
            track_id = int(item.get("track_id", -1))
            evidence_step = target_step_overrides.get(
                track_id,
                _best_sampled_step(visible_steps, step_stride),
            )
            frame_index = min(
                int(source.n_frames) - 1,
                max(0, round(evidence_step / step_stride)),
            )
            source.seek(frame_index)
            frame = source.convert("RGB")
            rgb_panel = frame.crop(
                (0, 0, frame.width // 2, int(round(frame.height * 0.89)))
            )
            rgb_panel = ImageOps.fit(
                rgb_panel,
                (252, 252),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )

            draw.rounded_rectangle(
                (card_x, card_y, card_x + 602, card_y + 270),
                radius=6,
                fill="#151d24",
                outline="#2a3741",
                width=2,
            )
            canvas.paste(rgb_panel, (card_x + 9, card_y + 9))
            text_x = card_x + 280
            confidence = float(item.get("confidence", 0.0))
            views = int(item.get("views", 0))
            position = [float(value) for value in item.get("position_3d", [])]
            draw.text(
                (text_x, card_y + 22),
                f"Cup track {track_id}",
                fill="#65d2ad",
                font=track_font,
            )
            draw.text(
                (text_x, card_y + 70),
                f"confidence  {confidence:.3f}",
                fill="#eef3f5",
                font=body_font,
            )
            draw.text(
                (text_x, card_y + 108),
                f"independent views  {views}",
                fill="#eef3f5",
                font=body_font,
            )
            draw.text(
                (text_x, card_y + 146),
                "world XYZ",
                fill="#9fb0bd",
                font=body_font,
            )
            draw.text(
                (text_x, card_y + 178),
                ", ".join(f"{value:.2f}" for value in position[:3]),
                fill="#e5b74e",
                font=body_font,
            )
            draw.text(
                (text_x, card_y + 224),
                f"online evidence step {evidence_step}",
                fill="#9fb0bd",
                font=tiny_font,
            )
            selections.append(
                {
                    "track_id": track_id,
                    "evidence_step": evidence_step,
                    "source_frame_index": frame_index,
                    "confidence": confidence,
                    "views": views,
                    "position_3d": position,
                }
            )

    draw.text(
        (34, 690),
        "展示帧来自任务执行阶段的在线 RGB + Grounding；Habitat oracle 不参与目标确认。",
        fill="#7f929f",
        font=tiny_font,
    )
    canvas.save(output_path)
    return selections


def _best_sampled_step(visible_steps: list[int], step_stride: int) -> int:
    if not visible_steps:
        return 0
    return min(
        visible_steps,
        key=lambda step: (
            abs(step - round(step / step_stride) * step_stride),
            -step,
        ),
    )


def _target_step_overrides(values: list[str]) -> dict[int, int]:
    overrides: dict[int, int] = {}
    for value in values:
        track_text, separator, step_text = value.partition(":")
        if not separator:
            raise ValueError(f"Invalid --target-step value: {value!r}")
        overrides[int(track_text)] = int(step_text)
    return overrides


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc")
        if bold
        else ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc")
    )
    candidates = (
        *names,
        "/workspace/yujiexiao/.rscnav/fonts/STHeiti-Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
