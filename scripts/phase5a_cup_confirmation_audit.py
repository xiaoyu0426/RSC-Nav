from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


CANVAS_WIDTH = 1440
HEADER_HEIGHT = 128
ROW_HEIGHT = 260
TEXT_WIDTH = 480
THUMB_SIZE = (286, 175)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render task-stage cup confirmation crops and gate evidence."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-path")
    parser.add_argument("--max-crops-per-track", type=int, default=3)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    summary = json.loads(
        (run_dir / "online_summary.json").read_text(encoding="utf-8")
    )
    bundle = summary.get("cup_confirmation") or {}
    results = bundle.get("results") or {}
    observations = bundle.get("observations") or {}
    track_ids = sorted(
        set(results) | set(observations),
        key=lambda value: int(value),
    )
    out_path = (
        Path(args.out_path).expanduser().resolve()
        if args.out_path
        else run_dir / "report" / "cup_confirmation_audit.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new(
        "RGB",
        (
            CANVAS_WIDTH,
            HEADER_HEIGHT + max(1, len(track_ids)) * ROW_HEIGHT,
        ),
        "#0f1419",
    )
    draw = ImageDraw.Draw(image)
    title_font = _font(30)
    body_font = _font(19)
    small_font = _font(16)
    draw.text(
        (34, 24),
        "Task-stage cup confirmation audit",
        fill="#f4f7f9",
        font=title_font,
    )
    draw.text(
        (34, 70),
        (
            f"Candidates {len(track_ids)}  |  "
            f"Strictly verified {int(summary.get('num_confirmed_cups', 0))}  |  "
            f"Detector-reobserved {int(summary.get('num_detector_reobserved_cups', 0))}"
        ),
        fill="#9fb0bd",
        font=body_font,
    )

    if not track_ids:
        draw.text(
            (34, HEADER_HEIGHT + 44),
            "No task-stage cup confirmation attempt was recorded.",
            fill="#c6d1d9",
            font=body_font,
        )
    for row_index, track_id in enumerate(track_ids):
        top = HEADER_HEIGHT + row_index * ROW_HEIGHT
        _draw_track_row(
            image=image,
            draw=draw,
            run_dir=run_dir,
            track_id=track_id,
            result=results.get(track_id) or {},
            observations=observations.get(track_id) or [],
            max_crops=max(1, int(args.max_crops_per_track)),
            top=top,
            body_font=body_font,
            small_font=small_font,
        )
    image.save(out_path)
    print(
        json.dumps(
            {
                "audit_png": str(out_path),
                "num_tracks": len(track_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _draw_track_row(
    *,
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    run_dir: Path,
    track_id: str,
    result: dict[str, Any],
    observations: list[dict[str, Any]],
    max_crops: int,
    top: int,
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    status = str(result.get("status", "unknown"))
    color = _status_color(status)
    draw.rectangle(
        (18, top + 8, CANVAS_WIDTH - 18, top + ROW_HEIGHT - 8),
        fill="#172028",
        outline="#2c3a45",
        width=2,
    )
    draw.rectangle((18, top + 8, 26, top + ROW_HEIGHT - 8), fill=color)
    draw.text(
        (48, top + 28),
        f"Track {track_id}",
        fill="#f4f7f9",
        font=body_font,
    )
    draw.text(
        (48, top + 62),
        status,
        fill=color,
        font=body_font,
    )
    lines = [
        (
            f"attempts {int(result.get('attempts', 0))} | "
            f"views {int(result.get('task_independent_views', 0))} | "
            f"inliers {int(result.get('geometry_inlier_views', 0))}"
        ),
        (
            f"visual +/- {int(result.get('visual_passes', 0))}/"
            f"{int(result.get('visual_negatives', 0))} | "
            f"errors {int(result.get('verifier_errors', 0))}"
        ),
        (
            f"depth relief passes {int(result.get('depth_relief_passes', 0))} | "
            f"mean {float(result.get('mean_depth_surface_relief_m', 0.0)):.3f} m"
        ),
        (
            f"3D spread {float(result.get('position_spread_m', 0.0)):.3f} m | "
            f"raw {float(result.get('raw_position_spread_m', 0.0)):.3f} m"
        ),
        (
            f"crop score +{float(result.get('mean_crop_positive_score', 0.0)):.3f} "
            f"-{float(result.get('mean_crop_negative_score', 0.0)):.3f}"
        ),
    ]
    for index, line in enumerate(lines):
        draw.text(
            (48, top + 102 + index * 25),
            line,
            fill="#b9c6cf",
            font=small_font,
        )

    crop_observations = [
        item for item in observations if item.get("crop_path")
    ]
    selected = _spread_selection(crop_observations, max_crops)
    for index, observation in enumerate(selected):
        left = TEXT_WIDTH + 18 + index * (THUMB_SIZE[0] + 18)
        thumb = _annotated_crop(run_dir, observation)
        _paste_fit(image, thumb, (left, top + 28, *THUMB_SIZE))
        draw.text(
            (left, top + 211),
            (
                f"step {int(observation.get('step', -1))} | "
                f"{observation.get('crop_verifier_status', 'unknown')}"
            ),
            fill="#aebbc5",
            font=small_font,
        )
        draw.text(
            (left, top + 232),
            (
                "depth relief "
                f"{float(observation.get('depth_surface_relief_m') or 0.0):.3f} m"
            ),
            fill="#8698a5",
            font=small_font,
        )


def _annotated_crop(
    run_dir: Path,
    observation: dict[str, Any],
) -> Image.Image:
    crop_path = Path(str(observation["crop_path"]))
    if not crop_path.is_absolute():
        crop_path = run_dir / crop_path
    crop = Image.open(crop_path).convert("RGB")
    draw = ImageDraw.Draw(crop)
    target_box = observation.get("crop_target_box")
    if target_box and len(target_box) == 4:
        draw.rectangle(
            tuple(float(value) for value in target_box),
            outline="#ffd166",
            width=max(2, crop.width // 100),
        )
    for detection in observation.get("crop_associated_detections", []):
        box = detection.get("box")
        if not box or len(box) != 4:
            continue
        label = str(detection.get("label", ""))
        positive = label in {
            "real drinking cup",
            "cup",
            "mug",
            "drinking glass",
        }
        outline = "#55d6a9" if positive else "#ff6b6b"
        draw.rectangle(
            tuple(float(value) for value in box),
            outline=outline,
            width=max(2, crop.width // 120),
        )
    return crop


def _paste_fit(
    canvas: Image.Image,
    source: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    left, top, width, height = box
    fitted = ImageOps.contain(
        source,
        (width, height),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", (width, height), "#0b1014")
    panel.paste(
        fitted,
        ((width - fitted.width) // 2, (height - fitted.height) // 2),
    )
    canvas.paste(panel, (left, top))


def _spread_selection(
    items: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[-1]]
    indices = [
        round(index * (len(items) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [items[index] for index in indices]


def _status_color(status: str) -> str:
    if status == "verified":
        return "#55d6a9"
    if status.startswith("rejected"):
        return "#ff6b6b"
    if status.startswith("inconclusive") or status.startswith("conflicting"):
        return "#f0c768"
    return "#86b7e5"


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
