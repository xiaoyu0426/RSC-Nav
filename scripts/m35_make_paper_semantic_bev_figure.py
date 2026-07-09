from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a paper-style semantic BEV summary figure for M3.5.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="RSC-Nav Semantic BEV Memory")
    parser.add_argument("--subtitle", default="RGB-D coverage loop -> open-vocabulary grounding -> semantic BEV -> object memory -> landmark retrieval")
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--traditional-bev", required=True)
    parser.add_argument("--object-evidence", required=True)
    parser.add_argument("--semantic-bev", required=True)
    parser.add_argument("--confidence-map", required=True)
    parser.add_argument("--retrieval-bev", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--candidates-json", required=True)
    parser.add_argument("--landmarks-json", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_json(args.metrics_json)
    candidates = _read_json(args.candidates_json)
    landmarks = _read_json(args.landmarks_json)

    images = {
        "RGB observation": Path(args.rgb),
        "Grounding evidence": Path(args.overlay),
        "G: Geometry BEV": Path(args.traditional_bev),
        "O: Object evidence": Path(args.object_evidence),
        "S: Semantic BEV": Path(args.semantic_bev),
        "S confidence": Path(args.confidence_map),
        "L: Landmark retrieval": Path(args.retrieval_bev),
    }
    figure = _render_figure(args.title, args.subtitle, images, metrics, candidates, landmarks)
    figure_path = out_dir / "paper_semantic_bev_overview.png"
    figure.save(figure_path, quality=95)
    metadata = {
        "title": args.title,
        "subtitle": args.subtitle,
        "inputs": {key: str(value) for key, value in images.items()},
        "metrics": _metric_summary(metrics),
        "num_candidates": len(candidates.get("items", [])),
        "num_landmarks": len(landmarks),
        "outputs": {
            "figure": str(figure_path),
            "html": str(out_dir / "paper_semantic_bev_overview.html"),
            "metadata": str(out_dir / "paper_semantic_bev_overview_metadata.json"),
        },
    }
    (out_dir / "paper_semantic_bev_overview_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_html(out_dir / "paper_semantic_bev_overview.html", metadata, figure_path.name)
    print(json.dumps(metadata, indent=2))


def _render_figure(
    title: str,
    subtitle: str,
    image_paths: dict[str, Path],
    metrics: dict[str, Any],
    candidates: dict[str, Any],
    landmarks: list[dict[str, Any]],
) -> Image.Image:
    width = 2400
    margin = 56
    gap = 28
    header_h = 210
    panel_w = (width - 2 * margin - 2 * gap) // 3
    panel_h = 520
    row1_y = header_h
    row2_y = row1_y + panel_h + 70
    summary_h = 270
    height = row2_y + panel_h + summary_h + margin

    canvas = Image.new("RGB", (width, height), "#f6f8fb")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(50, bold=True)
    subtitle_font = _font(25)
    panel_title_font = _font(25, bold=True)
    body_font = _font(22)
    small_font = _font(18)

    draw.text((margin, 40), title, fill="#172033", font=title_font)
    draw.text((margin, 108), subtitle, fill="#506070", font=subtitle_font)
    _draw_metric_strip(draw, (margin, 152), width - 2 * margin, metrics, candidates, landmarks, body_font, small_font)

    panels = [
        ("A. First-person RGB", image_paths["RGB observation"], "Raw visual input from coverage traversal."),
        ("B. Open-vocabulary evidence", image_paths["Grounding evidence"], "GroundingDINO detections before BEV fusion."),
        ("C. Traditional BEV (G)", image_paths["G: Geometry BEV"], "Free / occupied / unknown geometry for navigation."),
        ("D. Object projection (O)", image_paths["O: Object evidence"], "Object inventory projected into the BEV frame."),
        ("E. Semantic BEV (S)", image_paths["S: Semantic BEV"], "Class evidence fused on top of geometry substrate."),
        ("F. Landmark retrieval (L)", image_paths["L: Landmark retrieval"], "Object memory converted into queryable landmark nodes."),
    ]
    for idx, (panel_title, path, caption) in enumerate(panels):
        row = idx // 3
        col = idx % 3
        x = margin + col * (panel_w + gap)
        y = row1_y if row == 0 else row2_y
        _draw_panel(canvas, draw, x, y, panel_w, panel_h, panel_title, caption, path, panel_title_font, small_font)

    summary_y = row2_y + panel_h + 34
    _draw_summary(draw, margin, summary_y, width - 2 * margin, summary_h - 30, body_font, small_font)
    return canvas


def _draw_metric_strip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    width: int,
    metrics: dict[str, Any],
    candidates: dict[str, Any],
    landmarks: list[dict[str, Any]],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    h = 44
    summary = _metric_summary(metrics)
    rows = [
        ("Candidates", str(len(candidates.get("items", [])))),
        ("Landmarks", str(len(landmarks))),
        ("Precision", f"{summary['precision']:.3f}"),
        ("Recall", f"{summary['recall']:.3f}"),
        ("F1", f"{summary['f1']:.3f}"),
        ("Centroid err.", f"{summary['mean_error_m']:.3f} m"),
    ]
    cell_w = width // len(rows)
    for idx, (label, value) in enumerate(rows):
        cx = x + idx * cell_w
        fill = "#ffffff" if idx % 2 == 0 else "#eef3f8"
        draw.rounded_rectangle((cx, y, cx + cell_w - 6, y + h), radius=8, fill=fill, outline="#d7dfe8", width=1)
        draw.text((cx + 14, y + 10), label, fill="#5d6b7a", font=small_font)
        tw = draw.textlength(value, font=font)
        draw.text((cx + cell_w - 22 - tw, y + 7), value, fill="#111827", font=font)


def _draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    caption: str,
    image_path: Path,
    title_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill="#ffffff", outline="#d8e0ea", width=2)
    draw.text((x + 20, y + 18), title, fill="#172033", font=title_font)
    img_box = (x + 20, y + 66, x + w - 20, y + h - 58)
    image = Image.open(image_path).convert("RGB")
    if not title.startswith(("A.", "B.")):
        image = _crop_visual_content(image, ignore_top_left=not title.startswith("F."))
    fitted = _fit(image, img_box[2] - img_box[0], img_box[3] - img_box[1])
    ix = img_box[0] + (img_box[2] - img_box[0] - fitted.width) // 2
    iy = img_box[1] + (img_box[3] - img_box[1] - fitted.height) // 2
    draw.rectangle(img_box, fill="#f9fafb", outline="#e5eaf0")
    canvas.paste(fitted, (ix, iy))
    draw.text((x + 20, y + h - 42), caption, fill="#526170", font=small_font)


def _draw_summary(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, font: ImageFont.ImageFont, small_font: ImageFont.ImageFont) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill="#ffffff", outline="#d8e0ea", width=2)
    draw.text((x + 22, y + 18), "Representation contract", fill="#172033", font=font)
    text = (
        "G handles navigable geometry; S stores semantic evidence and confidence; "
        "O keeps reusable object memory; L exposes queryable landmarks for planning. "
        "The colored BEV is a visualization of structured state, not the only planner input."
    )
    _wrapped(draw, text, x + 22, y + 62, w - 44, small_font, "#4b5563", line_gap=8)
    rows = [
        ("Planner-facing", "goal query + G summary + active/stale O + top-k L + compact S summary"),
        ("Diagnostic", "RGB overlay, semantic BEV image, confidence map, oracle TP/FP/FN metrics"),
        ("Current default", "GroundingDINO box evidence with multi-view/confidence filtering"),
    ]
    cy = y + 140
    for label, value in rows:
        draw.text((x + 26, cy), label, fill="#111827", font=small_font)
        draw.text((x + 210, cy), value, fill="#5b6675", font=small_font)
        cy += 34


def _fit(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _crop_visual_content(image: Image.Image, ignore_top_left: bool) -> Image.Image:
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    brightness = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    mask = (brightness < 185) | (saturation > 34)
    if ignore_top_left:
        h, w = mask.shape
        mask[: int(h * 0.26), : int(w * 0.58)] = False
    ys, xs = np.where(mask)
    if xs.size < 64 or ys.size < 64:
        return image
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    if w < image.width * 0.08 or h < image.height * 0.08:
        return image
    pad = max(18, int(max(w, h) * 0.10))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.width - 1, x2 + pad)
    y2 = min(image.height - 1, y2 + pad)
    return image.crop((x1, y1, x2 + 1, y2 + 1))


def _metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    validation = metrics.get("validation", {})
    return {
        "precision": float(validation.get("precision", 0.0)),
        "recall": float(validation.get("recall", 0.0)),
        "f1": float(validation.get("f1", 0.0)),
        "mean_error_m": float(validation.get("mean_centroid_error_m", 0.0) or 0.0),
    }


def _wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    font: ImageFont.ImageFont,
    fill: str,
    line_gap: int = 4,
) -> None:
    words = text.split()
    line = ""
    cy = y
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
            continue
        if line:
            draw.text((x, cy), line, fill=fill, font=font)
            cy += font.size + line_gap if hasattr(font, "size") else 24 + line_gap
        line = word
    if line:
        draw.text((x, cy), line, fill=fill, font=font)


def _write_html(path: Path, metadata: dict[str, Any], figure_name: str) -> None:
    title = html.escape(metadata["title"])
    subtitle = html.escape(metadata["subtitle"])
    metrics = metadata["metrics"]
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202124;background:#fff;line-height:1.5}}
    h1{{font-size:28px;margin:0 0 6px}}
    .note{{color:#4b5563;max-width:1100px}}
    img{{max-width:100%;border:1px solid #d8e0ea;border-radius:8px;background:#f8fafc}}
    table{{border-collapse:collapse;margin:16px 0;width:760px;max-width:100%}}
    th,td{{border:1px solid #d8e0ea;padding:8px;text-align:left}}
    th{{background:#eef3f8}}
    code{{background:#eef2f7;padding:2px 4px;border-radius:4px}}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="note">{subtitle}</p>
  <img src="{html.escape(figure_name)}" alt="paper semantic BEV overview">
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Precision</td><td>{metrics['precision']:.3f}</td></tr>
    <tr><td>Recall</td><td>{metrics['recall']:.3f}</td></tr>
    <tr><td>F1</td><td>{metrics['f1']:.3f}</td></tr>
    <tr><td>Mean centroid error</td><td>{metrics['mean_error_m']:.3f} m</td></tr>
    <tr><td>Candidates</td><td>{metadata['num_candidates']}</td></tr>
    <tr><td>Landmarks</td><td>{metadata['num_landmarks']}</td></tr>
  </table>
  <p class="note">This figure treats the colored semantic BEV as a visualization of structured state: <code>G/S/O/L</code>, not as the only planner input.</p>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
