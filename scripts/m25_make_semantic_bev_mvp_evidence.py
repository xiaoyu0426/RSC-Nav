from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SEMANTIC_COLORS = {
    "wall": "#2f2f2f",
    "door": "#e15759",
    "table": "#f28e2b",
    "chair": "#59a14f",
    "bed": "#1f77b4",
    "sofa": "#9467bd",
}

BASE_LEGEND = [
    ("unknown", (217, 217, 217)),
    ("free/explored", (255, 255, 255)),
    ("occupied", (51, 51, 51)),
    ("trajectory", (31, 119, 180)),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create M2.5 RGB/OWLv2 -> BEV -> semantic BEV MVP evidence visuals.")
    parser.add_argument("--bev-state", required=True)
    parser.add_argument("--candidates-json", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--owlv2-overlay", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--evidence-radius-cells", type=int, default=5)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bev = np.load(args.bev_state, allow_pickle=False)
    metadata = json.loads(str(bev["metadata"]))
    candidates_payload = _read_json(args.candidates_json)
    metrics = _read_json(args.metrics_json)
    matched = {item["predicted_id"]: item for item in metrics["validation"].get("matches", [])}

    context = _RenderContext(
        metadata=metadata,
        occupancy_logodds=bev["occupancy_logodds"].astype(np.float32),
        explored=bev["explored"].astype(bool),
        trajectory=bev["trajectory"].astype(np.int32),
        candidates=candidates_payload["items"],
        metrics=metrics,
        matched=matched,
        scale=max(2, int(args.scale)),
    )

    traditional = render_traditional_bev(context)
    projection = render_projection_evidence(context)
    semantic_mvp, semantic_state, semantic_confidence = render_semantic_bev_mvp(context, radius_cells=max(1, int(args.evidence_radius_cells)))
    confidence_map = render_semantic_confidence_map(context, semantic_state, semantic_confidence)

    paths = {
        "traditional_bev": out_dir / "traditional_bev_phase2_style.png",
        "object_projection": out_dir / "object_inventory_projection_evidence.png",
        "semantic_mvp": out_dir / "semantic_bev_from_owlv2_mvp.png",
        "confidence_map": out_dir / "semantic_evidence_confidence.png",
        "pipeline": out_dir / "rgb_to_semantic_bev_mvp_pipeline.png",
        "html": out_dir / "rgb_to_semantic_bev_mvp.html",
        "metadata": out_dir / "rgb_to_semantic_bev_mvp_metadata.json",
    }
    traditional.save(paths["traditional_bev"])
    projection.save(paths["object_projection"])
    semantic_mvp.save(paths["semantic_mvp"])
    confidence_map.save(paths["confidence_map"])

    pipeline = render_pipeline(
        rgb_path=Path(args.rgb),
        overlay_path=Path(args.owlv2_overlay),
        traditional=traditional,
        projection=projection,
        semantic_mvp=semantic_mvp,
        confidence_map=confidence_map,
        context=context,
    )
    pipeline.save(paths["pipeline"])

    mask_backend = candidates_payload.get("metadata", {}).get("mask_backend", "unknown")
    caveat = (
        "This uses GroundingDINO box-prompted SAM mask evidence projected through depth and pose. It is mask-level evidence, but still depends on detector prompts, depth validity, and multi-view filtering."
        if mask_backend == "sam"
        else "This is object-centric semantic BEV evidence from detector box/depth projection, not dense SAM/GroundingDINO mask-level segmentation yet."
    )
    summary = {
        "goal": "RGB/OWLv2 -> traditional BEV + object projection evidence -> Phase2-style semantic BEV MVP",
        "inputs": {
            "bev_state": str(Path(args.bev_state).resolve()),
            "candidates_json": str(Path(args.candidates_json).resolve()),
            "metrics_json": str(Path(args.metrics_json).resolve()),
            "rgb": str(Path(args.rgb).resolve()),
            "owlv2_overlay": str(Path(args.owlv2_overlay).resolve()),
        },
        "grid": {
            "size": list(context.grid_size),
            "resolution_m": context.resolution,
            "origin_world_xz": list(context.origin_world_xz),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "metrics": {
            "projected_candidates": len(context.candidates),
            "semantic_mvp_cells": int((semantic_state >= 0).sum()),
            "semantic_confidence_mean": float(semantic_confidence[semantic_confidence > 0].mean()) if np.any(semantic_confidence > 0) else 0.0,
            "semantic_confidence_max": float(semantic_confidence.max()) if semantic_confidence.size else 0.0,
            "tp": metrics["validation"]["true_positive"],
            "fp": metrics["validation"]["false_positive"],
            "fn": metrics["validation"]["false_negative"],
            "precision": metrics["validation"]["precision"],
            "recall": metrics["validation"]["recall"],
            "f1": metrics["validation"]["f1"],
            "mask_backend": mask_backend,
        },
        "representation_note": (
            "Following common semantic map / semantic BEV practice, the PNG is only a visualization. "
            "The intended internal representation is occupancy/explored geometry plus per-class semantic evidence/confidence channels and an object inventory."
        ),
        "caveat": caveat,
    }
    paths["metadata"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(paths["html"], summary)
    print(json.dumps(summary, indent=2))


class _RenderContext:
    def __init__(
        self,
        metadata: dict[str, Any],
        occupancy_logodds: np.ndarray,
        explored: np.ndarray,
        trajectory: np.ndarray,
        candidates: list[dict[str, Any]],
        metrics: dict[str, Any],
        matched: dict[str, dict[str, Any]],
        scale: int,
    ) -> None:
        self.metadata = metadata
        self.occupancy_logodds = occupancy_logodds
        self.explored = explored
        self.trajectory = trajectory
        self.candidates = candidates
        self.metrics = metrics
        self.matched = matched
        self.scale = scale
        self.grid_size = tuple(int(v) for v in metadata["grid_size"])
        self.resolution = float(metadata["resolution"])
        self.origin_world_xz = tuple(float(v) for v in metadata["origin_world_xz"])
        self.categories = list(metadata.get("semantic_categories", ["wall", "door", "table", "chair", "bed", "sofa"]))

    def grid_to_px(self, col: float, row: float) -> tuple[float, float]:
        return col * self.scale + self.scale / 2, (self.grid_size[1] - 1 - row) * self.scale + self.scale / 2

    def world_to_grid(self, position_3d: list[float]) -> tuple[float, float]:
        x, _, z = position_3d
        return (float(x) - self.origin_world_xz[0]) / self.resolution, (float(z) - self.origin_world_xz[1]) / self.resolution

    def in_grid(self, col: float, row: float) -> bool:
        return -0.5 <= col <= self.grid_size[0] - 0.5 and -0.5 <= row <= self.grid_size[1] - 0.5


def render_traditional_bev(ctx: _RenderContext) -> Image.Image:
    state = np.full(ctx.grid_size, 0, dtype=np.int8)
    state[np.logical_and(ctx.explored, ctx.occupancy_logodds <= 0.2)] = 1
    state[ctx.occupancy_logodds > 0.2] = 2
    colors = {
        0: (217, 217, 217),
        1: (255, 255, 255),
        2: (51, 51, 51),
    }
    img = _grid_image(ctx, state, colors)
    draw = ImageDraw.Draw(img)
    _draw_trajectory(draw, ctx)
    _draw_title(draw, "B. Traditional BEV: geometry only")
    _draw_base_legend(draw, (10, 36))
    return img


def render_projection_evidence(ctx: _RenderContext) -> Image.Image:
    img = render_traditional_bev(ctx)
    draw = ImageDraw.Draw(img)
    font = _font(11)
    for item in ctx.candidates:
        col, row = ctx.world_to_grid(item["position_3d"])
        if not ctx.in_grid(col, row):
            continue
        px, py = ctx.grid_to_px(col, row)
        label = item["label"]
        fill = _rgb(SEMANTIC_COLORS.get(label, "#9467bd"))
        ring = (35, 164, 85) if item["id"] in ctx.matched else (213, 62, 62)
        r = 9
        draw.ellipse((px - r - 5, py - r - 5, px + r + 5, py + r + 5), outline=ring, width=4)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=fill, outline=(0, 0, 0), width=2)
        text = f"{label} {float(item.get('confidence', 0.0)):.2f}"
        _label(draw, text, px + r + 6, py - r - 3, font, ring, img.size)
    _draw_title(draw, "C. Object inventory projected to BEV")
    _draw_base_legend(draw, (10, 36))
    _draw_match_legend(draw, (10, 122))
    return img


def render_semantic_bev_mvp(ctx: _RenderContext, radius_cells: int) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    evidence = np.zeros((len(ctx.categories), *ctx.grid_size), dtype=np.float32)
    for item in ctx.candidates:
        label = item["label"]
        if label not in ctx.categories:
            continue
        col, row = ctx.world_to_grid(item["position_3d"])
        if not ctx.in_grid(col, row):
            continue
        class_idx = ctx.categories.index(label)
        confidence = max(0.01, float(item.get("confidence", 0.0)))
        for gx, gy, dist in _disk_cells(int(round(col)), int(round(row)), radius_cells, ctx.grid_size):
            # Smooth footprint: strongest at centroid, fades out at edge.
            evidence[class_idx, gx, gy] += confidence * math.exp(-0.5 * (dist / max(1.0, radius_cells * 0.55)) ** 2)

    state = np.full(ctx.grid_size, -1, dtype=np.int16)
    max_ev = evidence.max(axis=0)
    state[max_ev > 0] = np.argmax(evidence, axis=0)[max_ev > 0]
    confidence = max_ev / max(1e-6, float(max_ev.max())) if max_ev.size else max_ev

    # Start from traditional BEV so the final map keeps the geometry/explored context.
    img = render_traditional_bev(ctx).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for gx in range(ctx.grid_size[0]):
        for gy in range(ctx.grid_size[1]):
            idx = int(state[gx, gy])
            if idx < 0:
                continue
            category = ctx.categories[idx]
            r, g, b = _rgb(SEMANTIC_COLORS.get(category, "#9467bd"))
            # Preserve occupied cells but still let object evidence be visible around them.
            conf = float(confidence[gx, gy])
            alpha = int(90 + 145 * min(1.0, conf))
            if ctx.occupancy_logodds[gx, gy] > 0.2:
                alpha = min(alpha, 165)
            x0, y0 = ctx.grid_to_px(gx - 0.5, gy + 0.5)
            x1, y1 = ctx.grid_to_px(gx + 0.5, gy - 0.5)
            draw.rectangle((x0, y0, x1, y1), fill=(r, g, b, alpha))
    merged = Image.alpha_composite(img, overlay).convert("RGB")
    draw2 = ImageDraw.Draw(merged)
    _draw_trajectory(draw2, ctx)
    _draw_candidate_rings(draw2, ctx, show_text=False)
    _draw_title(draw2, "D. Semantic BEV: geometry + evidence channels")
    _draw_semantic_legend(draw2, ctx, (10, 36))
    _draw_match_legend(draw2, (10, 198))
    return merged, state, confidence


def render_semantic_confidence_map(ctx: _RenderContext, state: np.ndarray, confidence: np.ndarray) -> Image.Image:
    img = Image.new("RGB", (ctx.grid_size[0] * ctx.scale, ctx.grid_size[1] * ctx.scale), (246, 248, 250))
    pix = img.load()
    for gx in range(ctx.grid_size[0]):
        for gy in range(ctx.grid_size[1]):
            py = ctx.grid_size[1] - 1 - gy
            if int(state[gx, gy]) < 0:
                color = (220, 225, 230) if ctx.explored[gx, gy] else (238, 240, 242)
            else:
                v = max(0.0, min(1.0, float(confidence[gx, gy])))
                color = (int(255 - 220 * v), int(246 - 126 * v), int(214 - 174 * v))
            for dx in range(ctx.scale):
                for dy in range(ctx.scale):
                    pix[gx * ctx.scale + dx, py * ctx.scale + dy] = color
    draw = ImageDraw.Draw(img)
    _draw_trajectory(draw, ctx)
    _draw_title(draw, "E. Semantic evidence confidence")
    _draw_confidence_legend(draw, (10, 36))
    return img


def render_pipeline(
    rgb_path: Path,
    overlay_path: Path,
    traditional: Image.Image,
    projection: Image.Image,
    semantic_mvp: Image.Image,
    confidence_map: Image.Image,
    context: _RenderContext,
) -> Image.Image:
    target_w = 520
    panel_h = 520
    rgb = _fit_square(Image.open(rgb_path).convert("RGB"), target_w)
    owlv2 = _fit_square(Image.open(overlay_path).convert("RGB"), target_w)
    left = Image.new("RGB", (target_w, target_w * 2 + 80), "white")
    draw = ImageDraw.Draw(left)
    title = _font(18, bold=True)
    draw.text((12, 8), "A1. RGB frame", fill=(20, 20, 20), font=title)
    left.paste(rgb, (0, 40))
    draw.text((12, target_w + 48), "A2. OWLv2 detection overlay", fill=(20, 20, 20), font=title)
    left.paste(owlv2, (0, target_w + 80))

    maps = [
        ("B. Traditional BEV", _fit_map_crop(traditional, context, panel_h)),
        ("C. Object Projection Evidence", _fit_map_crop(projection, context, panel_h)),
        ("D. Semantic BEV Evidence", _fit_map_crop(semantic_mvp, context, panel_h)),
        ("E. Evidence Confidence", _fit_map_crop(confidence_map, context, panel_h)),
    ]
    gap = 18
    width = target_w + gap + panel_h * 2 + gap + 360
    height = max(left.height, panel_h * 2 + gap + 90)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(left, (0, 0))
    x0_maps = target_w + gap
    y0_maps = 40
    for index, (panel_title, panel) in enumerate(maps):
        x = x0_maps + (index % 2) * (panel_h + gap)
        y = y0_maps + (index // 2) * (panel_h + gap)
        draw = ImageDraw.Draw(canvas)
        draw.text((x + 8, y - 26), panel_title, fill=(20, 20, 20), font=_font(17, bold=True))
        canvas.paste(panel, (x, y))

    draw = ImageDraw.Draw(canvas)
    arrow_y = y0_maps + panel_h // 2
    # connectors from OWLv2 overlay to map chain
    _arrow(draw, (target_w - 24, target_w + 80 + target_w // 2), (x0_maps - 8, arrow_y), fill=(40, 90, 140))
    _arrow(draw, (x0_maps + panel_h - 8, arrow_y), (x0_maps + panel_h + gap + 8, arrow_y), fill=(40, 90, 140))
    _arrow(draw, (x0_maps + panel_h + gap + panel_h // 2, y0_maps + panel_h + 8), (x0_maps + panel_h + gap + panel_h // 2, y0_maps + panel_h + gap - 8), fill=(40, 90, 140))

    x0 = x0_maps + panel_h * 2 + gap + 18
    draw.text((x0, 42), "MVP Evidence Chain", fill=(20, 20, 20), font=_font(20, bold=True))
    rows = [
        ("Detector", "OWLv2"),
        ("Mask backend", "box evidence"),
        ("Object candidates", str(len(context.candidates))),
        ("TP / FP / FN", f"{context.metrics['validation']['true_positive']} / {context.metrics['validation']['false_positive']} / {context.metrics['validation']['false_negative']}"),
        ("P / R / F1", f"{context.metrics['validation']['precision']:.3f} / {context.metrics['validation']['recall']:.3f} / {context.metrics['validation']['f1']:.3f}"),
        ("Map state", "occupancy + semantic evidence"),
    ]
    yy = 86
    for key, value in rows:
        draw.text((x0, yy), key, fill=(90, 90, 90), font=_font(13))
        draw.text((x0 + 130, yy), value, fill=(20, 20, 20), font=_font(13, bold=True))
        yy += 25
    yy += 16
    for line in [
        "B keeps the traditional BEV geometry.",
        "C shows object inventory projected into",
        "the same allocentric grid.",
        "D renders semantic evidence channels",
        "on top of the geometry substrate.",
        "E shows normalized evidence strength.",
        "",
        "This is an OWLv2 lightweight baseline.",
        "GroundingDINO + SAM should become the",
        "high-quality mask-level branch.",
    ]:
        draw.text((x0, yy), line, fill=(55, 55, 55), font=_font(13))
        yy += 21

    return canvas


def write_html(path: Path, summary: dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RGB to Semantic BEV MVP Evidence</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202124;line-height:1.45}}
img{{max-width:100%;border:1px solid #d7dce2;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;align-items:start}}
.card{{border:1px solid #dfe5ec;border-radius:6px;padding:12px;background:#f8fafc}}
code{{background:#f1f3f4;padding:2px 4px;border-radius:4px}}
</style></head>
<body>
<h1>RGB / OWLv2 -> BEV -> Semantic BEV MVP Evidence</h1>
<p>Goal: show that RGB-derived OWLv2 object evidence is projected onto the traditional BEV and merged into a Phase2-style semantic BEV memory map.</p>
<p><strong>Representation note:</strong> {summary["representation_note"]}</p>
<img src="rgb_to_semantic_bev_mvp_pipeline.png" alt="RGB to semantic BEV MVP pipeline">
<div class="grid">
<div class="card"><h2>Traditional BEV</h2><img src="traditional_bev_phase2_style.png"></div>
<div class="card"><h2>Object Projection Evidence</h2><img src="object_inventory_projection_evidence.png"></div>
<div class="card"><h2>Semantic BEV MVP</h2><img src="semantic_bev_from_owlv2_mvp.png"></div>
<div class="card"><h2>Semantic Evidence Confidence</h2><img src="semantic_evidence_confidence.png"></div>
</div>
<h2>How To Read The Figure</h2>
<p>Gray/white/black cells are unknown/free/occupied geometry. Colored cells are semantic evidence accumulated from projected object candidates. Green rings are candidates matched to Habitat oracle under the current threshold; red rings are unmatched predictions. This makes the figure closer to common semantic-map practice: the image is a visualization of evidence channels, not a ground-truth semantic segmentation mask.</p>
<h2>Caveat</h2>
<p>{summary["caveat"]}</p>
<p><a href="../m25_grounding_report.html">Back to M2.5 validation report</a></p>
</body></html>
"""
    path.write_text(html, encoding="utf-8")


def _grid_image(ctx: _RenderContext, state: np.ndarray, colors: dict[int, tuple[int, int, int]]) -> Image.Image:
    img = Image.new("RGB", (ctx.grid_size[0] * ctx.scale, ctx.grid_size[1] * ctx.scale), colors[0])
    pix = img.load()
    for gx in range(ctx.grid_size[0]):
        for gy in range(ctx.grid_size[1]):
            color = colors.get(int(state[gx, gy]), colors[0])
            py = ctx.grid_size[1] - 1 - gy
            for dx in range(ctx.scale):
                for dy in range(ctx.scale):
                    pix[gx * ctx.scale + dx, py * ctx.scale + dy] = color
    return img


def _draw_trajectory(draw: ImageDraw.ImageDraw, ctx: _RenderContext) -> None:
    if len(ctx.trajectory) == 0:
        return
    points = [ctx.grid_to_px(float(x), float(y)) for x, y in ctx.trajectory]
    if len(points) > 1:
        draw.line(points, fill=(31, 119, 180), width=5, joint="curve")
    x, y = points[-1]
    draw.line((x - 8, y, x + 8, y), fill=(31, 119, 180), width=3)
    draw.line((x, y - 8, x, y + 8), fill=(31, 119, 180), width=3)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(31, 119, 180))


def _draw_candidate_rings(draw: ImageDraw.ImageDraw, ctx: _RenderContext, show_text: bool) -> None:
    font = _font(11)
    for item in ctx.candidates:
        col, row = ctx.world_to_grid(item["position_3d"])
        if not ctx.in_grid(col, row):
            continue
        px, py = ctx.grid_to_px(col, row)
        ring = (35, 164, 85) if item["id"] in ctx.matched else (213, 62, 62)
        r = 8
        draw.ellipse((px - r - 4, py - r - 4, px + r + 4, py + r + 4), outline=ring, width=3)
        if show_text:
            _label(draw, item["label"], px + 12, py - 8, font, ring, (ctx.grid_size[0] * ctx.scale, ctx.grid_size[1] * ctx.scale))


def _draw_base_legend(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    x, y = origin
    font = _font(10)
    for label, color in BASE_LEGEND:
        draw.rectangle((x, y, x + 12, y + 12), fill=color, outline=(80, 80, 80))
        draw.text((x + 18, y - 1), label, fill=(30, 30, 30), font=font)
        y += 18


def _draw_semantic_legend(draw: ImageDraw.ImageDraw, ctx: _RenderContext, origin: tuple[int, int]) -> None:
    x, y = origin
    font = _font(10)
    draw.text((x, y), "semantic evidence", fill=(20, 20, 20), font=_font(11, bold=True))
    y += 18
    for category in ctx.categories:
        if category not in SEMANTIC_COLORS:
            continue
        draw.rectangle((x, y, x + 12, y + 12), fill=_rgb(SEMANTIC_COLORS[category]), outline=(80, 80, 80))
        draw.text((x + 18, y - 1), category, fill=(30, 30, 30), font=font)
        y += 17


def _draw_match_legend(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    x, y = origin
    font = _font(10)
    rows = [("matched oracle", (35, 164, 85)), ("unmatched pred", (213, 62, 62))]
    for label, color in rows:
        draw.ellipse((x, y, x + 13, y + 13), outline=color, width=3)
        draw.text((x + 20, y - 1), label, fill=(30, 30, 30), font=font)
        y += 18


def _draw_confidence_legend(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    x, y = origin
    font = _font(10)
    draw.text((x, y), "semantic evidence strength", fill=(20, 20, 20), font=_font(11, bold=True))
    y += 20
    for i in range(90):
        v = i / 89
        color = (int(255 - 220 * v), int(246 - 126 * v), int(214 - 174 * v))
        draw.line((x + i, y, x + i, y + 12), fill=color)
    draw.rectangle((x, y, x + 90, y + 12), outline=(80, 80, 80))
    draw.text((x, y + 16), "low", fill=(45, 45, 45), font=font)
    draw.text((x + 67, y + 16), "high", fill=(45, 45, 45), font=font)


def _draw_title(draw: ImageDraw.ImageDraw, text: str) -> None:
    font = _font(14, bold=True)
    draw.rectangle((0, 0, 520, 30), fill=(255, 255, 255))
    draw.text((10, 7), text, fill=(20, 20, 20), font=font)


def _label(draw: ImageDraw.ImageDraw, text: str, x: float, y: float, font: ImageFont.ImageFont, outline: tuple[int, int, int], image_size: tuple[int, int]) -> None:
    width = max(draw.textlength(line, font=font) for line in text.split("\n"))
    height = 14 * len(text.split("\n")) + 4
    x = min(max(2, x), image_size[0] - width - 8)
    y = min(max(2, y), image_size[1] - height - 2)
    draw.rectangle((x - 3, y - 2, x + width + 5, y + height), fill=(255, 255, 255), outline=outline)
    for index, line in enumerate(text.split("\n")):
        draw.text((x, y + index * 14), line, fill=(20, 20, 20), font=font)


def _disk_cells(cx: int, cy: int, radius: int, grid_size: tuple[int, int]):
    for gx in range(cx - radius, cx + radius + 1):
        for gy in range(cy - radius, cy + radius + 1):
            if gx < 0 or gy < 0 or gx >= grid_size[0] or gy >= grid_size[1]:
                continue
            dist = math.hypot(gx - cx, gy - cy)
            if dist <= radius:
                yield gx, gy, dist


def _fit_square(image: Image.Image, size: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def _fit_map_crop(image: Image.Image, ctx: _RenderContext, size: int) -> Image.Image:
    xs: list[float] = []
    ys: list[float] = []
    explored = np.argwhere(ctx.explored)
    if explored.size:
        for gx, gy in explored[:: max(1, len(explored) // 2500)]:
            px, py = ctx.grid_to_px(float(gx), float(gy))
            xs.append(px)
            ys.append(py)
    for item in ctx.candidates:
        col, row = ctx.world_to_grid(item["position_3d"])
        if ctx.in_grid(col, row):
            px, py = ctx.grid_to_px(col, row)
            xs.append(px)
            ys.append(py)
    for gx, gy in ctx.trajectory:
        px, py = ctx.grid_to_px(float(gx), float(gy))
        xs.append(px)
        ys.append(py)
    if not xs or not ys:
        return _fit_square(image, size)
    margin = 90
    left = max(0, int(min(xs) - margin))
    top = max(0, int(min(ys) - margin))
    right = min(image.width, int(max(xs) + margin))
    bottom = min(image.height, int(max(ys) + margin))
    if right <= left or bottom <= top:
        return _fit_square(image, size)
    crop = image.crop((left, top, right, bottom))
    return _fit_square(crop, size)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: tuple[int, int, int]) -> None:
    draw.line((start, end), fill=fill, width=4)
    ex, ey = end
    sx, sy = start
    angle = math.atan2(ey - sy, ex - sx)
    for offset in (math.pi * 0.82, -math.pi * 0.82):
        x = ex + math.cos(angle + offset) * 14
        y = ey + math.sin(angle + offset) * 14
        draw.line((ex, ey, x, y), fill=fill, width=4)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
