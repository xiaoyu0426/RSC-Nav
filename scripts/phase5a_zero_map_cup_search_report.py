from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_bev_mapper import DenseBEVConfig, DenseBEVMapper  # noqa: E402


TASK_TEXT = "自行熟悉房间并找到所有水杯"
CUP_LABELS = {
    "cup",
    "mug",
    "drinking glass",
    "drinking-glass",
    "wine glass",
    "wine-glass",
    "glass",
}
LABEL_COLORS = {
    "cup": "#d62728",
    "bottle": "#17becf",
    "sink": "#00a6d6",
    "faucet": "#2ca02c",
    "counter": "#8c564b",
    "table": "#f28e2b",
    "chair": "#59a14f",
    "bed": "#1f77b4",
    "sofa": "#9467bd",
    "door": "#e15759",
}


@dataclass
class EvidenceTrack:
    label: str
    position_sum: np.ndarray
    score_sum: float
    weight_sum: float
    view_ids: set[int] = field(default_factory=set)
    best_score: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return self.position_sum / max(1e-6, self.weight_sum)

    @property
    def confidence(self) -> float:
        return self.score_sum / max(1e-6, self.weight_sum)


class MatrixRotation:
    def __init__(self, matrix: Any) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float32).reshape(3, 3)

    def transform_vector(self, vector: Any) -> np.ndarray:
        return self.matrix @ np.asarray(vector, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a from-zero semantic-BEV cup-search demo report.")
    parser.add_argument("--frames-metadata", required=True)
    parser.add_argument("--detections-json", action="append", required=True)
    parser.add_argument("--grounding-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task", default=TASK_TEXT)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-video-frames", type=int, default=420)
    parser.add_argument("--max-gif-frames", type=int, default=96)
    parser.add_argument("--cup-min-views", type=int, default=5)
    parser.add_argument("--cup-min-confidence", type=float, default=0.28)
    parser.add_argument("--track-merge-radius-m", type=float, default=0.72)
    parser.add_argument(
        "--oracle-target-instance-ids",
        default="",
        help="Comma-separated semantic instance IDs used only for post-hoc coverage validation.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.frames_metadata).expanduser().resolve()
    detections_paths = [Path(value).expanduser().resolve() for value in args.detections_json]
    grounding_dirs = [Path(value).expanduser().resolve() for value in args.grounding_dir]
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_out = out_dir / "storyboard_frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    metadata = _read_json(metadata_path)
    frames = list(metadata.get("frames", []))
    if not frames:
        raise RuntimeError("frames_metadata has no frames")
    detections = [
        detection
        for detections_path in detections_paths
        for detection in _read_json(detections_path).get("detections", [])
    ]
    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in detections:
        detections_by_frame.setdefault(int(item.get("frame_index", -1)), []).append(item)
    oracle_target_ids = {
        int(value.strip())
        for value in str(args.oracle_target_instance_ids).split(",")
        if value.strip()
    }
    oracle_visible_ids, oracle_visible_frames = _oracle_visibility(frames, oracle_target_ids)

    mapper = _new_mapper(metadata, frames)
    tracks: list[EvidenceTrack] = []
    rendered_paths: list[Path] = []
    frame_records: list[dict[str, Any]] = []
    total_pitch_frames = sum(1 for item in frames if _is_pitch_action(str(item.get("action", ""))))
    observed_pitch_frames = 0

    blank = _compose_frame(
        rgb=Image.open(frames[0]["rgb_path"]).convert("RGB"),
        depth=Image.open(frames[0]["depth_png"]).convert("L"),
        bev=_render_bev(mapper, tracks, current_frame=-1),
        task=args.task,
        stage="初始化：清空 BEV 与对象记忆",
        action="memory_reset",
        frame_index=0,
        total_frames=len(frames),
        explored_ratio=0.0,
        tracks=tracks,
        observed_pitch_frames=0,
        total_pitch_frames=total_pitch_frames,
        current_detections=[],
    )
    for _ in range(max(3, int(args.fps))):
        path = frames_out / f"story_{len(rendered_paths):04d}.jpg"
        blank.save(path, quality=94)
        rendered_paths.append(path)

    for frame_pos, frame in enumerate(frames):
        depth = np.load(frame["depth_npy"]).astype(np.float32)
        mapper.update_from_depth(
            depth=depth,
            agent_position_xyz=np.asarray(frame["agent_position_xyz"], dtype=np.float32),
            sensor_position_xyz=np.asarray(frame["sensor_position_xyz"], dtype=np.float32),
            sensor_rotation=MatrixRotation(frame["sensor_rotation_matrix"]),
            hfov_deg=float(metadata.get("hfov_deg", 90.0)),
        )
        frame_index = int(frame.get("frame_index", frame_pos))
        current_detections = detections_by_frame.get(frame_index, [])
        for detection in current_detections:
            _merge_detection(
                tracks,
                detection,
                frame_index=frame_index,
                merge_radius_m=float(args.track_merge_radius_m),
            )
        action = str(frame.get("action", "observe"))
        if _is_pitch_action(action):
            observed_pitch_frames += 1
        frame_records.append(
            {
                "frame_index": frame_index,
                "action": action,
                "num_detections": len(current_detections),
                "num_tracks": len(tracks),
                "confirmed_cups": len(
                    _confirmed_cups(
                        tracks,
                        min_views=int(args.cup_min_views),
                        min_confidence=float(args.cup_min_confidence),
                    )
                ),
            }
        )
        if not _should_render(frame_pos, frame, len(frames)):
            continue
        rgb_path = _find_overlay(grounding_dirs, frame_index) or Path(frame["rgb_path"])
        composed = _compose_frame(
            rgb=Image.open(rgb_path).convert("RGB"),
            depth=Image.open(frame["depth_png"]).convert("L"),
            bev=_render_bev(mapper, tracks, current_frame=frame_index),
            task=args.task,
            stage=_stage_for_action(action),
            action=action,
            frame_index=frame_pos + 1,
            total_frames=len(frames),
            explored_ratio=float(mapper.explored.mean()),
            tracks=tracks,
            observed_pitch_frames=observed_pitch_frames,
            total_pitch_frames=total_pitch_frames,
            current_detections=current_detections,
        )
        path = frames_out / f"story_{len(rendered_paths):04d}.jpg"
        composed.save(path, quality=94)
        rendered_paths.append(path)

    confirmed_cups = _confirmed_cups(
        tracks,
        min_views=int(args.cup_min_views),
        min_confidence=float(args.cup_min_confidence),
    )
    final_frame = _compose_summary_frame(
        task=args.task,
        bev=_render_bev(mapper, tracks, current_frame=10**9),
        tracks=tracks,
        cups=confirmed_cups,
        explored_ratio=float(mapper.explored.mean()),
        pitch_frames=observed_pitch_frames,
        oracle_visible_count=len(oracle_visible_ids),
    )
    for _ in range(max(12, int(args.fps) * 3)):
        path = frames_out / f"story_{len(rendered_paths):04d}.jpg"
        final_frame.save(path, quality=94)
        rendered_paths.append(path)

    video_paths = _sample_paths(rendered_paths, int(args.max_video_frames), keep_ends=True)
    gif_paths = _sample_paths(rendered_paths, int(args.max_gif_frames), keep_ends=True)
    video_path = out_dir / "zero_map_find_all_cups.mp4"
    gif_path = out_dir / "zero_map_find_all_cups.gif"
    _write_mp4(video_path, video_paths, fps=max(1, int(args.fps)))
    _write_gif(gif_path, gif_paths, fps=max(1, int(args.fps)))

    metrics = {
        "task": args.task,
        "schema_version": "phase5a_zero_map_cup_search_v1",
        "source_frames_metadata": str(metadata_path),
        "source_detections": [str(path) for path in detections_paths],
        "geometry_source": "Habitat RGB-D + exact simulator pose for MVP validation",
        "semantic_source": "GroundingDINO open-vocabulary detections projected with depth + pose",
        "memory_initialized_empty": True,
        "num_input_frames": len(frames),
        "num_projected_detections": len(detections),
        "num_evidence_tracks": len(tracks),
        "num_stable_display_tracks": len(_stable_display_tracks(tracks)),
        "num_confirmed_cups": len(confirmed_cups),
        "confirmed_cups": [_track_dict(item) for item in confirmed_cups],
        "oracle_validation_only": True,
        "oracle_target_instance_ids": sorted(oracle_target_ids),
        "oracle_visible_target_instances": len(oracle_visible_ids),
        "oracle_visible_target_instance_ids": sorted(oracle_visible_ids),
        "oracle_visible_frames": oracle_visible_frames,
        "pitch_scan_frames": observed_pitch_frames,
        "explored_cell_ratio": float(mapper.explored.mean()),
        "search_completion_note": (
            "“所有水杯”在本 MVP 中指当前覆盖与主动扫描下通过多视角确认的杯具轨迹。"
            f"算法确认 {len(confirmed_cups)} 条；事后 oracle 显示输入帧覆盖到 "
            f"{len(oracle_visible_ids)} 个 glass 实例。二者类别粒度不同，不能直接当作实例级召回率。"
            if oracle_target_ids
            else "“所有水杯”指当前覆盖与主动扫描下通过多视角确认的杯具轨迹；"
            "没有目标类别 oracle 时不声称绝对召回率。"
        ),
        "frame_records": frame_records,
        "artifacts": {
            "video": video_path.name,
            "gif": gif_path.name,
            "poster": "zero_map_find_all_cups_poster.png",
        },
    }
    (out_dir / "zero_map_find_all_cups_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_frame.save(out_dir / "zero_map_find_all_cups_poster.png")
    _write_html(out_dir, metrics)
    print(json.dumps({k: metrics[k] for k in ("task", "num_input_frames", "num_projected_detections", "num_confirmed_cups")}, ensure_ascii=False, indent=2))


def _new_mapper(metadata: dict[str, Any], frames: list[dict[str, Any]]) -> DenseBEVMapper:
    fit = (metadata.get("route_plan") or {}).get("bev_fit") or {}
    if fit.get("origin_world_xz") and fit.get("grid_size"):
        origin = tuple(float(value) for value in fit["origin_world_xz"][:2])
        grid = tuple(int(value) for value in fit["grid_size"][:2])
        resolution = float(fit.get("resolution", 0.05))
    else:
        xz = np.asarray([[item["agent_position_xyz"][0], item["agent_position_xyz"][2]] for item in frames], dtype=np.float32)
        resolution = 0.05
        margin = 6.0
        origin = (float(xz[:, 0].min() - margin), float(xz[:, 1].min() - margin))
        span = max(float(np.ptp(xz[:, 0])), float(np.ptp(xz[:, 1]))) + 2 * margin
        side = max(240, int(math.ceil(span / resolution)))
        grid = (side, side)
    config = DenseBEVConfig(grid_size=grid, resolution=resolution, sample_stride=8, obstacle_dilation_radius_cells=1)
    return DenseBEVMapper(origin_world_xz=origin, config=config)


def _canonical_label(label: str) -> str:
    normalized = str(label).strip().lower()
    return "cup" if normalized in CUP_LABELS else normalized


def _merge_detection(tracks: list[EvidenceTrack], detection: dict[str, Any], frame_index: int, merge_radius_m: float) -> None:
    label = _canonical_label(str(detection.get("label", "unknown")))
    position = np.asarray(detection.get("position_3d", []), dtype=np.float32)
    if position.shape != (3,) or not np.isfinite(position).all():
        return
    score = float(detection.get("score", 0.0))
    best: EvidenceTrack | None = None
    best_distance = math.inf
    for track in tracks:
        if track.label != label:
            continue
        distance = float(np.linalg.norm(track.position[[0, 2]] - position[[0, 2]]))
        if distance < best_distance:
            best = track
            best_distance = distance
    weight = max(0.01, score)
    if best is None or best_distance > merge_radius_m:
        tracks.append(
            EvidenceTrack(
                label=label,
                position_sum=position * weight,
                score_sum=score * weight,
                weight_sum=weight,
                view_ids={frame_index},
                best_score=score,
            )
        )
        return
    best.position_sum += position * weight
    best.score_sum += score * weight
    best.weight_sum += weight
    best.view_ids.add(frame_index)
    best.best_score = max(best.best_score, score)


def _confirmed_cups(tracks: list[EvidenceTrack], min_views: int, min_confidence: float) -> list[EvidenceTrack]:
    return [
        item
        for item in tracks
        if item.label == "cup" and len(item.view_ids) >= min_views and item.confidence >= min_confidence
    ]


def _render_bev(mapper: DenseBEVMapper, tracks: list[EvidenceTrack], current_frame: int, size: int = 520) -> Image.Image:
    state = np.flipud(mapper.occupancy_state().T)
    colors = np.asarray(
        [
            [225, 228, 232],
            [251, 252, 253],
            [43, 47, 52],
        ],
        dtype=np.uint8,
    )
    rgb = colors[np.clip(state, 0, 2)]
    image = Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image, "RGBA")
    sx = size / mapper.config.grid_size[0]
    sy = size / mapper.config.grid_size[1]

    def point(cell: tuple[int, int]) -> tuple[float, float]:
        return cell[0] * sx, size - 1 - cell[1] * sy

    if mapper.trajectory:
        trajectory = [point(cell) for cell in mapper.trajectory]
        if len(trajectory) >= 2:
            draw.line(trajectory, fill=(31, 119, 180, 220), width=3)
        x, y = trajectory[-1]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(31, 119, 180, 255), outline="white", width=2)

    font = _font(14)
    display_tracks = _stable_display_tracks(tracks)
    labeled_counts: dict[str, int] = {}
    cup_index = 0
    for track in sorted(display_tracks, key=lambda item: (item.label != "cup", -len(item.view_ids), -item.confidence)):
        cell = mapper.world_to_grid((float(track.position[0]), float(track.position[2])))
        if cell is None:
            continue
        x, y = point(cell)
        color = _hex_rgba(LABEL_COLORS.get(track.label, "#6b7280"), 235)
        radius = 9 if track.label == "cup" else 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255, 240), width=2)
        should_label = track.label == "cup" or labeled_counts.get(track.label, 0) < 1
        if should_label:
            if track.label == "cup":
                cup_index += 1
                label = f"C{cup_index}"
            else:
                label = track.label
            draw.text((x + radius + 3, y - 8), label, font=font, fill=(25, 28, 32, 255), stroke_width=2, stroke_fill=(255, 255, 255, 230))
            labeled_counts[track.label] = labeled_counts.get(track.label, 0) + 1

    draw.rectangle((0, 0, size, 30), fill=(255, 255, 255, 220))
    draw.text((10, 6), "Dynamic semantic BEV / object memory", font=_font(15), fill=(28, 33, 39, 255))
    return image


def _compose_frame(
    rgb: Image.Image,
    depth: Image.Image,
    bev: Image.Image,
    task: str,
    stage: str,
    action: str,
    frame_index: int,
    total_frames: int,
    explored_ratio: float,
    tracks: list[EvidenceTrack],
    observed_pitch_frames: int,
    total_pitch_frames: int,
    current_detections: list[dict[str, Any]],
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(25)
    ui_font = _font(18)
    small_font = _font(15)
    draw.text((28, 17), f"RSC-Nav | {task}", font=title_font, fill="#17212b")
    draw.text((1040, 22), f"{frame_index}/{total_frames}", font=ui_font, fill="#52606d")

    rgb_panel = ImageOps.fit(rgb, (720, 570), method=Image.Resampling.LANCZOS)
    depth_panel = ImageOps.fit(depth.convert("RGB"), (180, 135), method=Image.Resampling.LANCZOS)
    canvas.paste(rgb_panel, (28, 78))
    canvas.paste(depth_panel, (548, 500))
    draw.rectangle((548, 500, 728, 635), outline="#ffffff", width=2)
    draw.rectangle((548, 500, 728, 526), fill="#17212bcc")
    draw.text((558, 505), "Depth", font=small_font, fill="white")
    canvas.paste(ImageOps.fit(bev, (500, 390), method=Image.Resampling.LANCZOS), (752, 78))

    draw.rounded_rectangle((752, 486, 1252, 680), radius=7, fill="#ffffff", outline="#d7dee8", width=1)
    draw.text((772, 502), stage, font=_font(21), fill="#0f5f99")
    draw.text((772, 535), f"action: {action}", font=small_font, fill="#52606d")
    cup_tracks = [item for item in tracks if item.label == "cup"]
    confirmed = [item for item in cup_tracks if len(item.view_ids) >= 5 and item.confidence >= 0.28]
    stable = _stable_display_tracks(tracks)
    status_rows = [
        f"BEV explored cells: {explored_ratio * 100:.1f}%",
        f"stable semantic tracks: {len(stable)}",
        f"confirmed cups: {len(confirmed)}",
        f"tabletop scan: {observed_pitch_frames}/{max(1, total_pitch_frames)} frames",
    ]
    if current_detections:
        current_labels = sorted({_canonical_label(item.get("label", "")) for item in current_detections})
        labels = ", ".join(current_labels[:4])
        status_rows.append(f"current grounding: {labels}")
    for row_index, row in enumerate(status_rows):
        draw.text((772, 565 + row_index * 22), row, font=small_font, fill="#26323d")

    draw.rectangle((28, 78, 748, 130), fill="#111827cc")
    draw.text((42, 91), stage, font=ui_font, fill="#ffffff")
    draw.text((28, 662), "RGB + GroundingDINO | geometry: RGB-D + pose | semantic memory starts empty", font=small_font, fill="#52606d")
    return canvas


def _compose_summary_frame(
    task: str,
    bev: Image.Image,
    tracks: list[EvidenceTrack],
    cups: list[EvidenceTrack],
    explored_ratio: float,
    pitch_frames: int,
    oracle_visible_count: int,
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((38, 30), "Episode complete", font=_font(34), fill="#17212b")
    draw.text((38, 82), task, font=_font(25), fill="#0f5f99")
    canvas.paste(ImageOps.fit(bev, (580, 520), method=Image.Resampling.LANCZOS), (38, 142))
    draw.rounded_rectangle((655, 142, 1240, 662), radius=7, fill="#ffffff", outline="#d7dee8", width=1)
    draw.text((685, 172), "Search summary", font=_font(27), fill="#17212b")
    rows = [
        f"Dynamic BEV explored cells: {explored_ratio * 100:.1f}%",
        f"Stable semantic tracks: {len(_stable_display_tracks(tracks))}",
        f"Tabletop scan frames: {pitch_frames}",
        f"Multi-view confirmed cups: {len(cups)}",
        f"Post-hoc oracle-visible glass IDs: {oracle_visible_count}",
    ]
    for index, row in enumerate(rows):
        draw.text((685, 225 + index * 42), row, font=_font(20), fill="#26323d")
    y = 448
    if cups:
        draw.text((685, y), "Confirmed cup memory:", font=_font(20), fill="#0f5f99")
        y += 34
        for index, cup in enumerate(cups[:6], start=1):
            position = cup.position
            text = f"{index}. ({position[0]:.2f}, {position[2]:.2f}) | conf {cup.confidence:.2f} | {len(cup.view_ids)} views"
            draw.text((705, y), text, font=_font(17), fill="#26323d")
            y += 28
    else:
        draw.text((685, y), "No cup passed multi-view confirmation.", font=_font(20), fill="#b42318")
        draw.text((685, y + 38), "Result is 'not found under current coverage', not a fabricated target.", font=_font(16), fill="#52606d")
    return canvas


def _stage_for_action(action: str) -> str:
    if action == "memory_reset":
        return "初始化：清空 BEV 与对象记忆"
    if action.startswith("target_pitch"):
        return "主动搜索：俯视桌面与近处物体"
    if action.startswith("pitch_scan"):
        return "环境扫描：俯视桌面与近处物体"
    if action.startswith("target_scan"):
        return "主动搜索：围绕候选台面进行多视角确认"
    if action.startswith("target_route"):
        return "主动搜索：沿 navmesh 前往下一处候选区域"
    if action.startswith("target_face"):
        return "主动搜索：平滑朝向候选桌面或台面"
    if action.startswith("yaw_scan"):
        return "环境扫描：45° 分段观察"
    if action == "route_step":
        return "探索建图：沿可执行 navmesh 路线移动"
    return "环境初始化与首帧观测"


def _stable_display_tracks(tracks: list[EvidenceTrack]) -> list[EvidenceTrack]:
    return [
        track
        for track in tracks
        if (
            track.label == "cup"
            and len(track.view_ids) >= 5
            and track.confidence >= 0.28
        )
        or (
            track.label != "cup"
            and len(track.view_ids) >= 8
            and track.confidence >= 0.32
        )
    ]


def _is_pitch_action(action: str) -> bool:
    return action.startswith("pitch_scan") or action.startswith("target_pitch")


def _should_render(frame_pos: int, frame: dict[str, Any], total_frames: int) -> bool:
    action = str(frame.get("action", ""))
    if frame_pos in {0, total_frames - 1}:
        return True
    if action.startswith("target_"):
        return True
    return frame_pos % 3 == 0


def _find_overlay(grounding_dirs: list[Path], frame_index: int) -> Path | None:
    for grounding_dir in grounding_dirs:
        path = grounding_dir / "overlays" / f"frame_{frame_index:04d}_overlay.jpg"
        if path.exists():
            return path
    return None


def _oracle_visibility(
    frames: list[dict[str, Any]],
    target_ids: set[int],
) -> tuple[set[int], int]:
    if not target_ids:
        return set(), 0
    visible_ids: set[int] = set()
    visible_frames = 0
    for frame in frames:
        semantic_path = frame.get("semantic_npy")
        if not semantic_path or not Path(semantic_path).exists():
            continue
        present = set(np.unique(np.load(semantic_path)).astype(int).tolist()) & target_ids
        if present:
            visible_frames += 1
            visible_ids.update(present)
    return visible_ids, visible_frames


def _sample_paths(paths: list[Path], limit: int, keep_ends: bool) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    if not keep_ends:
        indexes = np.linspace(0, len(paths) - 1, limit).round().astype(int)
        return [paths[int(index)] for index in indexes]
    tail = min(18, max(6, limit // 6))
    head_count = max(1, limit - tail)
    head_end = max(0, len(paths) - tail - 1)
    indexes = np.linspace(0, head_end, head_count).round().astype(int).tolist()
    indexes.extend(range(len(paths) - tail, len(paths)))
    return [paths[int(index)] for index in indexes]


def _write_mp4(path: Path, frame_paths: list[Path], fps: int) -> None:
    import imageio.v2 as imageio

    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    try:
        for frame_path in frame_paths:
            writer.append_data(np.asarray(Image.open(frame_path).convert("RGB")))
    finally:
        writer.close()


def _write_gif(path: Path, frame_paths: list[Path], fps: int) -> None:
    frames = [
        ImageOps.contain(Image.open(frame_path).convert("RGB"), (960, 540), method=Image.Resampling.LANCZOS)
        for frame_path in frame_paths
    ]
    palette_frames = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=192) for frame in frames]
    palette_frames[0].save(
        path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=max(40, int(round(1000 / fps))),
        loop=0,
        optimize=False,
    )


def _write_html(out_dir: Path, metrics: dict[str, Any]) -> None:
    cup_rows = "".join(
        "<tr>"
        f"<td>{index}</td><td>{item['confidence']:.3f}</td><td>{item['views']}</td>"
        f"<td>{item['position_3d'][0]:.2f}, {item['position_3d'][1]:.2f}, {item['position_3d'][2]:.2f}</td>"
        "</tr>"
        for index, item in enumerate(metrics["confirmed_cups"], start=1)
    )
    if not cup_rows:
        cup_rows = '<tr><td colspan="4">No cup passed multi-view confirmation in this run.</td></tr>'
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>RSC-Nav Zero-map Cup Search Demo</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f5f7fa;color:#17212b;line-height:1.55}}
main{{max-width:1180px;margin:auto;padding:28px}}h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:30px}}
.lead{{font-size:18px;color:#52606d}}.grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}
.metric{{background:white;border:1px solid #d7dee8;border-radius:7px;padding:15px}}.metric b{{display:block;font-size:25px;color:#0f5f99}}
video,img{{display:block;width:100%;background:#111827;border:1px solid #d7dee8}}table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #d7dee8;padding:8px;text-align:left}}th{{background:#edf2f7}}code{{background:#edf2f7;padding:2px 5px}}
.note{{background:#fff8e8;border-left:4px solid #f59e0b;padding:12px 15px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>RSC-Nav：从零建图并寻找所有水杯</h1>
<p class="lead">自然语言任务：<b>{html.escape(metrics['task'])}</b></p>
<div class="grid">
<div class="metric"><b>{metrics['num_input_frames']}</b>RGB-D observations</div>
<div class="metric"><b>{metrics['num_projected_detections']}</b>projected detections</div>
<div class="metric"><b>{metrics['num_stable_display_tracks']}</b>stable semantic tracks</div>
<div class="metric"><b>{metrics['num_confirmed_cups']}</b>confirmed cups</div>
<div class="metric"><b>{metrics['oracle_visible_target_instances']}</b>oracle-visible glass IDs</div>
</div>
<h2>完整 episode</h2>
<video controls autoplay muted loop src="zero_map_find_all_cups.mp4"></video>
<p><a href="zero_map_find_all_cups.gif">GIF</a> |
<a href="zero_map_find_all_cups_metrics.json">metrics / trace</a> |
<a href="zero_map_find_all_cups_poster.png">poster</a></p>
<h2>证据来源</h2>
<p>Traditional BEV：Habitat RGB-D + pose。语义对象：GroundingDINO open-vocabulary detection，经 depth + pose 投影到 3D，并按多视角证据合并到 object memory。地图与对象记忆在该 episode 开头清空。</p>
<p>Oracle 语义只在 episode 结束后统计可见目标实例，不参与探索、检测、候选位置选择或 object memory 更新。</p>
<p class="note">{html.escape(metrics['search_completion_note'])}</p>
<h2>确认的水杯记忆</h2>
<table><tr><th>#</th><th>confidence</th><th>views</th><th>3D position</th></tr>{cup_rows}</table>
</main></body></html>"""
    (out_dir / "zero_map_find_all_cups.html").write_text(document, encoding="utf-8")


def _track_dict(track: EvidenceTrack) -> dict[str, Any]:
    return {
        "label": track.label,
        "position_3d": [float(value) for value in track.position],
        "confidence": float(track.confidence),
        "best_score": float(track.best_score),
        "views": len(track.view_ids),
        "view_ids": sorted(track.view_ids),
    }


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/workspace/yujiexiao/.rscnav/fonts/STHeiti-Light.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hex_rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), alpha


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
