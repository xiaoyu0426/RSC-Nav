#!/usr/bin/env python3
"""Replay GroundingDINO on a frozen RGB-D trajectory for paired auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from m25_groundingdino_export import (
    _detect,
    _load_detector,
    _project_box_detection,
)
from semantic_task_profile import get_task_profile


@dataclass
class ReplayTrack:
    track_id: int
    label: str
    position_sum: np.ndarray
    weight_sum: float

    @property
    def position(self) -> np.ndarray:
        return self.position_sum / max(1e-6, self.weight_sum)

    def add(self, position: np.ndarray, score: float) -> None:
        weight = max(0.01, float(score))
        self.position_sum += position * weight
        self.weight_sum += weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-metadata", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--task-profile", default="door")
    parser.add_argument("--labels")
    parser.add_argument(
        "--model-id",
        default="downloads/hf_models/grounding-dino-tiny",
    )
    parser.add_argument("--box-threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.22)
    parser.add_argument("--max-detections", type=int, default=32)
    parser.add_argument("--track-merge-radius-m", type=float)
    parser.add_argument("--depth-min-m", type=float, default=0.05)
    parser.add_argument("--depth-max-m", type=float, default=6.0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    metadata_path = Path(args.frames_metadata).expanduser().resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    profile = get_task_profile(args.task_profile)
    labels = [
        item.strip().lower()
        for item in (
            args.labels or ",".join(profile.detector_labels)
        ).split(",")
        if item.strip()
    ]
    merge_radius_m = (
        float(profile.track_merge_radius_m)
        if args.track_merge_radius_m is None
        else float(args.track_merge_radius_m)
    )
    selected_frames = _select_frames(
        metadata.get("frames", []),
        start=max(0, int(args.frame_start)),
        end=args.frame_end,
        stride=max(1, int(args.frame_stride)),
        max_frames=args.max_frames,
    )
    if not selected_frames:
        raise ValueError("No frames selected for detector replay")

    generator_path = Path(__file__).resolve()
    algorithm_contract = {
        "schema_version": 1,
        "generator_sha256": _sha256(generator_path),
        "backend": "GroundingDINO",
        "model_id": str(args.model_id),
        "model_artifacts": _model_artifact_manifest(args.model_id),
        "resolution": int(metadata["resolution"]),
        "labels": labels,
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "max_detections": int(args.max_detections),
        "projection": {
            "geometry_source": "saved online depth",
            "pose_source": "saved exact Habitat sensor pose",
            "hfov_deg": float(metadata.get("hfov_deg", 90.0)),
            "depth_min_m": float(args.depth_min_m),
            "depth_max_m": float(args.depth_max_m),
        },
        "canonicalization": {
            "target_label": profile.target_label,
            "target_aliases": list(profile.target_aliases),
        },
        "tracking": {
            "method": "nearest same-label online centroid merge",
            "merge_radius_m": merge_radius_m,
        },
        "semantic_oracle_access": False,
    }

    detector = _load_detector(args.model_id)
    tracks: list[ReplayTrack] = []
    detections: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    for selected_index, frame in enumerate(selected_frames):
        if selected_index == 0 or selected_index % 100 == 0:
            print(
                json.dumps(
                    {
                        "stage": "detector_replay",
                        "selected_index": selected_index,
                        "selected_total": len(selected_frames),
                        "frame_index": int(frame["frame_index"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        rgb_path = _source_path(frame.get("rgb_path"), metadata_path)
        depth_path = _source_path(frame.get("depth_npy"), metadata_path)
        image = Image.open(rgb_path).convert("RGB")
        started = time.perf_counter()
        frame_detections = _detect(
            detector,
            image,
            labels,
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
            max_detections=int(args.max_detections),
        )
        inference_ms.append((time.perf_counter() - started) * 1000.0)
        depth = np.squeeze(np.asarray(np.load(depth_path), dtype=np.float32))
        if depth.shape != (image.height, image.width):
            raise ValueError(
                f"Depth shape {depth.shape} does not match RGB "
                f"{(image.height, image.width)} for frame "
                f"{frame['frame_index']}"
            )
        projected = []
        for detection in frame_detections:
            projection = _project_box_detection(
                detection,
                depth,
                frame,
                hfov_deg=float(metadata.get("hfov_deg", 90.0)),
                depth_min_m=float(args.depth_min_m),
                depth_max_m=float(args.depth_max_m),
            )
            if projection is None:
                continue
            canonical_label = profile.canonical_label(detection["label"])
            row = {
                **detection,
                **projection,
                "canonical_label": canonical_label,
                "frame_index": int(frame["frame_index"]),
                "rgb_path": str(rgb_path),
                "online": False,
                "replay_causal": True,
            }
            _assign_track(
                tracks,
                row,
                merge_radius_m=merge_radius_m,
            )
            projected.append(row)
        detections.extend(projected)

    timing = _distribution(inference_ms)
    output = {
        "schema_version": 1,
        "source": {
            "frames_metadata": str(metadata_path),
            "frames_metadata_sha256": _sha256(metadata_path),
            "selected_frames": len(selected_frames),
            "frame_start": int(args.frame_start),
            "frame_end": (
                int(args.frame_end) if args.frame_end is not None else None
            ),
            "frame_stride": max(1, int(args.frame_stride)),
        },
        "algorithm_contract": algorithm_contract,
        "algorithm_sha256": _canonical_sha256(algorithm_contract),
        "resources": {
            "inference_ms": timing,
            "peak_cuda_memory_gb": _peak_cuda_memory_gb(),
        },
        "num_tracks": len(tracks),
        "detections": detections,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "selected_frames": len(selected_frames),
                "detections": len(detections),
                "tracks": len(tracks),
                "inference_p95_ms": timing["p95"],
            },
            indent=2,
        )
    )
    return 0


def _assign_track(
    tracks: list[ReplayTrack],
    detection: dict[str, Any],
    *,
    merge_radius_m: float,
) -> None:
    label = str(detection["canonical_label"])
    position = np.asarray(detection["position_3d"], dtype=np.float32)
    candidates = [track for track in tracks if track.label == label]
    best = min(
        candidates,
        key=lambda track: float(
            np.linalg.norm(track.position[[0, 2]] - position[[0, 2]])
        ),
        default=None,
    )
    distance = (
        float(np.linalg.norm(best.position[[0, 2]] - position[[0, 2]]))
        if best is not None
        else math.inf
    )
    score = float(detection["score"])
    if best is None or distance > merge_radius_m:
        weight = max(0.01, score)
        best = ReplayTrack(
            track_id=len(tracks) + 1,
            label=label,
            position_sum=position * weight,
            weight_sum=weight,
        )
        tracks.append(best)
    else:
        best.add(position, score)
    detection["online_track_id"] = best.track_id


def _select_frames(
    frames: list[dict[str, Any]],
    *,
    start: int,
    end: int | None,
    stride: int,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    selected = [
        frame
        for frame in frames
        if int(frame["frame_index"]) >= start
        and (end is None or int(frame["frame_index"]) <= int(end))
        and (int(frame["frame_index"]) - start) % stride == 0
    ]
    if max_frames is not None:
        selected = selected[: max(0, int(max_frames))]
    return selected


def _source_path(path_value: Any, metadata_path: Path) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Frame source path is missing")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = metadata_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _model_artifact_manifest(model_id: str) -> dict[str, Any]:
    path = Path(model_id).expanduser()
    if not path.is_dir():
        return {"local_directory": False, "identifier": str(model_id)}
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
        )
    return {
        "local_directory": True,
        "directory": str(path.resolve()),
        "files": files,
        "manifest_sha256": _canonical_sha256({"files": files}),
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _peak_cuda_memory_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        return None


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
