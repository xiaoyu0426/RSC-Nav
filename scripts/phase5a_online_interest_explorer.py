from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from dense_bev_mapper import DenseBEVConfig, DenseBEVMapper  # noqa: E402
from cup_confirmation import (  # noqa: E402
    CupConfirmationConfig,
    append_independent_observation,
    estimate_depth_surface_relief,
    evaluate_cup_confirmation,
    score_crop_verifier,
)
from interest_exploration import (  # noqa: E402
    InterestConfig,
    approach_cell_for_target,
    choose_semantic_target,
    deep_familiarization_status,
    plan_observed_path,
    planning_free_mask,
    rank_frontier_clusters,
    rank_frontiers,
    reachable_free_mask,
    smooth_motion_evidence_weight,
)
from m25_groundingdino_export import _project_box_detection, _write_overlay  # noqa: E402
from object_memory_store import ObjectMemoryStore  # noqa: E402
from online_semantic_task_planner import (  # noqa: E402
    build_online_planner_request,
    plan_online_task,
)
from scene_semantic_search import (  # noqa: E402
    apply_search_evidence,
    enrich_planner_request,
    initialize_search_beliefs,
    rank_search_candidates,
    select_scene_keyframes,
    understand_scene_with_vlm,
)
from phase23_habitat_control_server import (  # noqa: E402
    HabitatControlSession,
    _depth_image,
    _rgb_array,
    _valid_depth,
    ensure_conda_nvidia_egl_vendor,
)


TASK_TEXT = "请找到房间里的所有水杯，并按位置汇报"
CUP_LABELS = {
    "cup",
    "mug",
    "glass",
    "drinking glass",
    "drinking-glass",
    "wine glass",
    "wine-glass",
}
SURFACE_LABELS = {"table", "counter", "sink"}


@dataclass
class OnlineTrack:
    track_id: int
    label: str
    position_sum: np.ndarray
    score_sum: float
    weight_sum: float
    first_seen_step: int
    last_seen_step: int
    visible_steps: list[int] = field(default_factory=list)
    independent_views: list[tuple[float, float, float]] = field(default_factory=list)
    detection_count: int = 0
    best_score: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return self.position_sum / max(1e-6, self.weight_sum)

    @property
    def confidence(self) -> float:
        return self.score_sum / max(1e-6, self.weight_sum)

    def add(
        self,
        position: np.ndarray,
        score: float,
        step: int,
        camera_xzyaw: tuple[float, float, float],
    ) -> None:
        weight = max(0.01, float(score))
        self.position_sum += position * weight
        self.score_sum += float(score) * weight
        self.weight_sum += weight
        self.last_seen_step = int(step)
        if not self.visible_steps or self.visible_steps[-1] != int(step):
            self.visible_steps.append(int(step))
        self.detection_count += 1
        self.best_score = max(self.best_score, float(score))
        if _is_independent_view(self.independent_views, camera_xzyaw):
            self.independent_views.append(camera_xzyaw)

    def as_interest_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "position_3d": [float(value) for value in self.position],
            "confidence": float(self.confidence),
            "views": len(self.independent_views),
        }

    def as_memory_track(self) -> dict[str, Any]:
        return {
            "semantic_id": self.track_id,
            "object_id": f"gdino_online_{self.track_id}",
            "category": self.label,
            "centroid_xz": [float(self.position[0]), float(self.position[2])],
            "confidence": _calibrated_grounding_confidence(self.confidence),
            "freshness": 1.0,
            "first_seen_step": self.first_seen_step,
            "last_seen_step": self.last_seen_step,
            "visible_steps": list(self.visible_steps),
            "source": "grounding_dino_online",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.as_interest_dict(),
            "first_seen_step": self.first_seen_step,
            "last_seen_step": self.last_seen_step,
            "visible_steps": list(self.visible_steps),
            "detection_count": self.detection_count,
            "best_score": self.best_score,
            "independent_view_signatures": [list(value) for value in self.independent_views],
        }


class MatrixRotation:
    def __init__(self, matrix: Any) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float32).reshape(3, 3)

    def transform_vector(self, vector: Any) -> np.ndarray:
        return self.matrix @ np.asarray(vector, dtype=np.float32)


class GroundingWorker:
    def __init__(
        self,
        python_path: Path,
        script_path: Path,
        model_id: str,
        labels: str,
        box_threshold: float,
        text_threshold: float,
        max_detections: int,
        cuda_visible_devices: str,
        log_path: Path,
    ) -> None:
        env = dict(os.environ)
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": str(cuda_visible_devices),
            }
        )
        self.log_file = log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(python_path),
                str(script_path),
                "--model-id",
                model_id,
                "--labels",
                labels,
                "--box-threshold",
                str(box_threshold),
                "--text-threshold",
                str(text_threshold),
                "--max-detections",
                str(max_detections),
            ],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_file,
            text=True,
            bufsize=1,
        )
        ready = self._read()
        if ready.get("type") != "ready":
            raise RuntimeError(f"Grounding worker failed to become ready: {ready}")
        self.ready = ready

    def infer(
        self,
        request_id: Any,
        rgb_path: Path,
        *,
        labels: list[str] | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "infer",
            "request_id": request_id,
            "rgb_path": str(rgb_path),
        }
        if labels is not None:
            payload["labels"] = list(labels)
        if box_threshold is not None:
            payload["box_threshold"] = float(box_threshold)
        if text_threshold is not None:
            payload["text_threshold"] = float(text_threshold)
        if max_detections is not None:
            payload["max_detections"] = int(max_detections)
        self._write(payload)
        response = self._read()
        if response.get("type") == "error":
            raise RuntimeError(response.get("error", "Grounding worker error"))
        if response.get("type") != "result" or response.get("request_id") != request_id:
            raise RuntimeError(f"Unexpected Grounding worker response: {response}")
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._write({"type": "shutdown"})
                self._read()
            except Exception:
                self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.log_file.close()

    def _write(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Grounding worker stdin is unavailable")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _read(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("Grounding worker stdout is unavailable")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"Grounding worker exited unexpectedly with code {code}")
        return json.loads(line)


class LingBotMapWorker:
    def __init__(
        self,
        python_path: Path,
        repo_path: Path,
        model_path: Path,
        output_dir: Path,
        num_scale_frames: int,
        camera_num_iterations: int,
        keyframe_interval: int,
        cuda_visible_devices: str,
        log_path: Path,
    ) -> None:
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(cuda_visible_devices),
                "PYTHONUNBUFFERED": "1",
            }
        )
        self.log_file = log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(python_path),
                str(ROOT / "scripts" / "lingbot_map_online_worker.py"),
                "--repo",
                str(repo_path),
                "--model-pt",
                str(model_path),
                "--output-dir",
                str(output_dir),
                "--num-scale-frames",
                str(num_scale_frames),
                "--camera-num-iterations",
                str(camera_num_iterations),
                "--keyframe-interval",
                str(keyframe_interval),
            ],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_file,
            text=True,
            bufsize=1,
        )
        ready = self._read()
        if ready.get("type") != "ready":
            raise RuntimeError(f"LingBot-Map worker failed to become ready: {ready}")
        self.ready = ready

    def bootstrap(self, request_id: int, rgb_paths: list[Path]) -> list[dict[str, Any]]:
        self._write(
            {
                "type": "bootstrap",
                "request_id": request_id,
                "rgb_paths": [str(path) for path in rgb_paths],
            }
        )
        response = self._read_result(request_id, "bootstrap_result")
        return list(response["results"])

    def infer(self, request_id: int, frame_index: int, rgb_path: Path) -> dict[str, Any]:
        self._write(
            {
                "type": "infer",
                "request_id": request_id,
                "frame_index": frame_index,
                "rgb_path": str(rgb_path),
            }
        )
        response = self._read_result(request_id, "result")
        result = dict(response["result"])
        result["is_keyframe"] = bool(response.get("is_keyframe", True))
        return result

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._write({"type": "shutdown"})
                self._read()
            except Exception:
                self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.log_file.close()

    def _read_result(self, request_id: int, expected_type: str) -> dict[str, Any]:
        response = self._read()
        if response.get("type") == "error":
            raise RuntimeError(
                f"LingBot-Map worker error: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        if response.get("type") != expected_type or response.get("request_id") != request_id:
            raise RuntimeError(f"Unexpected LingBot-Map worker response: {response}")
        return response

    def _write(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise RuntimeError("LingBot-Map worker stdin is unavailable")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _read(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("LingBot-Map worker stdout is unavailable")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"LingBot-Map worker exited unexpectedly with code {self.process.poll()}"
            )
        return json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict online Habitat interest-exploration and semantic-memory runner.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-dataset-config")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--graceful-stop-steps", type=int, default=48)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--move-amount", type=float, default=0.25)
    parser.add_argument("--turn-amount", type=float, default=15.0)
    parser.add_argument("--map-size", type=int, default=480)
    parser.add_argument("--bev-resolution", type=float, default=0.05)
    parser.add_argument("--sample-stride", type=int, default=12)
    parser.add_argument("--labels", default="cup,mug,bottle,table,counter,sink")
    parser.add_argument("--model-id", default="downloads/hf_models/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.22)
    parser.add_argument("--max-detections", type=int, default=16)
    parser.add_argument("--track-merge-radius-m", type=float, default=0.30)
    parser.add_argument(
        "--detector-python",
        default=os.getenv("RSCNAV_DETECTOR_PYTHON", sys.executable),
    )
    parser.add_argument("--detector-cuda-visible-devices", default="0")
    parser.add_argument("--cup-min-views", type=int, default=5)
    parser.add_argument("--cup-min-confidence", type=float, default=0.28)
    parser.add_argument(
        "--cup-confirmation-mode",
        choices=("grounding_crop", "off"),
        default="grounding_crop",
    )
    parser.add_argument("--cup-confirmation-min-task-views", type=int, default=2)
    parser.add_argument("--cup-confirmation-min-visual-passes", type=int, default=2)
    parser.add_argument("--cup-confirmation-max-attempts", type=int, default=3)
    parser.add_argument(
        "--cup-confirmation-min-depth-relief-m",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--cup-confirmation-max-position-spread-m",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--cup-verifier-labels",
        default=(
            "real drinking cup,cup,mug,drinking glass,printed picture,poster,"
            "wall outlet,light switch,wall decoration,cabinet handle,bottle"
        ),
    )
    parser.add_argument(
        "--cup-verifier-positive-labels",
        default="real drinking cup,cup,mug,drinking glass",
    )
    parser.add_argument("--cup-verifier-box-threshold", type=float, default=0.16)
    parser.add_argument("--cup-verifier-text-threshold", type=float, default=0.16)
    parser.add_argument("--cup-verifier-min-positive-score", type=float, default=0.30)
    parser.add_argument("--cup-verifier-min-score-margin", type=float, default=0.05)
    parser.add_argument("--cup-verifier-crop-padding-ratio", type=float, default=0.45)
    parser.add_argument("--surface-min-views", type=int, default=3)
    parser.add_argument("--surface-min-confidence", type=float, default=0.24)
    parser.add_argument("--surface-arrival-radius-m", type=float, default=1.15)
    parser.add_argument("--frontier-arrival-radius-m", type=float, default=0.45)
    parser.add_argument("--frontier-min-distance-m", type=float, default=0.45)
    parser.add_argument("--frontier-min-unknown-gain", type=float, default=0.02)
    parser.add_argument(
        "--frontier-strategy",
        choices=("greedy", "hierarchical"),
        default="hierarchical",
    )
    parser.add_argument("--frontier-min-cluster-cells", type=int, default=4)
    parser.add_argument("--frontier-ray-count", type=int, default=61)
    parser.add_argument("--frontier-sensor-range-m", type=float, default=6.0)
    parser.add_argument("--frontier-replan-interval", type=int, default=4)
    parser.add_argument("--frontier-arrival-scan-deg", type=float, default=90.0)
    parser.add_argument("--obstacle-stop-depth-m", type=float, default=0.55)
    parser.add_argument("--frontier-patience", type=int, default=12)
    parser.add_argument("--initial-yaw-steps", type=int, default=24)
    parser.add_argument("--scanned-surface-radius-m", type=float, default=2.0)
    parser.add_argument("--planning-inflation-radius-cells", type=int, default=3)
    parser.add_argument("--path-lookahead-m", type=float, default=0.45)
    parser.add_argument(
        "--execution-planner",
        choices=("hybrid_navmesh", "observed_bev"),
        default="hybrid_navmesh",
    )
    parser.add_argument("--navmesh-max-snap-m", type=float, default=1.0)
    parser.add_argument("--frontier-blacklist-radius-m", type=float, default=0.8)
    parser.add_argument("--stuck-forward-attempts", type=int, default=3)
    parser.add_argument("--coverage-confirmations", type=int, default=2)
    parser.add_argument("--coverage-scan-min-new-cells", type=int, default=180)
    parser.add_argument("--familiarization-min-steps", type=int, default=300)
    parser.add_argument("--familiarization-max-steps", type=int, default=420)
    parser.add_argument("--familiarization-saturation-window", type=int, default=60)
    parser.add_argument("--familiarization-max-new-cells", type=int, default=900)
    parser.add_argument("--familiarization-min-reobserve-ratio", type=float, default=0.55)
    parser.add_argument("--guided-correction-position-xyz", nargs=3, type=float)
    parser.add_argument("--guided-correction-trigger-step", type=int, default=120)
    parser.add_argument("--guided-correction-arrival-radius-m", type=float, default=0.55)
    parser.add_argument("--guided-correction-scan-turns", type=int, default=24)
    parser.add_argument("--task-text", default=TASK_TEXT)
    parser.add_argument(
        "--task-planner-mode",
        choices=("auto", "api", "deterministic"),
        default="auto",
    )
    parser.add_argument(
        "--task-planner-api-base",
        default=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
    parser.add_argument("--task-planner-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument(
        "--task-planner-model",
        default=os.getenv("DASHSCOPE_MODEL", "qwen3-max"),
    )
    parser.add_argument("--task-planner-timeout-s", type=float, default=90.0)
    parser.add_argument("--task-planning-wait-steps", type=int, default=8)
    parser.add_argument("--task-planner-max-candidates", type=int, default=32)
    parser.add_argument("--task-planner-max-support-candidates", type=int, default=6)
    parser.add_argument("--task-planner-support-merge-radius-m", type=float, default=1.25)
    parser.add_argument("--task-dynamic-cup-merge-radius-m", type=float, default=0.75)
    parser.add_argument(
        "--scene-vlm-mode",
        choices=("auto", "api", "deterministic"),
        default="auto",
    )
    parser.add_argument(
        "--scene-vlm-api-base",
        default=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
    parser.add_argument("--scene-vlm-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument(
        "--scene-vlm-model",
        default=os.getenv("DASHSCOPE_VLM_MODEL", "qwen3-vl-plus"),
    )
    parser.add_argument("--scene-vlm-timeout-s", type=float, default=120.0)
    parser.add_argument("--scene-vlm-max-images", type=int, default=8)
    parser.add_argument("--start-position-xyz", nargs=3, type=float)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--geometry-source",
        choices=("habitat_rgbd", "lingbot_depth_exact_pose"),
        default="habitat_rgbd",
    )
    parser.add_argument(
        "--lingbot-python",
        default=os.getenv("RSCNAV_LINGBOT_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--lingbot-repo",
        default=os.getenv(
            "RSCNAV_LINGBOT_REPO",
            str(ROOT / "third_party" / "lingbot-map"),
        ),
    )
    parser.add_argument(
        "--lingbot-model-pt",
        default=os.getenv(
            "RSCNAV_LINGBOT_MODEL",
            str(ROOT / "models" / "lingbot-map-long.pt"),
        ),
    )
    parser.add_argument("--lingbot-cuda-visible-devices", default="1")
    parser.add_argument("--lingbot-scale-frames", type=int, default=8)
    parser.add_argument("--lingbot-camera-num-iterations", type=int, default=4)
    parser.add_argument("--lingbot-keyframe-interval", type=int, default=1)
    parser.add_argument("--lingbot-depth-conf-threshold", type=float, default=1.2)
    args = parser.parse_args()

    ensure_conda_nvidia_egl_vendor()
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    overlays_dir = out_dir / "overlays"
    bev_frames_dir = out_dir / "bev_frames"
    checkpoints_dir = out_dir / "checkpoints"
    planner_dir = out_dir / "task_planner"
    scene_vlm_dir = planner_dir / "scene_vlm"
    confirmation_crops_dir = out_dir / "confirmation_crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    bev_frames_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    planner_dir.mkdir(parents=True, exist_ok=True)
    scene_vlm_dir.mkdir(parents=True, exist_ok=True)
    confirmation_crops_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "online_trace.jsonl"
    trace_file = trace_path.open("w", encoding="utf-8")
    search_belief_path = planner_dir / "search_belief.jsonl"
    search_belief_file = search_belief_path.open("w", encoding="utf-8")

    session = HabitatControlSession(
        scene=Path(args.scene).expanduser().resolve(),
        scene_dataset_config=Path(args.scene_dataset_config).expanduser().resolve() if args.scene_dataset_config else None,
        resolution=int(args.resolution),
        move_amount=float(args.move_amount),
        turn_amount=float(args.turn_amount),
        enable_oracle_metrics=False,
        enable_semantic_sensor=False,
        enable_autopilot_start_path=False,
    )
    if args.start_position_xyz is not None:
        _set_agent_start_pose(
            session,
            position_xyz=np.asarray(args.start_position_xyz, dtype=np.float32),
            yaw_deg=float(args.start_yaw_deg),
        )
    initial_state = session.sim.get_agent(0).get_state()
    initial_position = np.asarray(initial_state.position, dtype=np.float32)
    map_config = DenseBEVConfig(
        grid_size=(int(args.map_size), int(args.map_size)),
        resolution=float(args.bev_resolution),
        sample_stride=int(args.sample_stride),
        obstacle_dilation_radius_cells=1,
    )
    map_half_span = 0.5 * map_config.grid_size[0] * map_config.resolution
    mapper = DenseBEVMapper(
        origin_world_xz=(
            float(initial_position[0] - map_half_span),
            float(initial_position[2] - map_half_span),
        ),
        config=map_config,
    )
    memory = ObjectMemoryStore(scene_id=str(Path(args.scene).expanduser().resolve()))
    worker = GroundingWorker(
        python_path=Path(args.detector_python).expanduser().resolve(),
        script_path=ROOT / "scripts" / "groundingdino_online_worker.py",
        model_id=args.model_id,
        labels=args.labels,
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
        max_detections=int(args.max_detections),
        cuda_visible_devices=args.detector_cuda_visible_devices,
        log_path=out_dir / "grounding_worker.log",
    )
    lingbot_worker = (
        LingBotMapWorker(
            python_path=Path(args.lingbot_python).expanduser().resolve(),
            repo_path=Path(args.lingbot_repo).expanduser().resolve(),
            model_path=Path(args.lingbot_model_pt).expanduser().resolve(),
            output_dir=out_dir / "lingbot_geometry",
            num_scale_frames=int(args.lingbot_scale_frames),
            camera_num_iterations=int(args.lingbot_camera_num_iterations),
            keyframe_interval=int(args.lingbot_keyframe_interval),
            cuda_visible_devices=str(args.lingbot_cuda_visible_devices),
            log_path=out_dir / "lingbot_worker.log",
        )
        if args.geometry_source == "lingbot_depth_exact_pose"
        else None
    )

    tracks: list[OnlineTrack] = []
    all_detections: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    scanned_surface_ids: set[int] = set()
    scanned_surface_positions: list[np.ndarray] = []
    failed_surface_ids: set[int] = set()
    focused_cup_track_ids: set[int] = set()
    inspected_cup_track_ids: set[int] = set()
    cup_confirmation_observations: dict[int, list[dict[str, Any]]] = {}
    cup_confirmation_attempts: dict[int, int] = {}
    cup_confirmation_terminal_statuses: dict[int, str] = {}
    cup_scan_in_progress_id: int | None = None
    cup_confirmation_config = CupConfirmationConfig(
        min_task_views=max(1, int(args.cup_confirmation_min_task_views)),
        min_visual_passes=(
            0
            if str(args.cup_confirmation_mode) == "off"
            else max(1, int(args.cup_confirmation_min_visual_passes))
        ),
        min_visual_negatives=(
            0
            if str(args.cup_confirmation_mode) == "off"
            else max(1, int(args.cup_confirmation_min_visual_passes))
        ),
        min_depth_relief_passes=max(
            1,
            int(args.cup_confirmation_min_task_views),
        ),
        min_depth_relief_m=max(
            0.0,
            float(args.cup_confirmation_min_depth_relief_m),
        ),
        max_position_spread_m=max(
            0.01,
            float(args.cup_confirmation_max_position_spread_m),
        ),
    )
    cup_verifier_labels = _comma_separated_labels(args.cup_verifier_labels)
    cup_verifier_positive_labels = set(
        _comma_separated_labels(args.cup_verifier_positive_labels)
    )
    scan_queue: list[str] = []
    active_surface_id: int | None = None
    frontier_target_cell: tuple[int, int] | None = None
    frontier_target_view_yaw_deg: float | None = None
    blacklisted_frontiers: list[tuple[int, int]] = []
    recovery_queue: list[str] = []
    coverage_scan_queue: list[str] = []
    coverage_scan_start_cells = 0
    coverage_scan_pending_evaluation = False
    coverage_confirmations = 0
    exploration_phase = "deep_familiarization"
    explored_cell_history: list[int] = []
    familiarization_complete_step: int | None = None
    familiarization_snapshot: dict[str, Any] | None = None
    familiarity_metrics: dict[str, Any] = {}
    task_text = str(args.task_text)
    task_injection_step: int | None = None
    task_planning_complete_step: int | None = None
    task_planning_remaining_steps = 0
    task_planner_request: dict[str, Any] | None = None
    task_planner_output: dict[str, Any] | None = None
    task_planner_metadata: dict[str, Any] | None = None
    task_candidate_order: list[str] = []
    task_planner_seed_order: list[str] = []
    task_search_ranking: list[dict[str, Any]] = []
    task_search_beliefs: dict[str, dict[str, Any]] = {}
    search_belief_revision = 0
    scene_understanding: dict[str, Any] | None = None
    scene_vlm_metadata: dict[str, Any] | None = None
    task_plan_events: list[dict[str, Any]] = []
    search_evidence_updates: list[dict[str, Any]] = []
    surface_scan_in_progress: dict[str, Any] | None = None
    checkpoint_artifacts: dict[str, dict[str, Any]] = {}
    guided_correction_status = (
        "pending" if args.guided_correction_position_xyz is not None else "disabled"
    )
    guided_correction_target = (
        np.asarray(args.guided_correction_position_xyz, dtype=np.float32)
        if args.guided_correction_position_xyz is not None
        else None
    )
    guided_correction_scan_queue: list[str] = []
    guided_correction_start_step: int | None = None
    guided_correction_complete_step: int | None = None
    guided_correction_start_explored_cells: int | None = None
    guided_correction_result: dict[str, Any] | None = None
    no_frontier_steps = 0
    consecutive_collisions = 0
    no_progress_forward_attempts = 0
    navigation_target_key: str | None = None
    navigation_pose_history: list[np.ndarray] = []
    previous_pose: tuple[np.ndarray, float] | None = None
    lingbot_depth_scale: float | None = None
    cached_ranked_frontiers: list[dict[str, Any]] = []
    stop_reason = "max_steps"

    try:
        for step in range(int(args.max_steps) + max(0, int(args.graceful_stop_steps))):
            loop_started = time.perf_counter()
            frame, rgb, habitat_depth = _capture_rgbd_frame(session, frames_dir, step)
            observation_ready = time.perf_counter()
            frames.append(frame)
            depth = habitat_depth
            geometry_ready = True
            lingbot_result: dict[str, Any] | None = None
            if lingbot_worker is None:
                mapper.update_from_depth(
                    depth=depth,
                    agent_position_xyz=np.asarray(frame["agent_position_xyz"], dtype=np.float32),
                    sensor_position_xyz=np.asarray(frame["sensor_position_xyz"], dtype=np.float32),
                    sensor_rotation=MatrixRotation(frame["sensor_rotation_matrix"]),
                    hfov_deg=90.0,
                )
            elif step + 1 < int(args.lingbot_scale_frames):
                geometry_ready = False
            elif step + 1 == int(args.lingbot_scale_frames):
                bootstrap_results = lingbot_worker.bootstrap(
                    request_id=step,
                    rgb_paths=[Path(item["rgb_path"]) for item in frames],
                )
                lingbot_depth_scale = _calibrate_lingbot_depth_scale(
                    bootstrap_results,
                    frames,
                    confidence_threshold=float(args.lingbot_depth_conf_threshold),
                )
                mapper = DenseBEVMapper(
                    origin_world_xz=(
                        float(initial_position[0] - map_half_span),
                        float(initial_position[2] - map_half_span),
                    ),
                    config=map_config,
                )
                for bootstrap_result, bootstrap_frame in zip(bootstrap_results, frames):
                    bootstrap_depth = _load_lingbot_depth(
                        bootstrap_result,
                        target_shape=habitat_depth.shape,
                        scale=lingbot_depth_scale,
                        confidence_threshold=float(args.lingbot_depth_conf_threshold),
                    )
                    mapper.update_from_depth(
                        depth=bootstrap_depth,
                        agent_position_xyz=np.asarray(
                            bootstrap_frame["agent_position_xyz"], dtype=np.float32
                        ),
                        sensor_position_xyz=np.asarray(
                            bootstrap_frame["sensor_position_xyz"], dtype=np.float32
                        ),
                        sensor_rotation=MatrixRotation(
                            bootstrap_frame["sensor_rotation_matrix"]
                        ),
                        hfov_deg=90.0,
                    )
                    bootstrap_frame["lingbot_depth_npy"] = bootstrap_result["depth_npy"]
                    bootstrap_frame["lingbot_depth_conf_npy"] = bootstrap_result[
                        "depth_conf_npy"
                    ]
                    bootstrap_frame["geometry_source"] = str(args.geometry_source)
                    bootstrap_depth_png = (
                        frames_dir
                        / f"frame_{int(bootstrap_frame['frame_index']):04d}_lingbot_depth.png"
                    )
                    _depth_image(bootstrap_depth).save(bootstrap_depth_png)
                    bootstrap_frame["depth_png"] = str(bootstrap_depth_png)
                lingbot_result = bootstrap_results[-1]
                depth = _load_lingbot_depth(
                    lingbot_result,
                    target_shape=habitat_depth.shape,
                    scale=lingbot_depth_scale,
                    confidence_threshold=float(args.lingbot_depth_conf_threshold),
                )
            else:
                lingbot_result = lingbot_worker.infer(
                    request_id=step,
                    frame_index=step,
                    rgb_path=Path(frame["rgb_path"]),
                )
                depth = _load_lingbot_depth(
                    lingbot_result,
                    target_shape=habitat_depth.shape,
                    scale=float(lingbot_depth_scale),
                    confidence_threshold=float(args.lingbot_depth_conf_threshold),
                )
                mapper.update_from_depth(
                    depth=depth,
                    agent_position_xyz=np.asarray(frame["agent_position_xyz"], dtype=np.float32),
                    sensor_position_xyz=np.asarray(frame["sensor_position_xyz"], dtype=np.float32),
                    sensor_rotation=MatrixRotation(frame["sensor_rotation_matrix"]),
                    hfov_deg=90.0,
                )
            frame["geometry_source"] = str(args.geometry_source)
            frame["geometry_ready"] = bool(geometry_ready)
            frame["lingbot_depth_scale"] = lingbot_depth_scale
            if lingbot_result is not None:
                frame["lingbot_depth_npy"] = lingbot_result["depth_npy"]
                frame["lingbot_depth_conf_npy"] = lingbot_result["depth_conf_npy"]
                lingbot_depth_png = frames_dir / f"frame_{step:04d}_lingbot_depth.png"
                _depth_image(depth).save(lingbot_depth_png)
                frame["depth_png"] = str(lingbot_depth_png)
            mapping_ready = time.perf_counter()

            grounding_response = worker.infer(step, Path(frame["rgb_path"]))
            grounding_ready = time.perf_counter()
            projected = []
            for detection in (
                grounding_response.get("detections", []) if geometry_ready else []
            ):
                projection = _project_box_detection(
                    detection,
                    depth,
                    frame,
                    hfov_deg=90.0,
                    depth_min_m=0.05,
                    depth_max_m=6.0,
                )
                if projection is None:
                    continue
                projected.append(
                    {
                        **detection,
                        **projection,
                        "frame_index": step,
                        "rgb_path": frame["rgb_path"],
                        "online": True,
                    }
                )
            _write_overlay(rgb.copy(), projected, overlays_dir / f"frame_{step:04d}_overlay.jpg")
            camera_signature = (
                float(frame["sensor_position_xyz"][0]),
                float(frame["sensor_position_xyz"][2]),
                _yaw_from_matrix(frame["sensor_rotation_matrix"]),
            )
            positive_track_ids = _merge_online_detections(
                tracks,
                projected,
                step=step,
                camera_signature=camera_signature,
                merge_radius_m=float(args.track_merge_radius_m),
            )
            all_detections.extend(projected)
            confirmation_verification_ms = 0.0

            current_position = np.asarray(frame["agent_position_xyz"], dtype=np.float32)
            current_yaw = _yaw_from_matrix(frame["agent_rotation_matrix"])
            if previous_pose is None:
                evidence_weight = 0.08
            else:
                translation = float(np.linalg.norm(current_position - previous_pose[0]))
                rotation = _angle_delta_deg(current_yaw, previous_pose[1])
                evidence_weight = smooth_motion_evidence_weight(
                    translation,
                    rotation,
                    nominal_translation_m=float(args.move_amount),
                    nominal_rotation_deg=float(args.turn_amount),
                )
            previous_pose = (current_position.copy(), current_yaw)
            observability = _observability_for_memory(
                memory,
                tracks=tracks,
                positive_track_ids=positive_track_ids,
                frame=frame,
                depth=depth,
                hfov_deg=90.0,
            )
            observed_memory_tracks = [
                track.as_memory_track()
                for track in tracks
                if track.track_id in positive_track_ids
            ]
            memory_update = memory.update_from_tracks(
                observed_memory_tracks,
                current_step=step,
                observability=observability,
                evidence_weight=evidence_weight,
                negative_evidence_weight=evidence_weight,
            )
            memory_ready = time.perf_counter()
            if (
                surface_scan_in_progress is not None
                and scan_queue
                and geometry_ready
                and int(np.isfinite(depth).sum()) >= int(depth.size * 0.25)
            ):
                surface_scan_in_progress["observable_frames"] = (
                    int(
                        surface_scan_in_progress.get(
                            "observable_frames",
                            0,
                        )
                    )
                    + 1
                )
            if (
                surface_scan_in_progress is not None
                and not scan_queue
            ):
                support_position_xz = np.asarray(
                    surface_scan_in_progress["position_xz"],
                    dtype=np.float32,
                )
                scan_start_step = int(surface_scan_in_progress["start_step"])
                observed_target_ids = sorted(
                    {
                        track.track_id
                        for track in tracks
                        if track.label == "cup"
                        and any(
                            int(visible_step) >= scan_start_step
                            for visible_step in track.visible_steps
                        )
                        and float(
                            np.linalg.norm(
                                track.position[[0, 2]]
                                - support_position_xz
                            )
                        )
                        <= 2.0
                    }
                )
                evidence_update = {
                    "step": step,
                    "event": "support_surface_inspection_completed",
                    "candidate_id": surface_scan_in_progress[
                        "candidate_id"
                    ],
                    "start_step": scan_start_step,
                    "observed_target_candidate_ids": [
                        f"track_{track_id}"
                        for track_id in observed_target_ids
                    ],
                    "outcome": (
                        "target_evidence_observed"
                        if observed_target_ids
                        else "no_target_evidence_observed"
                    ),
                }
                observable_scan = (
                    int(
                        surface_scan_in_progress.get(
                            "observable_frames",
                            0,
                        )
                    )
                    >= 4
                )
                evidence_update["observable_scan"] = observable_scan
                belief_update = apply_search_evidence(
                    task_search_beliefs,
                    candidate_id=str(evidence_update["candidate_id"]),
                    event_id=(
                        f"surface_scan:{evidence_update['candidate_id']}:"
                        f"{scan_start_step}"
                    ),
                    outcome=str(evidence_update["outcome"]),
                    step=step,
                    observable=observable_scan,
                )
                evidence_update["belief_update"] = belief_update
                scanned_id = int(surface_scan_in_progress["track_id"])
                scanned_surface_ids.add(scanned_id)
                scanned_surface_positions.append(support_position_xz.copy())
                search_evidence_updates.append(evidence_update)
                task_plan_events.append(evidence_update)
                search_belief_revision += 1
                _write_jsonl_event(
                    search_belief_file,
                    {
                        "revision": search_belief_revision,
                        **evidence_update,
                        "beliefs": task_search_beliefs,
                    },
                )
                surface_scan_in_progress = None

            current_cell = mapper.world_to_grid((float(current_position[0]), float(current_position[2])))
            safe_free = planning_free_mask(
                mapper.free_mask(),
                mapper.occupied_mask(),
                inflation_radius_cells=int(args.planning_inflation_radius_cells),
            )
            frontier_config = InterestConfig(
                min_unknown_gain=float(args.frontier_min_unknown_gain),
                min_frontier_distance_m=float(args.frontier_min_distance_m),
                min_frontier_cluster_cells=int(args.frontier_min_cluster_cells),
                ray_count=int(args.frontier_ray_count),
                sensor_range_m=float(args.frontier_sensor_range_m),
            )
            replan_frontiers = (
                current_cell is not None
                and (
                    not cached_ranked_frontiers
                    or frontier_target_cell is None
                    or step % max(1, int(args.frontier_replan_interval)) == 0
                )
            )
            if current_cell is None:
                cached_ranked_frontiers = []
            elif replan_frontiers and args.frontier_strategy == "hierarchical":
                cached_ranked_frontiers = rank_frontier_clusters(
                    mapper.explored,
                    safe_free,
                    mapper.occupied_mask(),
                    mapper.observation_count,
                    current_cell=current_cell,
                    resolution=mapper.config.resolution,
                    config=frontier_config,
                )
            elif replan_frontiers:
                cached_ranked_frontiers = rank_frontiers(
                    mapper.explored,
                    safe_free,
                    mapper.observation_count,
                    current_cell=current_cell,
                    resolution=mapper.config.resolution,
                    config=frontier_config,
                )
            ranked_frontiers = list(cached_ranked_frontiers)
            ranked_frontiers = _without_blacklisted_frontiers(
                ranked_frontiers,
                blacklisted_frontiers,
                radius_cells=max(
                    1,
                    int(round(float(args.frontier_blacklist_radius_m) / mapper.config.resolution)),
                ),
            )
            if frontier_target_cell is not None and current_cell is not None:
                frontier_distance_m = float(
                    np.linalg.norm(
                        np.asarray(frontier_target_cell, dtype=np.float32)
                        - np.asarray(current_cell, dtype=np.float32)
                    )
                    * mapper.config.resolution
                )
                if frontier_distance_m <= float(args.frontier_arrival_radius_m):
                    frontier_target_cell = None
                    coverage_scan_queue.extend(
                        _frontier_observation_actions(
                            current_yaw=current_yaw,
                            target_yaw=frontier_target_view_yaw_deg,
                            turn_amount_deg=float(args.turn_amount),
                            scan_deg=float(args.frontier_arrival_scan_deg),
                        )
                    )
                    frontier_target_view_yaw_deg = None
            surface_interest_tracks = [
                track.as_interest_dict()
                for track in tracks
                if track.label in SURFACE_LABELS and track.track_id not in failed_surface_ids
                and not _near_any_scanned_surface(
                    track.position[[0, 2]],
                    scanned_surface_positions,
                    radius_m=float(args.scanned_surface_radius_m),
                )
            ]
            explored_cell_history.append(int(mapper.explored.sum()))
            familiarization_ready, familiarity_metrics = deep_familiarization_status(
                step=step,
                explored_cell_history=explored_cell_history,
                free=safe_free,
                observation_count=mapper.observation_count,
                min_steps=int(args.familiarization_min_steps),
                max_steps=int(args.familiarization_max_steps),
                saturation_window=int(args.familiarization_saturation_window),
                max_new_cells=int(args.familiarization_max_new_cells),
                min_reobserve_ratio=float(args.familiarization_min_reobserve_ratio),
            )

            if coverage_scan_pending_evaluation and not coverage_scan_queue:
                new_cells = int(mapper.explored.sum()) - int(coverage_scan_start_cells)
                if not ranked_frontiers and new_cells < int(args.coverage_scan_min_new_cells):
                    coverage_confirmations += 1
                else:
                    coverage_confirmations = 0
                coverage_scan_pending_evaluation = False

            if (
                exploration_phase == "deep_familiarization"
                and familiarization_ready
                and not scan_queue
                and not coverage_scan_queue
                and not recovery_queue
                and guided_correction_status in {"disabled", "completed"}
            ):
                familiarization_complete_step = step
                familiarization_snapshot = {
                    **familiarity_metrics,
                    "explored_cells": int(mapper.explored.sum()),
                    "num_tracks": len(tracks),
                    "candidate_cup_track_ids": [
                        track.track_id
                        for track in _confirmed_cups(
                            tracks,
                            min_views=int(args.cup_min_views),
                            min_confidence=float(args.cup_min_confidence),
                        )
                    ],
                }
                task_injection_step = step
                stable_task_tracks = _stable_task_tracks(
                    tracks,
                    cup_min_views=int(args.cup_min_views),
                    cup_min_confidence=float(args.cup_min_confidence),
                    surface_min_views=int(args.surface_min_views),
                    surface_min_confidence=float(args.surface_min_confidence),
                )
                task_planner_request = build_online_planner_request(
                    task_text=task_text,
                    current_xz=(
                        float(current_position[0]),
                        float(current_position[2]),
                    ),
                    tracks=[track.as_interest_dict() for track in stable_task_tracks],
                    memory_items=memory.to_dict().get("items", []),
                    max_candidates=int(args.task_planner_max_candidates),
                    max_support_candidates=int(
                        args.task_planner_max_support_candidates
                    ),
                    support_merge_radius_m=float(
                        args.task_planner_support_merge_radius_m
                    ),
                )
                scene_keyframes = _prepare_scene_vlm_keyframes(
                    frames=frames,
                    detections=all_detections,
                    candidate_landmarks=task_planner_request[
                        "candidate_landmarks"
                    ],
                    out_dir=scene_vlm_dir / "keyframes",
                    max_images=int(args.scene_vlm_max_images),
                )
                scene_understanding, scene_vlm_metadata = (
                    understand_scene_with_vlm(
                        task_text=task_text,
                        candidates=task_planner_request[
                            "candidate_landmarks"
                        ],
                        keyframes=scene_keyframes,
                        mode=str(args.scene_vlm_mode),
                        api_base=str(args.scene_vlm_api_base),
                        api_key=os.getenv(
                            str(args.scene_vlm_api_key_env),
                            "",
                        ),
                        model=str(args.scene_vlm_model),
                        timeout_s=float(args.scene_vlm_timeout_s),
                    )
                )
                _write_json(
                    scene_vlm_dir / "scene_understanding.json",
                    scene_understanding,
                )
                (scene_vlm_dir / "scene_vlm_prompt.txt").write_text(
                    str(scene_vlm_metadata.get("prompt", "")),
                    encoding="utf-8",
                )
                if scene_vlm_metadata.get("raw_response") is not None:
                    _write_json(
                        scene_vlm_dir / "scene_vlm_raw_response.json",
                        scene_vlm_metadata["raw_response"],
                    )
                _write_json(
                    scene_vlm_dir / "scene_vlm_metadata.json",
                    {
                        key: value
                        for key, value in scene_vlm_metadata.items()
                        if key not in {"prompt", "raw_response"}
                    },
                )
                task_planner_request = enrich_planner_request(
                    task_planner_request,
                    scene_understanding,
                )
                task_search_beliefs = initialize_search_beliefs(
                    task_planner_request["candidate_landmarks"],
                    scene_understanding=scene_understanding,
                    step=step,
                )
                search_belief_revision += 1
                _write_jsonl_event(
                    search_belief_file,
                    {
                        "revision": search_belief_revision,
                        "step": step,
                        "event": "search_beliefs_initialized",
                        "beliefs": task_search_beliefs,
                    },
                )
                _write_json(planner_dir / "planner_request.json", task_planner_request)
                task_planner_output, task_planner_metadata = plan_online_task(
                    request_payload=task_planner_request,
                    mode=str(args.task_planner_mode),
                    api_base=str(args.task_planner_api_base),
                    api_key=os.getenv(str(args.task_planner_api_key_env), ""),
                    model=str(args.task_planner_model),
                    timeout_s=float(args.task_planner_timeout_s),
                )
                task_candidate_order = list(
                    task_planner_output.get("ordered_candidate_ids", [])
                )
                task_planner_seed_order = list(task_candidate_order)
                _write_json(planner_dir / "planner_output.json", task_planner_output)
                (planner_dir / "planner_prompt.txt").write_text(
                    str(task_planner_metadata.get("prompt", "")),
                    encoding="utf-8",
                )
                if task_planner_metadata.get("raw_response") is not None:
                    _write_json(
                        planner_dir / "api_raw_response.json",
                        task_planner_metadata["raw_response"],
                    )
                _write_json(
                    planner_dir / "planner_metadata.json",
                    {
                        key: value
                        for key, value in task_planner_metadata.items()
                        if key not in {"prompt", "raw_response"}
                    },
                )
                task_plan_events.append(
                    {
                        "step": step,
                        "event": "task_injected_and_planned",
                        "task_text": task_text,
                        "mode_used": task_planner_metadata.get("mode_used"),
                        "model": task_planner_metadata.get("model"),
                        "scene_vlm_mode_used": scene_vlm_metadata.get(
                            "mode_used"
                        ),
                        "scene_vlm_model": scene_vlm_metadata.get("model"),
                        "scene_summary": scene_understanding.get(
                            "scene_summary"
                        ),
                        "candidate_ids": list(task_candidate_order),
                    }
                )
                checkpoint_artifacts["task_start"] = _save_online_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    name="task_start",
                    step=step,
                    phase="task_planning",
                    mapper=mapper,
                    memory=memory,
                    tracks=tracks,
                    current_position=current_position,
                    current_yaw=current_yaw,
                )
                exploration_phase = "task_planning"
                task_planning_remaining_steps = max(
                    1,
                    int(args.task_planning_wait_steps),
                )
                frontier_target_cell = None
                frontier_target_view_yaw_deg = None
                cached_ranked_frontiers = []

            if (
                exploration_phase == "task_planning"
                and task_planning_remaining_steps <= 0
            ):
                exploration_phase = "task_execution"
                task_planning_complete_step = step

            semantic_target = None
            if exploration_phase == "task_execution":
                order_before_update = list(task_candidate_order)
                new_candidate_ids = _append_new_task_candidates(
                    tracks,
                    task_candidate_order,
                    inspected_cup_track_ids=inspected_cup_track_ids,
                    scanned_surface_ids=scanned_surface_ids,
                    failed_surface_ids=failed_surface_ids,
                    cup_min_views=int(args.cup_min_views),
                    cup_min_confidence=float(args.cup_min_confidence),
                    surface_min_views=int(args.surface_min_views),
                    surface_min_confidence=float(args.surface_min_confidence),
                    dynamic_cup_merge_radius_m=float(
                        args.task_dynamic_cup_merge_radius_m
                    ),
                )
                if new_candidate_ids:
                    task_plan_events.append(
                        {
                            "step": step,
                            "event": "online_memory_candidates_appended",
                            "candidate_ids": new_candidate_ids,
                        }
                    )
                live_task_candidates = _live_task_candidates(
                    seed_candidates=(
                        task_planner_request.get(
                            "candidate_landmarks",
                            [],
                        )
                        if task_planner_request
                        else []
                    ),
                    tracks=tracks,
                    memory_items=memory.to_dict().get("items", []),
                    candidate_ids=task_candidate_order,
                    current_xz=(
                        float(current_position[0]),
                        float(current_position[2]),
                    ),
                )
                missing_belief_candidates = [
                    candidate
                    for candidate in live_task_candidates
                    if candidate["id"] not in task_search_beliefs
                ]
                if missing_belief_candidates:
                    task_search_beliefs.update(
                        initialize_search_beliefs(
                            missing_belief_candidates,
                            scene_understanding=scene_understanding,
                            step=step,
                        )
                    )
                    search_belief_revision += 1
                    _write_jsonl_event(
                        search_belief_file,
                        {
                            "revision": search_belief_revision,
                            "step": step,
                            "event": "online_candidates_initialized",
                            "candidate_ids": [
                                candidate["id"]
                                for candidate in missing_belief_candidates
                            ],
                            "beliefs": task_search_beliefs,
                        },
                    )
                task_search_ranking = rank_search_candidates(
                    candidates=live_task_candidates,
                    current_xz=(
                        float(current_position[0]),
                        float(current_position[2]),
                    ),
                    scene_understanding=scene_understanding,
                    planner_order=task_planner_seed_order,
                    completed_ids=(
                        {
                            f"track_{track_id}"
                            for track_id in inspected_cup_track_ids
                        }
                        | {
                            f"track_{track_id}"
                            for track_id in scanned_surface_ids
                        }
                    ),
                    failed_ids={
                        f"track_{track_id}"
                        for track_id in failed_surface_ids
                    },
                    attempts={
                        f"track_{track_id}": attempt_count
                        for track_id, attempt_count
                        in cup_confirmation_attempts.items()
                    },
                    active_candidate_id=(
                        f"track_{int(active_surface_id)}"
                        if active_surface_id is not None
                        else None
                    ),
                    beliefs=task_search_beliefs,
                )
                task_candidate_order = [
                    item["candidate_id"]
                    for item in task_search_ranking
                ]
                if task_candidate_order != order_before_update:
                    replan_event = {
                        "step": step,
                        "event": "search_priority_replanned",
                        "previous_candidate_ids": order_before_update,
                        "candidate_ids": list(task_candidate_order),
                        "top_candidates": task_search_ranking[:5],
                        "trigger": (
                            "new_online_target_evidence"
                            if new_candidate_ids
                            else "pose_or_evidence_update"
                        ),
                    }
                    task_plan_events.append(replan_event)
                    search_belief_revision += 1
                    _write_jsonl_event(
                        search_belief_file,
                        {
                            "revision": search_belief_revision,
                            **replan_event,
                            "beliefs": task_search_beliefs,
                        },
                    )
                semantic_target = _choose_planned_task_target(
                    tracks=tracks,
                    ordered_candidate_ids=task_candidate_order,
                    inspected_cup_track_ids=inspected_cup_track_ids,
                    scanned_surface_ids=scanned_surface_ids,
                    failed_surface_ids=failed_surface_ids,
                    cup_min_views=int(args.cup_min_views),
                    cup_min_confidence=float(args.cup_min_confidence),
                    surface_min_views=int(args.surface_min_views),
                    surface_min_confidence=float(args.surface_min_confidence),
                )
                if semantic_target is None:
                    ranked_frontiers = _rerank_frontiers_for_task(
                        ranked_frontiers=ranked_frontiers,
                        mapper=mapper,
                        candidate_landmarks=(
                            task_planner_request.get(
                                "candidate_landmarks",
                                [],
                            )
                            if task_planner_request
                            else []
                        ),
                        scene_understanding=scene_understanding,
                        excluded_candidate_ids={
                            f"track_{track_id}"
                            for track_id in scanned_surface_ids
                        }
                        | {
                            f"track_{track_id}"
                            for track_id in failed_surface_ids
                        },
                    )

            if guided_correction_status == "scanning" and not guided_correction_scan_queue:
                guided_correction_status = "completed"
                guided_correction_complete_step = step
                guided_correction_result = {
                    "status": guided_correction_status,
                    "target_position_xyz": [
                        float(value) for value in guided_correction_target
                    ],
                    "start_step": guided_correction_start_step,
                    "complete_step": guided_correction_complete_step,
                    "explored_cells_before": guided_correction_start_explored_cells,
                    "explored_cells_after": int(mapper.explored.sum()),
                    "explored_cell_gain": (
                        int(mapper.explored.sum()) - int(guided_correction_start_explored_cells)
                        if guided_correction_start_explored_cells is not None
                        else None
                    ),
                }
                checkpoint_artifacts[
                    "after_guidance"
                ] = _save_online_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    name="after_guidance",
                    step=step,
                    phase="deep_familiarization",
                    mapper=mapper,
                    memory=memory,
                    tracks=tracks,
                    current_position=current_position,
                    current_yaw=current_yaw,
                )
            if (
                guided_correction_status == "pending"
                and exploration_phase == "deep_familiarization"
                and step >= int(args.guided_correction_trigger_step)
            ):
                checkpoint_artifacts[
                    "autonomous_before_guidance"
                ] = _save_online_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    name="autonomous_before_guidance",
                    step=step,
                    phase="deep_familiarization",
                    mapper=mapper,
                    memory=memory,
                    tracks=tracks,
                    current_position=current_position,
                    current_yaw=current_yaw,
                )
                guided_correction_status = "navigating"
                guided_correction_start_step = step
                guided_correction_start_explored_cells = int(mapper.explored.sum())
            if guided_correction_status == "navigating":
                guided_distance_m = float(
                    np.linalg.norm(
                        current_position[[0, 2]]
                        - guided_correction_target[[0, 2]]
                    )
                )
                if guided_distance_m <= float(args.guided_correction_arrival_radius_m):
                    guided_correction_status = "scanning"
                    scan_turns = max(4, int(args.guided_correction_scan_turns))
                    guided_correction_scan_queue.extend(
                        ["turn_left"] * scan_turns
                        + ["look_down"]
                        + ["turn_left"] * scan_turns
                        + ["look_up"]
                    )

            if step < int(args.initial_yaw_steps):
                action = "turn_left"
                decision = {
                    "mode": "initial_panorama_scan",
                    "active_surface_id": None,
                    "frontier_target_cell": None,
                    "remaining_scan_steps": int(args.initial_yaw_steps) - step - 1,
                }
            elif cup_scan_in_progress_id is not None and not scan_queue:
                action = "wait"
                decision = {
                    "mode": "cup_confirmation_finalize",
                    "active_surface_id": int(cup_scan_in_progress_id),
                    "frontier_target_cell": frontier_target_cell,
                    "target_kind": "cup",
                }
            elif recovery_queue:
                action = recovery_queue.pop(0)
                decision = {
                    "mode": "stuck_recovery",
                    "active_surface_id": None,
                    "frontier_target_cell": None,
                }
            elif guided_correction_scan_queue:
                action = guided_correction_scan_queue.pop(0)
                decision = {
                    "mode": "guided_correction_scan",
                    "active_surface_id": None,
                    "frontier_target_cell": None,
                    "target_world_xz": [
                        float(guided_correction_target[0]),
                        float(guided_correction_target[2]),
                    ],
                    "remaining_scan_steps": len(guided_correction_scan_queue),
                }
            elif guided_correction_status == "navigating":
                action, decision = _choose_guided_correction_action(
                    depth=depth,
                    current_position=current_position,
                    current_yaw=current_yaw,
                    target_position_xyz=guided_correction_target,
                    obstacle_stop_depth_m=float(args.obstacle_stop_depth_m),
                    turn_amount_deg=float(args.turn_amount),
                    path_lookahead_m=float(args.path_lookahead_m),
                    pathfinder=session.sim.pathfinder,
                    navmesh_max_snap_m=float(args.navmesh_max_snap_m),
                )
            elif exploration_phase == "task_planning":
                task_planning_remaining_steps = max(
                    0,
                    task_planning_remaining_steps - 1,
                )
                action = "wait"
                decision = {
                    "mode": "task_injection_and_api_planning",
                    "active_surface_id": None,
                    "frontier_target_cell": None,
                    "remaining_wait_steps": task_planning_remaining_steps,
                    "task_text": task_text,
                    "task_planner_mode": (
                        task_planner_metadata.get("mode_used")
                        if task_planner_metadata
                        else None
                    ),
                    "task_planner_model": (
                        task_planner_metadata.get("model")
                        if task_planner_metadata
                        else None
                    ),
                }
            elif coverage_scan_queue:
                action = coverage_scan_queue.pop(0)
                decision = {
                    "mode": "coverage_viewpoint_scan",
                    "active_surface_id": None,
                    "frontier_target_cell": frontier_target_cell,
                    "remaining_scan_steps": len(coverage_scan_queue),
                }
            elif step >= int(args.max_steps) and not scan_queue:
                action = "turn_left"
                decision = {
                    "mode": "step_budget_graceful_stop",
                    "active_surface_id": None,
                    "frontier_target_cell": None,
                }
            elif exploration_phase == "deep_familiarization" and not ranked_frontiers:
                scan_turns = max(4, int(round(360.0 / max(1.0, float(args.turn_amount)))))
                coverage_scan_start_cells = int(mapper.explored.sum())
                coverage_scan_pending_evaluation = True
                coverage_scan_queue.extend(
                    ["turn_left"] * scan_turns
                    + ["look_down"]
                    + ["turn_left"] * scan_turns
                    + ["look_up"]
                )
                action = coverage_scan_queue.pop(0)
                decision = {
                    "mode": "coverage_completion_scan",
                    "active_surface_id": None,
                    "frontier_target_cell": None,
                    "remaining_scan_steps": len(coverage_scan_queue),
                }
            else:
                action, decision = _choose_action(
                    depth=depth,
                    mapper=mapper,
                    planning_free=safe_free,
                    current_position=current_position,
                    current_yaw=current_yaw,
                    semantic_target=semantic_target,
                    ranked_frontiers=ranked_frontiers,
                    scan_queue=scan_queue,
                    scanned_surface_ids=scanned_surface_ids,
                    active_surface_id=active_surface_id,
                    frontier_target_cell=frontier_target_cell,
                    frontier_target_view_yaw_deg=frontier_target_view_yaw_deg,
                    surface_arrival_radius_m=float(args.surface_arrival_radius_m),
                    frontier_arrival_radius_m=float(args.frontier_arrival_radius_m),
                    obstacle_stop_depth_m=float(args.obstacle_stop_depth_m),
                    turn_amount_deg=float(args.turn_amount),
                    path_lookahead_m=float(args.path_lookahead_m),
                    execution_planner=str(args.execution_planner),
                    pathfinder=session.sim.pathfinder,
                    navmesh_max_snap_m=float(args.navmesh_max_snap_m),
                )
            decision["exploration_phase"] = exploration_phase
            decision["coverage_confirmations"] = coverage_confirmations
            decision["familiarity"] = familiarity_metrics
            decision["familiarization_complete_step"] = familiarization_complete_step
            decision["guided_correction_status"] = guided_correction_status
            decision["guided_correction_start_step"] = guided_correction_start_step
            decision["guided_correction_complete_step"] = guided_correction_complete_step
            decision["task_injection_step"] = task_injection_step
            decision["task_planning_complete_step"] = task_planning_complete_step
            decision["task_text"] = (
                task_text if task_injection_step is not None else None
            )
            decision["task_candidate_order"] = list(task_candidate_order)
            decision["task_active_candidate_id"] = (
                f"track_{int(semantic_target['track_id'])}"
                if semantic_target is not None
                else None
            )
            decision["task_planner_mode"] = (
                task_planner_metadata.get("mode_used")
                if task_planner_metadata
                else None
            )
            decision["task_planner_model"] = (
                task_planner_metadata.get("model")
                if task_planner_metadata
                else None
            )
            decision["task_planner_reason"] = (
                task_planner_output.get("reason")
                if task_planner_output
                else None
            )
            decision["scene_vlm_mode"] = (
                scene_vlm_metadata.get("mode_used")
                if scene_vlm_metadata
                else None
            )
            decision["scene_vlm_model"] = (
                scene_vlm_metadata.get("model")
                if scene_vlm_metadata
                else None
            )
            decision["scene_summary"] = (
                scene_understanding.get("scene_summary")
                if scene_understanding
                else None
            )
            decision["task_search_ranking"] = task_search_ranking[:5]
            frame["action"] = action
            frame["decision_mode"] = decision.get("mode")
            active_surface_id = decision.get("active_surface_id")
            decision["task_active_candidate_id"] = (
                f"track_{int(active_surface_id)}"
                if active_surface_id is not None
                else None
            )
            frontier_target_cell = tuple(decision["frontier_target_cell"]) if decision.get("frontier_target_cell") else None
            if "frontier_target_view_yaw_deg" in decision:
                frontier_target_view_yaw_deg = decision.get("frontier_target_view_yaw_deg")
            if decision.get("target_unreachable"):
                if decision.get("target_kind") == "frontier" and decision.get("navigation_goal_cell"):
                    blacklisted_frontiers.append(tuple(int(value) for value in decision["navigation_goal_cell"]))
                    frontier_target_cell = None
                    frontier_target_view_yaw_deg = None
                if active_surface_id is not None:
                    if decision.get("target_kind") == "cup":
                        inspected_cup_track_ids.add(int(active_surface_id))
                        cup_confirmation_terminal_statuses[
                            int(active_surface_id)
                        ] = "inconclusive_unreachable"
                    elif decision.get("target_kind") == "semantic":
                        failed_surface_ids.add(int(active_surface_id))
                    if exploration_phase == "task_execution":
                        task_plan_events.append(
                            {
                                "step": step,
                                "event": "candidate_unreachable",
                                "candidate_id": f"track_{int(active_surface_id)}",
                                "target_kind": decision.get("target_kind"),
                            }
                        )
                    active_surface_id = None
            if decision.get("surface_scan_started") is not None:
                scanned_id = int(decision["surface_scan_started"])
                scanned_track = next((track for track in tracks if track.track_id == scanned_id), None)
                if exploration_phase == "task_execution":
                    surface_scan_in_progress = {
                        "candidate_id": f"track_{scanned_id}",
                        "track_id": scanned_id,
                        "start_step": step,
                        "observable_frames": 0,
                        "position_xz": [
                            float(scanned_track.position[0]),
                            float(scanned_track.position[2]),
                        ]
                        if scanned_track is not None
                        else list(
                            decision.get(
                                "target_world_xz",
                                [0.0, 0.0],
                            )
                        ),
                    }
                    task_plan_events.append(
                        {
                            "step": step,
                            "event": "support_surface_inspection_started",
                            "candidate_id": f"track_{scanned_id}",
                            "search_hypothesis": next(
                                (
                                    item
                                    for item in task_search_ranking
                                    if item["candidate_id"]
                                    == f"track_{scanned_id}"
                                ),
                                None,
                            ),
                        }
                    )
            if decision.get("cup_scan_started") is not None:
                scanned_cup_id = int(decision["cup_scan_started"])
                cup_scan_in_progress_id = scanned_cup_id
                cup_confirmation_attempts[scanned_cup_id] = (
                    cup_confirmation_attempts.get(scanned_cup_id, 0) + 1
                )
                if exploration_phase == "task_execution":
                    task_plan_events.append(
                        {
                            "step": step,
                            "event": "cup_candidate_reobservation_started",
                            "candidate_id": f"track_{scanned_cup_id}",
                            "attempt": cup_confirmation_attempts[scanned_cup_id],
                        }
                    )
            if (
                active_surface_id is not None
                and decision.get("mode")
                in {
                    "cup_candidate_scan",
                    "semantic_target_scan",
                    "cup_confirmation_finalize",
                }
                and int(active_surface_id) in positive_track_ids
                and next(
                    (
                        track.label
                        for track in tracks
                        if track.track_id == int(active_surface_id)
                    ),
                    None,
                )
                == "cup"
            ):
                focused_cup_track_ids.add(int(active_surface_id))
            confirmation_track_id = (
                int(active_surface_id)
                if active_surface_id is not None
                and decision.get("mode")
                in {
                    "cup_candidate_scan",
                    "semantic_target_scan",
                    "cup_confirmation_finalize",
                }
                and next(
                    (
                        track.label
                        for track in tracks
                        if track.track_id == int(active_surface_id)
                    ),
                    None,
                )
                == "cup"
                else None
            )
            if confirmation_track_id is not None:
                verification_started = time.perf_counter()
                confirmation_observation = _record_cup_confirmation_observation(
                    track_id=confirmation_track_id,
                    step=step,
                    rgb=rgb,
                    depth=depth,
                    projected_detections=projected,
                    camera_signature=camera_signature,
                    observations=cup_confirmation_observations,
                    config=cup_confirmation_config,
                    worker=worker,
                    crops_dir=confirmation_crops_dir,
                    mode=str(args.cup_confirmation_mode),
                    verifier_labels=cup_verifier_labels,
                    verifier_positive_labels=cup_verifier_positive_labels,
                    verifier_box_threshold=float(args.cup_verifier_box_threshold),
                    verifier_text_threshold=float(args.cup_verifier_text_threshold),
                    verifier_min_positive_score=float(
                        args.cup_verifier_min_positive_score
                    ),
                    verifier_min_score_margin=float(
                        args.cup_verifier_min_score_margin
                    ),
                    crop_padding_ratio=float(
                        args.cup_verifier_crop_padding_ratio
                    ),
                )
                confirmation_verification_ms = (
                    time.perf_counter() - verification_started
                ) * 1000.0
                if confirmation_observation is not None:
                    confirmation_result = evaluate_cup_confirmation(
                        cup_confirmation_observations[confirmation_track_id],
                        cup_confirmation_config,
                    )
                    task_plan_events.append(
                        {
                            "step": step,
                            "event": "cup_confirmation_observation",
                            "candidate_id": f"track_{confirmation_track_id}",
                            "status": confirmation_result["status"],
                            "task_independent_views": confirmation_result[
                                "task_independent_views"
                            ],
                            "visual_passes": confirmation_result[
                                "visual_passes"
                            ],
                            "crop_path": confirmation_observation.get(
                                "crop_path"
                            ),
                        }
                    )
            if decision.get("mode") == "cup_confirmation_finalize":
                finalized_track_id = int(active_surface_id)
                confirmation_result = evaluate_cup_confirmation(
                    cup_confirmation_observations.get(finalized_track_id, []),
                    cup_confirmation_config,
                )
                confirmation_status = str(confirmation_result["status"])
                terminal = confirmation_status in {
                    "verified",
                    "rejected_planar_surface",
                    "rejected_visual_verifier",
                }
                attempts = cup_confirmation_attempts.get(finalized_track_id, 0)
                if terminal:
                    inspected_cup_track_ids.add(finalized_track_id)
                    cup_confirmation_terminal_statuses[
                        finalized_track_id
                    ] = confirmation_status
                    event_name = "cup_confirmation_completed"
                elif attempts >= max(1, int(args.cup_confirmation_max_attempts)):
                    inspected_cup_track_ids.add(finalized_track_id)
                    cup_confirmation_terminal_statuses[
                        finalized_track_id
                    ] = (
                        "rejected_geometry_inconsistent"
                        if confirmation_status
                        == "rejected_geometry_inconsistent"
                        else "inconclusive_max_attempts"
                    )
                    event_name = "cup_confirmation_deferred"
                else:
                    event_name = "cup_confirmation_retry_scheduled"
                final_status = cup_confirmation_terminal_statuses.get(
                    finalized_track_id,
                    confirmation_status,
                )
                belief_outcome = (
                    final_status
                    if (
                        final_status == "verified"
                        or str(final_status).startswith("rejected_")
                        or str(final_status).startswith("inconclusive")
                    )
                    else "inconclusive_confirmation"
                )
                belief_update = apply_search_evidence(
                    task_search_beliefs,
                    candidate_id=f"track_{finalized_track_id}",
                    event_id=(
                        f"cup_confirmation:track_{finalized_track_id}:"
                        f"{attempts}:{step}"
                    ),
                    outcome=belief_outcome,
                    step=step,
                    observable=True,
                )
                confirmation_event = {
                    "step": step,
                    "event": event_name,
                    "candidate_id": f"track_{finalized_track_id}",
                    "status": final_status,
                    "attempt": attempts,
                    "task_independent_views": confirmation_result[
                        "task_independent_views"
                    ],
                    "visual_passes": confirmation_result["visual_passes"],
                    "visual_negatives": confirmation_result[
                        "visual_negatives"
                    ],
                    "belief_update": belief_update,
                }
                task_plan_events.append(confirmation_event)
                search_evidence_updates.append(confirmation_event)
                search_belief_revision += 1
                _write_jsonl_event(
                    search_belief_file,
                    {
                        "revision": search_belief_revision,
                        **confirmation_event,
                        "beliefs": task_search_beliefs,
                    },
                )
                decision["cup_confirmation_status"] = (
                    final_status
                )
                cup_scan_in_progress_id = None
                active_surface_id = None
            if ranked_frontiers:
                no_frontier_steps = 0
            else:
                no_frontier_steps += 1
            policy_ready = time.perf_counter()
            bev_path = bev_frames_dir / f"frame_{step:04d}_bev.png"
            _render_online_bev(
                mapper,
                tracks=tracks,
                current_position=current_position,
                current_yaw=current_yaw,
                target_world_xz=decision.get("target_world_xz"),
            ).save(bev_path)
            frame["bev_png"] = str(bev_path)
            visualization_ready = time.perf_counter()

            before_action = np.asarray(session.sim.get_agent(0).get_state().position, dtype=np.float32)
            if action != "wait":
                session.sim.step(action)
            after_action = np.asarray(session.sim.get_agent(0).get_state().position, dtype=np.float32)
            moved_m = float(np.linalg.norm(after_action - before_action))
            forward_direction = np.asarray(
                [
                    -math.sin(math.radians(current_yaw)),
                    0.0,
                    -math.cos(math.radians(current_yaw)),
                ],
                dtype=np.float32,
            )
            forward_progress_m = float(np.dot(after_action - before_action, forward_direction))
            waypoint = decision.get("navigation_waypoint_world_xz")
            waypoint_progress_m = 0.0
            if waypoint is not None:
                before_distance = math.hypot(
                    float(waypoint[0]) - float(before_action[0]),
                    float(waypoint[1]) - float(before_action[2]),
                )
                after_distance = math.hypot(
                    float(waypoint[0]) - float(after_action[0]),
                    float(waypoint[1]) - float(after_action[2]),
                )
                waypoint_progress_m = before_distance - after_distance
            collision = bool(
                action == "move_forward"
                and (
                    moved_m < 0.55 * float(args.move_amount)
                    or forward_progress_m < 0.45 * float(args.move_amount)
                )
            )
            consecutive_collisions = consecutive_collisions + 1 if collision else 0
            target_key = decision.get("navigation_target_key")
            if target_key != navigation_target_key:
                navigation_target_key = str(target_key) if target_key is not None else None
                navigation_pose_history = []
                no_progress_forward_attempts = 0
            if target_key is not None:
                navigation_pose_history.append(after_action.copy())
                navigation_pose_history = navigation_pose_history[-18:]
            if action == "move_forward":
                if collision:
                    no_progress_forward_attempts += 1
                elif waypoint is None:
                    no_progress_forward_attempts = 0
                elif waypoint_progress_m < 0.025:
                    no_progress_forward_attempts += 1
                else:
                    no_progress_forward_attempts = 0
            stuck = no_progress_forward_attempts >= int(args.stuck_forward_attempts)
            if len(navigation_pose_history) >= 18:
                window_displacement = float(
                    np.linalg.norm(navigation_pose_history[-1] - navigation_pose_history[0])
                )
                stuck = stuck or window_displacement < max(0.35, 1.5 * float(args.move_amount))
            hard_navigation_failure = bool(
                stuck or consecutive_collisions >= int(args.stuck_forward_attempts)
            )
            if hard_navigation_failure:
                frontier_target_cell = None
                frontier_target_view_yaw_deg = None
                if decision.get("target_kind") == "frontier" and decision.get("navigation_goal_cell"):
                    blacklisted_frontiers.append(tuple(int(value) for value in decision["navigation_goal_cell"]))
                if cup_scan_in_progress_id is not None:
                    scan_queue.clear()
                    task_plan_events.append(
                        {
                            "step": step,
                            "event": "cup_confirmation_baseline_motion_failed",
                            "candidate_id": f"track_{int(cup_scan_in_progress_id)}",
                        }
                    )
                elif active_surface_id is not None:
                    if decision.get("target_kind") == "cup":
                        inspected_cup_track_ids.add(active_surface_id)
                        cup_confirmation_terminal_statuses[
                            int(active_surface_id)
                        ] = "inconclusive_navigation_failed"
                    else:
                        failed_surface_ids.add(active_surface_id)
                    if exploration_phase == "task_execution":
                        task_plan_events.append(
                            {
                                "step": step,
                                "event": "candidate_navigation_failed",
                                "candidate_id": f"track_{int(active_surface_id)}",
                                "target_kind": decision.get("target_kind"),
                            }
                        )
                    active_surface_id = None
                recovery_queue.extend(["turn_right"] * 3)
                navigation_target_key = None
                navigation_pose_history = []
                no_progress_forward_attempts = 0
            action_ready = time.perf_counter()

            detector_reobserved_cups = _confirmed_cups(
                tracks,
                min_views=int(args.cup_min_views),
                min_confidence=float(args.cup_min_confidence),
                required_track_ids=focused_cup_track_ids,
            )
            confirmation_evaluations = _cup_confirmation_evaluations(
                cup_confirmation_observations,
                cup_confirmation_config,
                terminal_statuses=cup_confirmation_terminal_statuses,
                attempts=cup_confirmation_attempts,
            )
            verified_track_ids = {
                track_id
                for track_id, status in cup_confirmation_terminal_statuses.items()
                if status == "verified"
            }
            confirmed_cups = [
                track for track in tracks if track.track_id in verified_track_ids
            ]
            track_by_id = {track.track_id: track for track in tracks}
            positive_memory_ids = [
                f"{track_by_id[track_id].label}_{track_id}"
                for track_id in sorted(positive_track_ids)
                if track_id in track_by_id
            ]
            evidence_events = {
                "positive_observation_ids": positive_memory_ids,
                "not_observable_ids": sorted(
                    str(item_id)
                    for item_id, state in observability.items()
                    if state == "not_observable"
                ),
                "expected_visible_miss_ids": sorted(
                    str(item_id)
                    for item_id, state in observability.items()
                    if state == "expected_visible_miss"
                ),
            }
            record = {
                "step": step,
                "task": task_text if task_injection_step is not None else None,
                "task_injection_step": task_injection_step,
                "task_plan_events": [
                    event
                    for event in task_plan_events
                    if int(event.get("step", -1)) == step
                ],
                "observation_frame": step,
                "causal_frame_max": step,
                "action": action,
                "collision": collision,
                "stuck": stuck,
                "moved_m": moved_m,
                "forward_progress_m": forward_progress_m,
                "waypoint_progress_m": waypoint_progress_m,
                "evidence_weight": evidence_weight,
                "num_raw_detections": len(grounding_response.get("detections", [])),
                "num_projected_detections": len(projected),
                "num_tracks": len(tracks),
                "bev": mapper.snapshot(),
                "positive_track_ids": sorted(positive_track_ids),
                "memory_update": memory_update,
                "evidence_events": evidence_events,
                "memory_summary": memory.summary(),
                "confirmed_cup_track_ids": [track.track_id for track in confirmed_cups],
                "detector_reobserved_cup_track_ids": [
                    track.track_id for track in detector_reobserved_cups
                ],
                "cup_confirmation": {
                    str(track_id): result
                    for track_id, result in confirmation_evaluations.items()
                },
                "focused_cup_track_ids": sorted(focused_cup_track_ids),
                "interest": {
                    "semantic_target": semantic_target,
                    "frontier_top5": ranked_frontiers[:5],
                    "blacklisted_frontiers": [list(cell) for cell in blacklisted_frontiers],
                    **decision,
                },
                "timing_ms": {
                    "capture": (observation_ready - loop_started) * 1000.0,
                    "mapping": (mapping_ready - observation_ready) * 1000.0,
                    "grounding": (grounding_ready - mapping_ready) * 1000.0,
                    "confirmation_verification": confirmation_verification_ms,
                    "memory": (memory_ready - grounding_ready) * 1000.0,
                    "policy": (policy_ready - memory_ready) * 1000.0,
                    "visualization": (visualization_ready - policy_ready) * 1000.0,
                    "action": (action_ready - visualization_ready) * 1000.0,
                    "total": (action_ready - loop_started) * 1000.0,
                },
            }
            step_records.append(record)
            trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            trace_file.flush()

            if decision.get("mode") == "step_budget_graceful_stop":
                stop_reason = "step_budget_completed_current_scan"
                break
            if (
                exploration_phase == "task_execution"
                and familiarization_complete_step is not None
                and semantic_target is None
                and not scan_queue
                and not coverage_scan_queue
                and not recovery_queue
                and frontier_target_cell is None
                and not ranked_frontiers
                and no_frontier_steps >= max(2, int(args.frontier_patience))
                and cup_scan_in_progress_id is None
                and step >= int(args.initial_yaw_steps)
            ):
                stop_reason = "task_execution_exhausted"
                break
    finally:
        trace_file.close()
        search_belief_file.close()
        worker.close()
        if lingbot_worker is not None:
            lingbot_worker.close()
        session.close()

    detector_reobserved_cups = _confirmed_cups(
        tracks,
        min_views=int(args.cup_min_views),
        min_confidence=float(args.cup_min_confidence),
        required_track_ids=focused_cup_track_ids,
    )
    confirmation_evaluations = _cup_confirmation_evaluations(
        cup_confirmation_observations,
        cup_confirmation_config,
        terminal_statuses=cup_confirmation_terminal_statuses,
        attempts=cup_confirmation_attempts,
    )
    verified_track_ids = {
        track_id
        for track_id, result in confirmation_evaluations.items()
        if bool(result.get("verified"))
    }
    confirmed_cups = [
        track for track in tracks if track.track_id in verified_track_ids
    ]
    candidate_cups = _confirmed_cups(
        tracks,
        min_views=int(args.cup_min_views),
        min_confidence=float(args.cup_min_confidence),
    )
    checkpoint_artifacts["final"] = _save_online_checkpoint(
        checkpoints_dir=checkpoints_dir,
        name="final",
        step=max(0, len(step_records) - 1),
        phase=exploration_phase,
        mapper=mapper,
        memory=memory,
        tracks=tracks,
        current_position=current_position,
        current_yaw=current_yaw,
    )
    np.savez_compressed(
        out_dir / "online_bev_state.npz",
        occupancy_logodds=mapper.occupancy_logodds,
        explored=mapper.explored,
        observation_count=mapper.observation_count,
        origin_world_xz=np.asarray(mapper.origin_world_xz, dtype=np.float32),
        resolution=np.asarray([mapper.config.resolution], dtype=np.float32),
    )
    memory.save(out_dir / "online_object_memory.json")
    (out_dir / "online_tracks.json").write_text(
        json.dumps({"items": [track.to_dict() for track in tracks]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "detections.json").write_text(
        json.dumps({"detections": all_detections}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "phase": "phase5a_online_interest_explorer",
        "task": task_text,
        "scene": str(Path(args.scene).expanduser().resolve()),
        "scene_dataset_config": str(Path(args.scene_dataset_config).expanduser().resolve()) if args.scene_dataset_config else None,
        "resolution": int(args.resolution),
        "hfov_deg": 90.0,
        "num_frames": len(frames),
        "frames": frames,
        "online_contract": {
            "observe_ground_update_decide_step": True,
            "semantic_sensor_enabled": False,
            "oracle_metrics_enabled": False,
            "semantic_scene_read": False,
            "precomputed_route": False,
            "pose_source": "Habitat exact pose",
            "depth_source": (
                "LingBot-Map causal depth with one-time bootstrap scale calibration"
                if args.geometry_source == "lingbot_depth_exact_pose"
                else "Habitat depth"
            ),
            "geometry_source": str(args.geometry_source),
            "lingbot_depth_scale": lingbot_depth_scale,
            "lingbot_bootstrap_uses_habitat_depth_for_scale_only": (
                args.geometry_source == "lingbot_depth_exact_pose"
            ),
            "navigation_target_source": "online observed BEV frontier or online grounded semantic memory",
            "task_injection": {
                "task_hidden_during_familiarization": True,
                "task_text": task_text,
                "injection_step": task_injection_step,
                "planning_complete_step": task_planning_complete_step,
                "planner_role": "semantic candidate ordering only; no low-level actions",
                "scene_vlm": {
                    "mode": (
                        scene_vlm_metadata.get("mode_used")
                        if scene_vlm_metadata
                        else None
                    ),
                    "model": (
                        scene_vlm_metadata.get("model")
                        if scene_vlm_metadata
                        else None
                    ),
                    "role": (
                        "grounded room/support interpretation and search priors; "
                        "never final object confirmation"
                    ),
                    "uses_only_familiarization_keyframes": True,
                },
            },
            "cup_confirmation": {
                "mode": str(args.cup_confirmation_mode),
                "requires_task_stage_independent_views": True,
                "requires_3d_position_consistency": True,
                "requires_crop_visual_verification": (
                    args.cup_confirmation_mode == "grounding_crop"
                ),
            },
            "manual_guided_correction": {
                "enabled": guided_correction_target is not None,
                "target_position_xyz": (
                    [float(value) for value in guided_correction_target]
                    if guided_correction_target is not None
                    else None
                ),
                "trigger_step": int(args.guided_correction_trigger_step),
                "role": "one authorized human waypoint intervention; no teleport",
            },
            "navmesh_online_usage": (
                "complete-scene navmesh global shortest-path query for the current online target; "
                "privileged geometric execution"
                if args.execution_planner == "hybrid_navmesh"
                else "initial spawn only"
            ),
        },
    }
    (out_dir / "frames_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "task": task_text,
        "stop_reason": stop_reason,
        "num_steps": len(step_records),
        "num_projected_detections": len(all_detections),
        "num_tracks": len(tracks),
        "num_confirmed_cups": len(confirmed_cups),
        "confirmed_cups": [
            {
                **track.to_dict(),
                "confirmation": confirmation_evaluations[track.track_id],
            }
            for track in confirmed_cups
        ],
        "num_detector_reobserved_cups": len(detector_reobserved_cups),
        "detector_reobserved_cups": [
            track.to_dict() for track in detector_reobserved_cups
        ],
        "cup_confirmation": {
            "config": {
                "mode": str(args.cup_confirmation_mode),
                "min_task_views": cup_confirmation_config.min_task_views,
                "min_visual_passes": cup_confirmation_config.min_visual_passes,
                "min_visual_negatives": (
                    cup_confirmation_config.min_visual_negatives
                ),
                "max_attempts": max(1, int(args.cup_confirmation_max_attempts)),
                "min_depth_relief_passes": (
                    cup_confirmation_config.min_depth_relief_passes
                ),
                "min_depth_relief_m": (
                    cup_confirmation_config.min_depth_relief_m
                ),
                "max_position_spread_m": (
                    cup_confirmation_config.max_position_spread_m
                ),
                "verifier_labels": cup_verifier_labels,
                "verifier_positive_labels": sorted(
                    cup_verifier_positive_labels
                ),
                "verifier_min_positive_score": float(
                    args.cup_verifier_min_positive_score
                ),
                "verifier_min_score_margin": float(
                    args.cup_verifier_min_score_margin
                ),
            },
            "results": {
                str(track_id): result
                for track_id, result in confirmation_evaluations.items()
            },
            "observations": {
                str(track_id): items
                for track_id, items in cup_confirmation_observations.items()
            },
        },
        "num_candidate_cups": len(candidate_cups),
        "candidate_cups": [track.to_dict() for track in candidate_cups],
        "focused_cup_track_ids": sorted(focused_cup_track_ids),
        "inspected_cup_track_ids": sorted(inspected_cup_track_ids),
        "familiarization_complete_step": familiarization_complete_step,
        "familiarization_snapshot": familiarization_snapshot,
        "task_injection_step": task_injection_step,
        "task_planning_complete_step": task_planning_complete_step,
        "task_execution_steps": (
            max(0, len(step_records) - int(task_planning_complete_step))
            if task_planning_complete_step is not None
            else 0
        ),
        "task_planner": (
            {
                key: value
                for key, value in task_planner_metadata.items()
                if key not in {"prompt", "raw_response"}
            }
            if task_planner_metadata is not None
            else None
        ),
        "task_planner_output": task_planner_output,
        "task_candidate_order": task_candidate_order,
        "task_planner_seed_order": task_planner_seed_order,
        "scene_understanding": scene_understanding,
        "scene_vlm": (
            {
                key: value
                for key, value in scene_vlm_metadata.items()
                if key not in {"prompt", "raw_response"}
            }
            if scene_vlm_metadata is not None
            else None
        ),
        "search_beliefs": task_search_beliefs,
        "search_belief_revisions": search_belief_revision,
        "search_belief_trace": str(search_belief_path),
        "final_task_search_ranking": task_search_ranking,
        "search_evidence_updates": search_evidence_updates,
        "task_plan_events": task_plan_events,
        "cup_search_steps": (
            max(0, len(step_records) - int(familiarization_complete_step) - 1)
            if familiarization_complete_step is not None
            else 0
        ),
        "guided_correction": (
            guided_correction_result
            if guided_correction_result is not None
            else {
                "status": guided_correction_status,
                "target_position_xyz": (
                    [float(value) for value in guided_correction_target]
                    if guided_correction_target is not None
                    else None
                ),
                "start_step": guided_correction_start_step,
                "complete_step": guided_correction_complete_step,
            }
        ),
        "num_scanned_surfaces": len(scanned_surface_ids),
        "num_scanned_surface_regions": len(scanned_surface_positions),
        "num_failed_surfaces": len(failed_surface_ids),
        "num_blacklisted_frontiers": len(blacklisted_frontiers),
        "num_detected_collisions": sum(1 for record in step_records if record.get("collision")),
        "num_stuck_events": sum(1 for record in step_records if record.get("stuck")),
        "coverage_confirmations": coverage_confirmations,
        "final_exploration_phase": exploration_phase,
        "execution_planner": str(args.execution_planner),
        "frontier_strategy": str(args.frontier_strategy),
        "geometry_source": str(args.geometry_source),
        "lingbot_depth_scale": lingbot_depth_scale,
        "bev": mapper.snapshot(),
        "memory": memory.summary(),
        "worker": worker.ready,
        "lingbot_worker": lingbot_worker.ready if lingbot_worker is not None else None,
        "timing_ms": _timing_summary(step_records),
        "causal_invariants": {
            "all_decisions_use_current_or_past_frames": all(
                int(record["causal_frame_max"]) <= int(record["step"])
                for record in step_records
            ),
            "semantic_sensor_disabled": True,
            "precomputed_route_disabled": True,
            "task_hidden_until_memory_ready": (
                task_injection_step is not None
                and familiarization_complete_step is not None
                and int(task_injection_step) >= int(familiarization_complete_step)
            ),
        },
        "evidence_event_totals": {
            "positive_observation": sum(
                len(record.get("evidence_events", {}).get("positive_observation_ids", []))
                for record in step_records
            ),
            "not_observable": sum(
                len(record.get("evidence_events", {}).get("not_observable_ids", []))
                for record in step_records
            ),
            "expected_visible_miss": sum(
                len(record.get("evidence_events", {}).get("expected_visible_miss_ids", []))
                for record in step_records
            ),
        },
        "checkpoints": checkpoint_artifacts,
        "artifacts": {
            "trace": str(trace_path),
            "frames_metadata": str(out_dir / "frames_metadata.json"),
            "detections": str(out_dir / "detections.json"),
            "memory": str(out_dir / "online_object_memory.json"),
            "tracks": str(out_dir / "online_tracks.json"),
            "bev_state": str(out_dir / "online_bev_state.npz"),
            "bev_frames": str(bev_frames_dir),
            "task_planner": str(planner_dir),
            "checkpoints": str(checkpoints_dir),
            "confirmation_crops": str(confirmation_crops_dir),
        },
    }
    (out_dir / "online_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _capture_rgbd_frame(
    session: HabitatControlSession,
    frames_dir: Path,
    step: int,
) -> tuple[dict[str, Any], Image.Image, np.ndarray]:
    observations = session.sim.get_sensor_observations()
    if "semantic" in observations:
        raise RuntimeError("Strict online runner received a forbidden semantic observation")
    rgb_array = _rgb_array(observations.get("rgb"))
    depth = _valid_depth(observations.get("depth"))
    rgb = Image.fromarray(rgb_array)
    stem = f"frame_{step:04d}"
    rgb_path = frames_dir / f"{stem}_rgb.jpg"
    depth_npy = frames_dir / f"{stem}_depth.npy"
    depth_png = frames_dir / f"{stem}_depth.png"
    rgb.save(rgb_path, quality=94)
    np.save(depth_npy, depth.astype(np.float32))
    _depth_image(depth).save(depth_png)

    agent_state = session.sim.get_agent(0).get_state()
    sensor_state = agent_state.sensor_states.get("depth") or next(iter(agent_state.sensor_states.values()))
    frame = {
        "frame_index": step,
        "action": "online_observe",
        "rgb_path": str(rgb_path),
        "depth_npy": str(depth_npy),
        "depth_png": str(depth_png),
        "semantic_npy": None,
        "bev_png": None,
        "semantic_bev_png": None,
        "memory_step": step,
        "pose": {},
        "sensor_position_xyz": _list3(sensor_state.position),
        "sensor_rotation_matrix": _rotation_matrix(sensor_state.rotation),
        "agent_position_xyz": _list3(agent_state.position),
        "agent_rotation_matrix": _rotation_matrix(agent_state.rotation),
        "semantic_report": {},
        "memory_summary": {},
    }
    return frame, rgb, depth


def _set_agent_start_pose(
    session: HabitatControlSession,
    position_xyz: np.ndarray,
    yaw_deg: float,
) -> None:
    import quaternion

    agent = session.sim.get_agent(0)
    state = agent.get_state()
    state.position = np.asarray(position_xyz, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector(
        [0.0, math.radians(float(yaw_deg)), 0.0]
    )
    try:
        agent.set_state(state, infer_sensor_states=True)
    except TypeError:
        agent.set_state(state)


def _load_lingbot_depth(
    result: dict[str, Any],
    target_shape: tuple[int, int],
    scale: float,
    confidence_threshold: float = 0.0,
) -> np.ndarray:
    raw = np.load(result["depth_npy"]).astype(np.float32)
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        raise ValueError(f"Unexpected LingBot depth shape: {raw.shape}")
    if raw.shape != tuple(target_shape):
        raw = np.asarray(
            Image.fromarray(raw, mode="F").resize(
                (int(target_shape[1]), int(target_shape[0])),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
    depth = raw * float(scale)
    if float(confidence_threshold) > 0.0:
        confidence = np.squeeze(np.load(result["depth_conf_npy"]).astype(np.float32))
        if confidence.shape != tuple(target_shape):
            confidence = np.asarray(
                Image.fromarray(confidence, mode="F").resize(
                    (int(target_shape[1]), int(target_shape[0])),
                    resample=Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )
        depth[confidence < float(confidence_threshold)] = 0.0
    depth[~np.isfinite(depth) | (depth <= 0.05)] = 0.0
    return depth


def _calibrate_lingbot_depth_scale(
    results: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    confidence_threshold: float = 0.0,
) -> float:
    ratios = []
    for result, frame in zip(results, frames):
        target = np.load(frame["depth_npy"]).astype(np.float32)
        prediction = _load_lingbot_depth(
            result,
            target_shape=target.shape,
            scale=1.0,
            confidence_threshold=confidence_threshold,
        )
        mask = (
            np.isfinite(target)
            & (target > 0.10)
            & (target < 6.0)
            & np.isfinite(prediction)
            & (prediction > 0.05)
        )
        if mask.any():
            sampled = target[mask][::16] / np.maximum(prediction[mask][::16], 1e-6)
            ratios.append(sampled)
    if not ratios:
        raise RuntimeError("Unable to calibrate LingBot depth scale from bootstrap frames")
    ratio = np.concatenate(ratios)
    lower, upper = np.percentile(ratio, [10.0, 90.0])
    trimmed = ratio[(ratio >= lower) & (ratio <= upper)]
    return float(np.median(trimmed if trimmed.size else ratio))


def _merge_online_detections(
    tracks: list[OnlineTrack],
    detections: list[dict[str, Any]],
    step: int,
    camera_signature: tuple[float, float, float],
    merge_radius_m: float,
) -> set[int]:
    positive_ids: set[int] = set()
    for detection in detections:
        label = _canonical_label(str(detection.get("label", "unknown")))
        position = np.asarray(detection.get("position_3d", []), dtype=np.float32)
        if position.shape != (3,) or not np.isfinite(position).all():
            continue
        candidates = [
            track
            for track in tracks
            if track.label == label
        ]
        best = min(
            candidates,
            key=lambda track: float(np.linalg.norm(track.position[[0, 2]] - position[[0, 2]])),
            default=None,
        )
        distance = (
            float(np.linalg.norm(best.position[[0, 2]] - position[[0, 2]]))
            if best is not None
            else math.inf
        )
        if best is None or distance > float(merge_radius_m):
            track_id = len(tracks) + 1
            score = float(detection.get("score", 0.0))
            weight = max(0.01, score)
            best = OnlineTrack(
                track_id=track_id,
                label=label,
                position_sum=position * weight,
                score_sum=score * weight,
                weight_sum=weight,
                first_seen_step=step,
                last_seen_step=step,
                visible_steps=[step],
                independent_views=[camera_signature],
                detection_count=1,
                best_score=score,
            )
            tracks.append(best)
        else:
            best.add(position, float(detection.get("score", 0.0)), step, camera_signature)
        positive_ids.add(best.track_id)
        detection["online_track_id"] = best.track_id
        detection["canonical_label"] = label
    return positive_ids


def _choose_action(
    depth: np.ndarray,
    mapper: DenseBEVMapper,
    planning_free: np.ndarray,
    current_position: np.ndarray,
    current_yaw: float,
    semantic_target: dict[str, Any] | None,
    ranked_frontiers: list[dict[str, Any]],
    scan_queue: list[str],
    scanned_surface_ids: set[int],
    active_surface_id: int | None,
    frontier_target_cell: tuple[int, int] | None,
    frontier_target_view_yaw_deg: float | None,
    surface_arrival_radius_m: float,
    frontier_arrival_radius_m: float,
    obstacle_stop_depth_m: float,
    turn_amount_deg: float,
    path_lookahead_m: float,
    execution_planner: str,
    pathfinder: Any,
    navmesh_max_snap_m: float,
) -> tuple[str, dict[str, Any]]:
    if scan_queue:
        return scan_queue.pop(0), {
            "mode": "semantic_target_scan",
            "active_surface_id": active_surface_id,
            "frontier_target_cell": frontier_target_cell,
        }

    target_world: tuple[float, float] | None = None
    navigation_goal_cell: tuple[int, int] | None = None
    target_kind = "frontier"
    navigation_target_key: str | None = None
    mode = "frontier_exploration"
    surface_scan_started = None
    current_cell = mapper.world_to_grid((float(current_position[0]), float(current_position[2])))
    if semantic_target is not None:
        active_surface_id = int(semantic_target["track_id"])
        target_label = str(semantic_target.get("label", "")).lower()
        position = np.asarray(semantic_target["position_3d"], dtype=np.float32)
        target_world = (float(position[0]), float(position[2]))
        distance = float(np.linalg.norm(position[[0, 2]] - current_position[[0, 2]]))
        target_kind = "cup" if target_label == "cup" else "semantic"
        mode = "cup_candidate_approach" if target_kind == "cup" else "semantic_interest"
        navigation_target_key = f"{target_kind}:{active_surface_id}"
        if distance <= surface_arrival_radius_m + max(0.10, 2.0 * mapper.config.resolution):
            if target_kind == "cup":
                scan_queue.extend(
                    _semantic_observation_actions(
                        current_position=current_position,
                        current_yaw=current_yaw,
                        target_world=target_world,
                        turn_amount_deg=turn_amount_deg,
                        fan_scan_deg=45.0,
                    )
                )
            else:
                surface_scan_started = active_surface_id
                scan_queue.extend(
                    ["look_down"]
                    + ["turn_right"] * 4
                    + ["turn_left"] * 8
                    + ["turn_right"] * 4
                    + ["look_up"]
                )
            if not scan_queue:
                scan_queue.extend(["turn_left", "turn_right"])
            return scan_queue.pop(0), {
                "mode": "cup_candidate_scan" if target_kind == "cup" else "semantic_surface_scan",
                "active_surface_id": active_surface_id,
                "surface_scan_started": surface_scan_started,
                "cup_scan_started": active_surface_id if target_kind == "cup" else None,
                "frontier_target_cell": frontier_target_cell,
                "target_world_xz": list(target_world),
                "target_kind": target_kind,
                "navigation_target_key": navigation_target_key,
            }
        target_cell = mapper.world_to_grid(target_world)
        if current_cell is not None and target_cell is not None:
            reachable = reachable_free_mask(planning_free, current_cell)
            navigation_goal_cell = approach_cell_for_target(
                reachable,
                target_cell,
                desired_radius_cells=max(
                    1,
                    int(round(surface_arrival_radius_m / mapper.config.resolution)),
                ),
                current=current_cell,
            )
    else:
        active_surface_id = None

    if target_world is None:
        if frontier_target_cell is not None and current_cell is not None:
            if not mapper.in_bounds(frontier_target_cell):
                frontier_target_cell = None
                frontier_target_view_yaw_deg = None
            elif float(
                np.linalg.norm(
                    np.asarray(frontier_target_cell, dtype=np.float32)
                    - np.asarray(current_cell, dtype=np.float32)
                )
                * mapper.config.resolution
            ) <= frontier_arrival_radius_m:
                frontier_target_cell = None
                frontier_target_view_yaw_deg = None
        if frontier_target_cell is None and ranked_frontiers:
            selected_frontier = ranked_frontiers[0]
            frontier_target_cell = tuple(int(value) for value in selected_frontier["cell"])
            raw_view_yaw = selected_frontier.get("view_yaw_deg")
            frontier_target_view_yaw_deg = (
                float(raw_view_yaw) if raw_view_yaw is not None else None
            )
        elif frontier_target_cell is not None and frontier_target_view_yaw_deg is None:
            matching_frontier = next(
                (
                    item
                    for item in ranked_frontiers
                    if tuple(int(value) for value in item["cell"]) == frontier_target_cell
                ),
                None,
            )
            if matching_frontier is not None and matching_frontier.get("view_yaw_deg") is not None:
                frontier_target_view_yaw_deg = float(matching_frontier["view_yaw_deg"])
        if frontier_target_cell is not None:
            navigation_goal_cell = frontier_target_cell
            navigation_target_key = f"frontier:{frontier_target_cell[0]}:{frontier_target_cell[1]}"

    if navigation_goal_cell is None or current_cell is None:
        return "turn_left", {
            "mode": "viewpoint_novelty_turn",
            "active_surface_id": active_surface_id,
            "frontier_target_cell": frontier_target_cell,
            "frontier_target_view_yaw_deg": frontier_target_view_yaw_deg,
            "target_kind": target_kind,
            "target_unreachable": target_world is not None,
            "navigation_target_key": navigation_target_key,
        }

    planner_metadata: dict[str, Any]
    if execution_planner == "hybrid_navmesh":
        navmesh_plan = _plan_navmesh_waypoint(
            pathfinder=pathfinder,
            current_position=current_position,
            target_world=mapper.grid_to_world(navigation_goal_cell),
            lookahead_m=path_lookahead_m,
            max_snap_m=navmesh_max_snap_m,
        )
        if navmesh_plan is None:
            return "turn_left", {
                "mode": "unreachable_target_recovery",
                "active_surface_id": active_surface_id,
                "frontier_target_cell": list(frontier_target_cell) if frontier_target_cell else None,
                "frontier_target_view_yaw_deg": frontier_target_view_yaw_deg,
                "target_kind": target_kind,
                "target_unreachable": True,
                "navigation_goal_cell": list(navigation_goal_cell),
                "navigation_target_key": navigation_target_key,
                "execution_planner": execution_planner,
            }
        waypoint_world = navmesh_plan["waypoint_world_xz"]
        waypoint_cell = mapper.world_to_grid(waypoint_world)
        planner_metadata = navmesh_plan["metadata"]
    else:
        path = plan_observed_path(planning_free, current_cell, navigation_goal_cell)
        if not path:
            return "turn_left", {
                "mode": "unreachable_target_recovery",
                "active_surface_id": active_surface_id,
                "frontier_target_cell": list(frontier_target_cell) if frontier_target_cell else None,
                "frontier_target_view_yaw_deg": frontier_target_view_yaw_deg,
                "target_kind": target_kind,
                "target_unreachable": True,
                "navigation_goal_cell": list(navigation_goal_cell),
                "navigation_target_key": navigation_target_key,
                "execution_planner": execution_planner,
            }
        lookahead_cells = max(
            2,
            int(round(float(path_lookahead_m) / mapper.config.resolution)),
        )
        waypoint_cell = path[min(len(path) - 1, lookahead_cells)]
        waypoint_world = mapper.grid_to_world(waypoint_cell)
        planner_metadata = {
            "execution_planner": execution_planner,
            "navigation_path_cells": len(path),
        }
    action = _reactive_action(
        depth=depth,
        current_position=current_position,
        current_yaw=current_yaw,
        target_world=waypoint_world,
        obstacle_stop_depth_m=obstacle_stop_depth_m,
        turn_amount_deg=turn_amount_deg,
    )
    return action, {
        "mode": mode,
        "active_surface_id": active_surface_id,
        "surface_scan_started": surface_scan_started,
        "frontier_target_cell": list(frontier_target_cell) if frontier_target_cell else None,
        "frontier_target_view_yaw_deg": frontier_target_view_yaw_deg,
        "target_world_xz": [
            float(mapper.grid_to_world(navigation_goal_cell)[0]),
            float(mapper.grid_to_world(navigation_goal_cell)[1]),
        ],
        "target_kind": target_kind,
        "target_unreachable": False,
        "navigation_goal_cell": list(navigation_goal_cell),
        "navigation_waypoint_cell": list(waypoint_cell) if waypoint_cell is not None else None,
        "navigation_waypoint_world_xz": [float(waypoint_world[0]), float(waypoint_world[1])],
        "navigation_target_key": navigation_target_key,
        **planner_metadata,
    }


def _frontier_observation_actions(
    current_yaw: float,
    target_yaw: float | None,
    turn_amount_deg: float,
    scan_deg: float,
) -> list[str]:
    """Align with the predicted unknown region, then make a compact fan scan."""
    turn_amount = max(1.0, float(turn_amount_deg))
    actions: list[str] = []
    if target_yaw is not None:
        delta = _signed_angle_delta_deg(float(target_yaw), float(current_yaw))
        align_turns = int(round(abs(delta) / turn_amount))
        if align_turns:
            actions.extend(
                ["turn_left" if delta > 0.0 else "turn_right"] * align_turns
            )
    half_scan_turns = max(0, int(round(float(scan_deg) / (2.0 * turn_amount))))
    if half_scan_turns:
        actions.extend(["turn_left"] * half_scan_turns)
        actions.extend(["turn_right"] * (2 * half_scan_turns))
        actions.extend(["turn_left"] * half_scan_turns)
    return actions


def _semantic_observation_actions(
    current_position: np.ndarray,
    current_yaw: float,
    target_world: tuple[float, float],
    turn_amount_deg: float,
    fan_scan_deg: float,
) -> list[str]:
    dx = float(target_world[0] - current_position[0])
    dz = float(target_world[1] - current_position[2])
    desired_yaw = math.degrees(math.atan2(-dx, -dz))
    delta = _signed_angle_delta_deg(desired_yaw, current_yaw)
    turn_amount = max(1.0, float(turn_amount_deg))
    align_turns = int(round(abs(delta) / turn_amount))
    actions = [
        "turn_left" if delta > 0.0 else "turn_right"
    ] * align_turns
    half_turns = max(1, int(round(float(fan_scan_deg) / (2.0 * turn_amount))))
    actions.extend(["look_down"])
    actions.extend(["turn_left"] * half_turns)
    actions.extend(["turn_right"] * (2 * half_turns))
    actions.extend(["turn_left"] * half_turns)
    actions.extend(["look_up"])
    baseline_turns = max(1, int(round(45.0 / turn_amount)))
    actions.extend(["turn_left"] * baseline_turns)
    actions.extend(["move_forward"] * 2)
    actions.extend(["turn_right"] * baseline_turns)
    actions.extend(["look_down"])
    actions.extend(["turn_left"] * half_turns)
    actions.extend(["turn_right"] * (2 * half_turns))
    actions.extend(["turn_left"] * half_turns)
    actions.extend(["look_up"])
    return actions


def _choose_guided_correction_action(
    depth: np.ndarray,
    current_position: np.ndarray,
    current_yaw: float,
    target_position_xyz: np.ndarray,
    obstacle_stop_depth_m: float,
    turn_amount_deg: float,
    path_lookahead_m: float,
    pathfinder: Any,
    navmesh_max_snap_m: float,
) -> tuple[str, dict[str, Any]]:
    target_world = (
        float(target_position_xyz[0]),
        float(target_position_xyz[2]),
    )
    plan = _plan_navmesh_waypoint(
        pathfinder=pathfinder,
        current_position=current_position,
        target_world=target_world,
        lookahead_m=path_lookahead_m,
        max_snap_m=navmesh_max_snap_m,
    )
    if plan is None:
        return "turn_left", {
            "mode": "guided_correction_unreachable",
            "active_surface_id": None,
            "frontier_target_cell": None,
            "target_world_xz": list(target_world),
            "target_kind": "guided_correction",
            "target_unreachable": True,
            "navigation_target_key": "guided_correction:manual_once",
        }
    waypoint_world = plan["waypoint_world_xz"]
    action = _reactive_action(
        depth=depth,
        current_position=current_position,
        current_yaw=current_yaw,
        target_world=waypoint_world,
        obstacle_stop_depth_m=obstacle_stop_depth_m,
        turn_amount_deg=turn_amount_deg,
    )
    return action, {
        "mode": "guided_correction_navigation",
        "active_surface_id": None,
        "frontier_target_cell": None,
        "target_world_xz": list(target_world),
        "target_kind": "guided_correction",
        "target_unreachable": False,
        "navigation_waypoint_world_xz": [
            float(waypoint_world[0]),
            float(waypoint_world[1]),
        ],
        "navigation_target_key": "guided_correction:manual_once",
        **plan["metadata"],
    }


def _plan_navmesh_waypoint(
    pathfinder: Any,
    current_position: np.ndarray,
    target_world: tuple[float, float],
    lookahead_m: float,
    max_snap_m: float,
) -> dict[str, Any] | None:
    if pathfinder is None or not getattr(pathfinder, "is_loaded", False):
        return None
    import habitat_sim

    start = np.asarray(current_position, dtype=np.float32)
    requested_goal = np.asarray(
        [float(target_world[0]), float(start[1]), float(target_world[1])],
        dtype=np.float32,
    )
    snapped_start = np.asarray(pathfinder.snap_point(start), dtype=np.float32)
    snapped_goal = np.asarray(pathfinder.snap_point(requested_goal), dtype=np.float32)
    if not np.isfinite(snapped_start).all() or not np.isfinite(snapped_goal).all():
        return None
    snap_distance_m = float(np.linalg.norm(snapped_goal - requested_goal))
    if snap_distance_m > float(max_snap_m):
        return None

    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = snapped_start
    shortest_path.requested_end = snapped_goal
    if not pathfinder.find_path(shortest_path):
        return None
    points = [np.asarray(point, dtype=np.float32) for point in shortest_path.points]
    if not points:
        return None
    waypoint = _polyline_lookahead(points, max(0.15, float(lookahead_m)))
    return {
        "waypoint_world_xz": (float(waypoint[0]), float(waypoint[2])),
        "metadata": {
            "execution_planner": "hybrid_navmesh",
            "navmesh_geodesic_distance_m": float(shortest_path.geodesic_distance),
            "navmesh_goal_snap_distance_m": snap_distance_m,
            "navmesh_path_points": len(points),
        },
    }


def _polyline_lookahead(points: list[np.ndarray], distance_m: float) -> np.ndarray:
    if len(points) == 1:
        return points[0].copy()
    remaining = max(0.0, float(distance_m))
    for start, end in zip(points[:-1], points[1:]):
        segment = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
        length = float(np.linalg.norm(segment))
        if length <= 1e-6:
            continue
        if remaining <= length:
            return np.asarray(start, dtype=np.float32) + segment * (remaining / length)
        remaining -= length
    return np.asarray(points[-1], dtype=np.float32).copy()


def _reactive_action(
    depth: np.ndarray,
    current_position: np.ndarray,
    current_yaw: float,
    target_world: tuple[float, float],
    obstacle_stop_depth_m: float,
    turn_amount_deg: float,
) -> str:
    dx = float(target_world[0] - current_position[0])
    dz = float(target_world[1] - current_position[2])
    desired_yaw = math.degrees(math.atan2(-dx, -dz))
    delta = _signed_angle_delta_deg(desired_yaw, current_yaw)
    # A discrete turn can overshoot a waypoint bearing by up to half a step.
    # Keep a wider deadband so adjacent frames do not alternate left/right.
    heading_tolerance_deg = max(8.0, 0.85 * float(turn_amount_deg))
    if abs(delta) > heading_tolerance_deg:
        return "turn_left" if delta > 0.0 else "turn_right"

    h, w = depth.shape
    center = depth[int(0.35 * h) : int(0.75 * h), int(0.40 * w) : int(0.60 * w)]
    center_valid = center[np.isfinite(center) & (center > 0.05)]
    center_depth = float(np.percentile(center_valid, 20)) if center_valid.size else math.inf
    if center_depth >= float(obstacle_stop_depth_m):
        return "move_forward"
    left = depth[int(0.30 * h) : int(0.80 * h), int(0.05 * w) : int(0.45 * w)]
    right = depth[int(0.30 * h) : int(0.80 * h), int(0.55 * w) : int(0.95 * w)]
    left_clearance = _clearance(left)
    right_clearance = _clearance(right)
    return "turn_left" if left_clearance >= right_clearance else "turn_right"


def _observability_for_memory(
    memory: ObjectMemoryStore,
    tracks: list[OnlineTrack],
    positive_track_ids: set[int],
    frame: dict[str, Any],
    depth: np.ndarray,
    hfov_deg: float,
) -> dict[str, str]:
    sensor_position = np.asarray(frame["sensor_position_xyz"], dtype=np.float32)
    rotation = np.asarray(frame["sensor_rotation_matrix"], dtype=np.float32)
    h, w = depth.shape
    fx = w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = fx
    observability: dict[str, str] = {}
    tracks_by_id = {track.track_id: track for track in tracks}
    for item in memory.items.values():
        if int(item.semantic_id) in positive_track_ids:
            continue
        track = tracks_by_id.get(int(item.semantic_id))
        if (
            track is None
            or len(track.independent_views) < 3
            or track.confidence < 0.24
        ):
            observability[item.id] = "not_observable"
            continue
        world = track.position.astype(np.float32)
        camera = rotation.T @ (world - sensor_position)
        forward = float(-camera[2])
        if forward <= 0.05 or forward >= 6.0:
            observability[item.id] = "not_observable"
            continue
        horizontal = math.degrees(math.atan2(float(camera[0]), forward))
        vertical = math.degrees(math.atan2(float(camera[1]), forward))
        if abs(horizontal) > 0.45 * hfov_deg or abs(vertical) > 38.0:
            observability[item.id] = "not_observable"
            continue
        u = int(round((w - 1) / 2.0 + fx * float(camera[0]) / forward))
        v = int(round((h - 1) / 2.0 - fy * float(camera[1]) / forward))
        x0, x1 = max(0, u - 3), min(w, u + 4)
        y0, y1 = max(0, v - 3), min(h, v + 4)
        patch = depth[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.05)]
        if valid.size == 0 or float(np.median(valid)) < forward - 0.35:
            observability[item.id] = "not_observable"
        else:
            observability[item.id] = "expected_visible_miss"
    return observability


def _confirmed_cups(
    tracks: list[OnlineTrack],
    min_views: int,
    min_confidence: float,
    required_track_ids: set[int] | None = None,
) -> list[OnlineTrack]:
    return [
        track
        for track in tracks
        if track.label == "cup"
        and len(track.independent_views) >= int(min_views)
        and track.confidence >= float(min_confidence)
        and (required_track_ids is None or track.track_id in required_track_ids)
    ]


def _record_cup_confirmation_observation(
    *,
    track_id: int,
    step: int,
    rgb: Image.Image,
    depth: np.ndarray,
    projected_detections: list[dict[str, Any]],
    camera_signature: tuple[float, float, float],
    observations: dict[int, list[dict[str, Any]]],
    config: CupConfirmationConfig,
    worker: GroundingWorker,
    crops_dir: Path,
    mode: str,
    verifier_labels: list[str],
    verifier_positive_labels: set[str],
    verifier_box_threshold: float,
    verifier_text_threshold: float,
    verifier_min_positive_score: float,
    verifier_min_score_margin: float,
    crop_padding_ratio: float,
) -> dict[str, Any] | None:
    track_observations = observations.setdefault(int(track_id), [])
    cup_detections = [
        detection
        for detection in projected_detections
        if str(detection.get("canonical_label", "")).lower() == "cup"
    ]
    candidates = [
        detection
        for detection in cup_detections
        if int(detection.get("online_track_id", -1)) == int(track_id)
    ]
    track_reassociated = False
    if not candidates:
        reference_positions = [
            np.asarray(item.get("position_3d", ()), dtype=np.float32)
            for item in track_observations
            if len(item.get("position_3d", ())) == 3
            and str(item.get("crop_verifier_status", "")) != "error"
        ]
        if reference_positions:
            reference = np.median(
                np.stack(reference_positions, axis=0),
                axis=0,
            )
            nearby = [
                (
                    float(
                        np.linalg.norm(
                            np.asarray(
                                detection.get("position_3d", ()),
                                dtype=np.float32,
                            )
                            - reference
                        )
                    ),
                    detection,
                )
                for detection in cup_detections
                if len(detection.get("position_3d", ())) == 3
            ]
            nearby = [
                item
                for item in nearby
                if item[0] <= float(config.max_position_spread_m)
            ]
            if nearby:
                candidates = [min(nearby, key=lambda item: item[0])[1]]
                track_reassociated = True
    if not candidates:
        return None
    detection = max(candidates, key=lambda item: float(item.get("score", 0.0)))
    observation: dict[str, Any] = {
        "step": int(step),
        "camera_xzyaw": [float(value) for value in camera_signature],
        "position_3d": [
            float(value) for value in detection.get("position_3d", [])
        ],
        "primary_label": str(detection.get("label", "")),
        "primary_score": float(detection.get("score", 0.0)),
        "source_online_track_id": int(
            detection.get("online_track_id", -1)
        ),
        "track_reassociated": track_reassociated,
        "primary_box": [
            float(value) for value in detection.get("box", [])
        ],
        "depth_median": float(detection.get("depth_median", 0.0)),
        "depth_valid_ratio": float(detection.get("depth_valid_ratio", 0.0)),
        "crop_verifier_mode": str(mode),
        "crop_verifier_status": "pending",
        "crop_verifier_pass": False,
        "crop_positive_score": 0.0,
        "crop_negative_score": 0.0,
    }
    depth_relief = estimate_depth_surface_relief(
        depth,
        observation["primary_box"],
    )
    observation["depth_surface_relief_m"] = depth_relief.get("relief_m")
    observation["depth_surface_relief"] = depth_relief
    if not append_independent_observation(
        track_observations,
        observation,
        config,
    ):
        return None

    if str(mode) != "grounding_crop":
        observation["crop_verifier_status"] = "skipped"
        return observation

    try:
        crop_path = (
            crops_dir
            / f"track_{int(track_id):04d}_step_{int(step):04d}.jpg"
        )
        crop_bounds = _save_confirmation_crop(
            image=rgb,
            box=observation["primary_box"],
            output_path=crop_path,
            padding_ratio=crop_padding_ratio,
        )
        observation["crop_path"] = str(crop_path)
        observation["crop_bounds"] = crop_bounds
        target_box = [
            float(observation["primary_box"][0]) - float(crop_bounds[0]),
            float(observation["primary_box"][1]) - float(crop_bounds[1]),
            float(observation["primary_box"][2]) - float(crop_bounds[0]),
            float(observation["primary_box"][3]) - float(crop_bounds[1]),
        ]
        observation["crop_target_box"] = target_box
        response = worker.infer(
            f"cup-confirm-{int(step)}-{int(track_id)}",
            crop_path,
            labels=verifier_labels,
            box_threshold=verifier_box_threshold,
            text_threshold=verifier_text_threshold,
            max_detections=max(8, len(verifier_labels) * 2),
        )
        verifier_result = score_crop_verifier(
            detections=list(response.get("detections", [])),
            positive_labels=verifier_positive_labels,
            target_box=target_box,
            min_positive_score=verifier_min_positive_score,
            min_score_margin=verifier_min_score_margin,
        )
        observation.update(verifier_result)
        observation["crop_verifier_inference_ms"] = float(
            response.get("inference_ms", 0.0)
        )
    except Exception as exc:
        observation["crop_verifier_status"] = "error"
        observation["crop_verifier_error"] = f"{type(exc).__name__}: {exc}"
    return observation


def _save_confirmation_crop(
    *,
    image: Image.Image,
    box: list[float],
    output_path: Path,
    padding_ratio: float,
) -> list[int]:
    if len(box) != 4:
        raise ValueError("confirmation crop requires a four-value box")
    x1, y1, x2, y2 = [float(value) for value in box]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = max(0.0, float(padding_ratio)) * width
    pad_y = max(0.0, float(padding_ratio)) * height
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    crop_width = max(width + 2.0 * pad_x, 96.0)
    crop_height = max(height + 2.0 * pad_y, 96.0)
    bounds = [
        max(0, int(math.floor(center_x - 0.5 * crop_width))),
        max(0, int(math.floor(center_y - 0.5 * crop_height))),
        min(image.width, int(math.ceil(center_x + 0.5 * crop_width))),
        min(image.height, int(math.ceil(center_y + 0.5 * crop_height))),
    ]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("confirmation crop is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(tuple(bounds)).save(output_path, quality=95)
    return bounds


def _cup_confirmation_evaluations(
    observations: dict[int, list[dict[str, Any]]],
    config: CupConfirmationConfig,
    terminal_statuses: dict[int, str] | None = None,
    attempts: dict[int, int] | None = None,
) -> dict[int, dict[str, Any]]:
    track_ids = set(int(track_id) for track_id in observations)
    track_ids.update((terminal_statuses or {}).keys())
    results: dict[int, dict[str, Any]] = {}
    for track_id in sorted(track_ids):
        result = evaluate_cup_confirmation(
            observations.get(track_id, []),
            config,
        )
        terminal_status = (terminal_statuses or {}).get(track_id)
        if terminal_status is not None:
            result["terminal_status"] = terminal_status
            result["status"] = terminal_status
            result["verified"] = terminal_status == "verified"
        result["attempts"] = int((attempts or {}).get(track_id, 0))
        results[track_id] = result
    return results


def _prepare_scene_vlm_keyframes(
    frames: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    candidate_landmarks: list[dict[str, Any]],
    out_dir: Path,
    max_images: int,
) -> list[dict[str, Any]]:
    candidate_ids = {
        str(candidate["id"])
        for candidate in candidate_landmarks
    }
    selected = select_scene_keyframes(
        frames=frames,
        detections=detections,
        candidate_ids=candidate_ids,
        max_images=max_images,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for keyframe in selected:
        frame_index = int(keyframe["frame_index"])
        source_path = Path(keyframe["rgb_path"]).expanduser().resolve()
        image = Image.open(source_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        frame_detections = [
            detection
            for detection in detections
            if int(detection.get("frame_index", -1)) == frame_index
            and detection.get("online_track_id") is not None
            and f"track_{int(detection['online_track_id'])}"
            in candidate_ids
        ]
        for detection in frame_detections:
            box = detection.get("box") or []
            if len(box) < 4:
                continue
            candidate_id = f"track_{int(detection['online_track_id'])}"
            label = str(
                detection.get("canonical_label")
                or detection.get("label")
                or "candidate"
            )
            score = float(detection.get("score", 0.0))
            x1, y1, x2, y2 = [float(value) for value in box[:4]]
            color = "#00a6d6" if label == "cup" else "#f28e2b"
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            text = f"{candidate_id} {label} {score:.2f}"
            text_box = draw.textbbox((x1, max(0.0, y1 - 18.0)), text, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text(
                (x1, max(0.0, y1 - 18.0)),
                text,
                fill="white",
                font=font,
            )
        header = (
            f"{keyframe['frame_id']} | "
            f"{', '.join(keyframe.get('visible_candidate_ids', [])) or 'scene context'}"
        )
        header_box = draw.textbbox((4, 4), header, font=font)
        draw.rectangle(
            (0, 0, header_box[2] + 8, header_box[3] + 8),
            fill="#111820",
        )
        draw.text((4, 4), header, fill="white", font=font)
        annotated_path = out_dir / f"{keyframe['frame_id']}_annotated.jpg"
        image.save(annotated_path, quality=90)
        result.append(
            {
                **keyframe,
                "source_rgb_path": str(source_path),
                "rgb_path": str(annotated_path),
            }
        )
    return result


def _live_task_candidates(
    seed_candidates: list[dict[str, Any]],
    tracks: list[OnlineTrack],
    memory_items: list[dict[str, Any]],
    candidate_ids: list[str],
    current_xz: tuple[float, float],
) -> list[dict[str, Any]]:
    seed_by_id = {
        str(candidate["id"]): dict(candidate)
        for candidate in seed_candidates
        if candidate.get("id") is not None
    }
    track_by_id = {
        f"track_{track.track_id}": track
        for track in tracks
    }
    memory_by_id = {
        f"track_{int(item['semantic_id'])}": item
        for item in memory_items
        if item.get("semantic_id") is not None
    }
    candidates = []
    for candidate_id in candidate_ids:
        track = track_by_id.get(str(candidate_id))
        if track is None:
            continue
        candidate = dict(seed_by_id.get(str(candidate_id), {}))
        memory_item = memory_by_id.get(str(candidate_id), {})
        world_xz = [
            float(track.position[0]),
            float(track.position[2]),
        ]
        candidate.update(
            {
                "id": str(candidate_id),
                "track_id": track.track_id,
                "kind": (
                    "target_object"
                    if track.label == "cup"
                    else "support_surface"
                ),
                "label": track.label,
                "world_xz": world_xz,
                "confidence": float(
                    memory_item.get("confidence", track.confidence)
                ),
                "freshness": float(memory_item.get("freshness", 1.0)),
                "status": str(memory_item.get("status", "active")),
                "independent_views": len(track.independent_views),
                "negative_evidence_count": int(
                    memory_item.get("negative_evidence_count", 0)
                ),
                "distance_m": math.hypot(
                    world_xz[0] - float(current_xz[0]),
                    world_xz[1] - float(current_xz[1]),
                ),
            }
        )
        candidates.append(candidate)
    return candidates


def _rerank_frontiers_for_task(
    ranked_frontiers: list[dict[str, Any]],
    mapper: DenseBEVMapper,
    candidate_landmarks: list[dict[str, Any]],
    scene_understanding: dict[str, Any] | None,
    excluded_candidate_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    assessments = {
        str(item.get("candidate_id")): item
        for item in (scene_understanding or {}).get(
            "candidate_assessments",
            [],
        )
    }
    semantic_anchors = []
    excluded = {
        str(value)
        for value in (excluded_candidate_ids or set())
    }
    for candidate in candidate_landmarks:
        if candidate.get("kind") != "support_surface":
            continue
        if str(candidate.get("id")) in excluded:
            continue
        world_xz = candidate.get("world_xz") or []
        if len(world_xz) < 2:
            continue
        assessment = assessments.get(str(candidate.get("id")), {})
        likelihood = float(
            assessment.get("target_likelihood", 0.45)
        )
        visual_confidence = float(
            assessment.get("visual_confidence", 0.0)
        )
        anchor_weight = max(
            0.0,
            min(1.0, likelihood),
        ) * (0.65 + 0.35 * max(0.0, min(1.0, visual_confidence)))
        semantic_anchors.append(
            (
                np.asarray(world_xz[:2], dtype=np.float32),
                anchor_weight,
                str(candidate.get("id")),
            )
        )
    if not semantic_anchors:
        return ranked_frontiers

    enriched = []
    for frontier in ranked_frontiers:
        item = dict(frontier)
        frontier_world = np.asarray(
            mapper.grid_to_world(
                tuple(int(value) for value in frontier["cell"])
            ),
            dtype=np.float32,
        )
        best_anchor = max(
            (
                (
                    weight
                    * math.exp(
                        -float(np.linalg.norm(frontier_world - position))
                        / 3.0
                    ),
                    candidate_id,
                )
                for position, weight, candidate_id in semantic_anchors
            ),
            default=(0.0, None),
        )
        item["geometric_frontier_score"] = float(frontier.get("score", 0.0))
        item["semantic_search_bias"] = round(float(best_anchor[0]), 4)
        item["semantic_anchor_candidate_id"] = best_anchor[1]
        item["score"] = (
            float(frontier.get("score", 0.0))
            + 0.55 * float(best_anchor[0])
        )
        enriched.append(item)
    enriched.sort(
        key=lambda item: (
            -float(item["score"]),
            float(item.get("distance_m", 0.0)),
        )
    )
    return enriched


def _stable_task_tracks(
    tracks: list[OnlineTrack],
    cup_min_views: int,
    cup_min_confidence: float,
    surface_min_views: int,
    surface_min_confidence: float,
) -> list[OnlineTrack]:
    stable = []
    for track in tracks:
        if track.label == "cup":
            if (
                len(track.independent_views) >= int(cup_min_views)
                and track.confidence >= float(cup_min_confidence)
            ):
                stable.append(track)
        elif track.label in SURFACE_LABELS:
            if (
                len(track.independent_views) >= int(surface_min_views)
                and track.confidence >= float(surface_min_confidence)
            ):
                stable.append(track)
    return stable


def _append_new_task_candidates(
    tracks: list[OnlineTrack],
    ordered_candidate_ids: list[str],
    inspected_cup_track_ids: set[int],
    scanned_surface_ids: set[int],
    failed_surface_ids: set[int],
    cup_min_views: int,
    cup_min_confidence: float,
    surface_min_views: int,
    surface_min_confidence: float,
    dynamic_cup_merge_radius_m: float,
) -> list[str]:
    existing = set(ordered_candidate_ids)
    track_by_candidate_id = {
        f"track_{track.track_id}": track
        for track in tracks
    }
    planned_cup_positions = [
        track_by_candidate_id[candidate_id].position[[0, 2]]
        for candidate_id in ordered_candidate_ids
        if candidate_id in track_by_candidate_id
        and track_by_candidate_id[candidate_id].label == "cup"
    ]
    new_tracks = [
        track
        for track in _stable_task_tracks(
            tracks,
            cup_min_views=cup_min_views,
            cup_min_confidence=cup_min_confidence,
            surface_min_views=surface_min_views,
            surface_min_confidence=surface_min_confidence,
        )
        if f"track_{track.track_id}" not in existing
        and track.label == "cup"
        and track.track_id not in inspected_cup_track_ids
        and all(
            float(
                np.linalg.norm(
                    track.position[[0, 2]]
                    - np.asarray(position, dtype=np.float32)
                )
            )
            >= float(dynamic_cup_merge_radius_m)
            for position in planned_cup_positions
        )
    ]
    new_tracks.sort(
        key=lambda track: (
            track.label == "cup",
            track.confidence,
            len(track.independent_views),
        ),
        reverse=True,
    )
    selected_tracks: list[OnlineTrack] = []
    for track in new_tracks:
        if any(
            float(
                np.linalg.norm(
                    track.position[[0, 2]]
                    - existing_track.position[[0, 2]]
                )
            )
            < float(dynamic_cup_merge_radius_m)
            for existing_track in selected_tracks
        ):
            continue
        selected_tracks.append(track)
    new_ids = [f"track_{track.track_id}" for track in selected_tracks]
    first_support_index = next(
        (
            index
            for index, candidate_id in enumerate(ordered_candidate_ids)
            if candidate_id in track_by_candidate_id
            and track_by_candidate_id[candidate_id].label in SURFACE_LABELS
        ),
        len(ordered_candidate_ids),
    )
    ordered_candidate_ids[first_support_index:first_support_index] = new_ids
    return new_ids


def _choose_planned_task_target(
    tracks: list[OnlineTrack],
    ordered_candidate_ids: list[str],
    inspected_cup_track_ids: set[int],
    scanned_surface_ids: set[int],
    failed_surface_ids: set[int],
    cup_min_views: int,
    cup_min_confidence: float,
    surface_min_views: int,
    surface_min_confidence: float,
) -> dict[str, Any] | None:
    track_by_id = {f"track_{track.track_id}": track for track in tracks}
    for candidate_id in ordered_candidate_ids:
        track = track_by_id.get(str(candidate_id))
        if track is None:
            continue
        if track.label == "cup":
            if (
                track.track_id in inspected_cup_track_ids
                or len(track.independent_views) < int(cup_min_views)
                or track.confidence < float(cup_min_confidence)
            ):
                continue
            return track.as_interest_dict()
        if track.label in SURFACE_LABELS:
            if (
                track.track_id in scanned_surface_ids
                or track.track_id in failed_surface_ids
                or len(track.independent_views) < int(surface_min_views)
                or track.confidence < float(surface_min_confidence)
            ):
                continue
            return track.as_interest_dict()
    return None


def _choose_cup_target(
    tracks: list[OnlineTrack],
    current_xz: tuple[float, float],
    inspected_ids: set[int],
    min_views: int,
    min_confidence: float,
) -> dict[str, Any] | None:
    current = np.asarray(current_xz, dtype=np.float32)
    ranked: list[tuple[float, float, OnlineTrack]] = []
    for track in tracks:
        if (
            track.label != "cup"
            or track.track_id in inspected_ids
            or len(track.independent_views) < int(min_views)
            or track.confidence < float(min_confidence)
        ):
            continue
        distance_m = float(np.linalg.norm(track.position[[0, 2]] - current))
        stability = min(1.0, len(track.independent_views) / max(1.0, float(min_views)))
        score = 2.0 * track.confidence + 0.8 * stability - 0.12 * distance_m
        ranked.append((score, distance_m, track))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1], item[2].track_id))
    return ranked[0][2].as_interest_dict()


def _save_online_checkpoint(
    checkpoints_dir: Path,
    name: str,
    step: int,
    phase: str,
    mapper: DenseBEVMapper,
    memory: ObjectMemoryStore,
    tracks: list[OnlineTrack],
    current_position: np.ndarray,
    current_yaw: float,
) -> dict[str, Any]:
    checkpoint_dir = checkpoints_dir / str(name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bev_state_path = checkpoint_dir / "bev_state.npz"
    np.savez_compressed(
        bev_state_path,
        occupancy_logodds=mapper.occupancy_logodds,
        explored=mapper.explored,
        observation_count=mapper.observation_count,
        origin_world_xz=np.asarray(mapper.origin_world_xz, dtype=np.float32),
        resolution=np.asarray([mapper.config.resolution], dtype=np.float32),
    )
    memory_path = checkpoint_dir / "object_memory.json"
    memory.save(memory_path)
    tracks_path = checkpoint_dir / "tracks.json"
    _write_json(
        tracks_path,
        {"items": [track.to_dict() for track in tracks]},
    )
    bev_png = checkpoint_dir / "bev_memory.png"
    _render_online_bev(
        mapper,
        tracks=tracks,
        current_position=np.asarray(current_position, dtype=np.float32),
        current_yaw=float(current_yaw),
        target_world_xz=None,
    ).save(bev_png)
    manifest = {
        "name": str(name),
        "step": int(step),
        "phase": str(phase),
        "agent_position_xyz": [
            float(value) for value in np.asarray(current_position).reshape(3)
        ],
        "agent_yaw_deg": float(current_yaw),
        "bev": mapper.snapshot(),
        "memory": memory.summary(),
        "num_tracks": len(tracks),
        "artifacts": {
            "bev_state": str(bev_state_path),
            "bev_png": str(bev_png),
            "object_memory": str(memory_path),
            "tracks": str(tracks_path),
        },
    }
    manifest_path = checkpoint_dir / "checkpoint.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl_event(file_handle: Any, payload: dict[str, Any]) -> None:
    file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    file_handle.flush()


def _timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "capture",
        "mapping",
        "grounding",
        "confirmation_verification",
        "memory",
        "policy",
        "visualization",
        "action",
        "total",
    ]
    summary = {}
    for key in keys:
        values = np.asarray([record["timing_ms"][key] for record in records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()) if values.size else 0.0,
            "p95": float(np.percentile(values, 95)) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
        }
    return summary


def _is_independent_view(
    signatures: list[tuple[float, float, float]],
    candidate: tuple[float, float, float],
) -> bool:
    if not signatures:
        return True
    for x, z, yaw in signatures:
        translation = math.hypot(candidate[0] - x, candidate[1] - z)
        rotation = _angle_delta_deg(candidate[2], yaw)
        if translation < 0.35 and rotation < 20.0:
            return False
    return True


def _canonical_label(label: str) -> str:
    normalized = str(label).strip().lower()
    return "cup" if normalized in CUP_LABELS else normalized


def _comma_separated_labels(value: str) -> list[str]:
    labels = [
        item.strip().lower()
        for item in str(value).split(",")
        if item.strip()
    ]
    if not labels:
        raise ValueError("at least one verifier label is required")
    return labels


def _calibrated_grounding_confidence(score: float) -> float:
    return float(np.clip(0.45 + (float(score) - 0.22) * 3.0, 0.35, 0.95))


def _near_any_scanned_surface(
    position_xz: np.ndarray,
    scanned_positions: list[np.ndarray],
    radius_m: float,
) -> bool:
    position_xz = np.asarray(position_xz, dtype=np.float32).reshape(2)
    return any(
        float(np.linalg.norm(position_xz - np.asarray(scanned, dtype=np.float32).reshape(2)))
        <= float(radius_m)
        for scanned in scanned_positions
    )


def _without_blacklisted_frontiers(
    ranked: list[dict[str, Any]],
    blacklisted: list[tuple[int, int]],
    radius_cells: int,
) -> list[dict[str, Any]]:
    if not blacklisted:
        return ranked
    radius = max(1, int(radius_cells))
    blocked = [np.asarray(cell, dtype=np.float32) for cell in blacklisted]
    return [
        item
        for item in ranked
        if all(
            float(np.linalg.norm(np.asarray(item["cell"], dtype=np.float32) - cell)) > radius
            for cell in blocked
        )
    ]


def _render_online_bev(
    mapper: DenseBEVMapper,
    tracks: list[OnlineTrack],
    current_position: np.ndarray,
    current_yaw: float,
    target_world_xz: Any,
) -> Image.Image:
    state = np.flipud(mapper.occupancy_state().T)
    colors = np.asarray(
        [
            [214, 218, 221],
            [249, 249, 247],
            [42, 47, 52],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(colors[state], mode="RGB")
    draw = ImageDraw.Draw(image)
    height = image.height

    trajectory = [
        (int(cell[0]), int(height - 1 - cell[1]))
        for cell in mapper.trajectory
    ]
    if len(trajectory) >= 2:
        draw.line(trajectory, fill=(38, 116, 164), width=3)

    track_colors = {
        "cup": (205, 62, 67),
        "table": (37, 145, 99),
        "counter": (221, 144, 35),
        "sink": (45, 117, 182),
        "bottle": (139, 92, 246),
    }
    for track in tracks:
        stable = (
            len(track.independent_views) >= (5 if track.label == "cup" else 6)
            and track.confidence >= (0.28 if track.label == "cup" else 0.26)
        )
        if not stable:
            continue
        cell = mapper.world_to_grid((float(track.position[0]), float(track.position[2])))
        if cell is None:
            continue
        x, y = int(cell[0]), int(height - 1 - cell[1])
        radius = 5 if track.label == "cup" else 3
        color = track_colors.get(track.label, (92, 102, 112))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))

    if target_world_xz is not None:
        target = mapper.world_to_grid((float(target_world_xz[0]), float(target_world_xz[1])))
        if target is not None:
            tx, ty = int(target[0]), int(height - 1 - target[1])
            draw.rectangle((tx - 6, ty - 6, tx + 6, ty + 6), outline=(240, 183, 47), width=3)

    current = mapper.world_to_grid((float(current_position[0]), float(current_position[2])))
    if current is not None:
        x, y = int(current[0]), int(height - 1 - current[1])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(15, 105, 170), outline=(255, 255, 255), width=2)
        length = 15.0
        dx = -math.sin(math.radians(current_yaw)) * length
        dz = -math.cos(math.radians(current_yaw)) * length
        draw.line((x, y, x + dx, y - dz), fill=(15, 105, 170), width=4)
    return image


def _clearance(values: np.ndarray) -> float:
    valid = values[np.isfinite(values) & (values > 0.05)]
    return float(np.percentile(valid, 40)) if valid.size else 0.0


def _yaw_from_matrix(matrix: Any) -> float:
    rotation = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    forward = -(rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    return math.degrees(math.atan2(float(-forward[0]), float(-forward[2])))


def _signed_angle_delta_deg(target: float, source: float) -> float:
    return (float(target) - float(source) + 180.0) % 360.0 - 180.0


def _angle_delta_deg(a: float, b: float) -> float:
    return abs(_signed_angle_delta_deg(a, b))


def _rotation_matrix(rotation: Any) -> list[list[float]]:
    if hasattr(rotation, "transform_vector"):
        axes = [
            np.asarray(rotation.transform_vector([1.0, 0.0, 0.0]), dtype=np.float32),
            np.asarray(rotation.transform_vector([0.0, 1.0, 0.0]), dtype=np.float32),
            np.asarray(rotation.transform_vector([0.0, 0.0, 1.0]), dtype=np.float32),
        ]
        matrix = np.stack(axes, axis=1)
    else:
        import quaternion as np_quaternion

        matrix = np.asarray(np_quaternion.as_rotation_matrix(rotation), dtype=np.float32)
    return [[float(value) for value in row] for row in matrix]


def _list3(value: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(array[0]), float(array[1]), float(array[2])]


if __name__ == "__main__":
    main()
