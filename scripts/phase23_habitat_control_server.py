from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_bev_mapper import DenseBEVConfig, DenseBEVMapper, mapping_metrics, oracle_navmesh_mask
from object_memory_store import ObjectMemoryStore
from semantic_bev_memory import (
    SEMANTIC_COLORS,
    SemanticBEVAccumulator,
    _camera_axes,
    _object_visibility_points,
    _patch_depth,
    _project_world_point,
    semantic_array,
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RSC-Nav Habitat Control</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181b1f;
      --ink: #e7edf2;
      --muted: #9aa7b2;
      --accent: #61c6a7;
      --line: #2a3036;
      --bad: #ec7d7d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #0d0f11;
    }
    h1 { font-size: 16px; margin: 0; font-weight: 650; letter-spacing: 0; }
    #status { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) minmax(280px, 1fr) 300px;
      align-items: start;
      gap: 12px;
      padding: 12px;
    }
    section {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    section h2 {
      height: 34px;
      display: flex;
      align-items: center;
      margin: 0;
      padding: 0 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }
    .media {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      display: block;
      background: #050607;
    }
    .media-wrap {
      position: relative;
      width: 100%;
      background: #050607;
    }
    .media-wrap .media {
      height: auto;
      aspect-ratio: auto;
    }
    #box-layer {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }
    #bev-panel { grid-column: 1 / span 2; }
    #bev { image-rendering: auto; }
    aside {
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-width: 0;
    }
    .metrics {
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      font-size: 13px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-height: 52px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }
    .metric strong {
      font-size: 15px;
      overflow-wrap: anywhere;
    }
    .controls {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    button {
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #20252b;
      color: var(--ink);
      font-size: 15px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button.primary { background: #1f3b34; color: #d7fff1; }
    .wide { grid-column: span 3; }
    .hint {
      padding: 0 12px 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .error { color: var(--bad); }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      #bev-panel { grid-column: auto; }
      aside { display: grid; grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>RSC-Nav Habitat Control</h1>
    <div id="status">connecting</div>
  </header>
  <main>
    <section>
      <h2>RGB + Grounding</h2>
      <div class="media-wrap">
        <img id="rgb" class="media" alt="RGB view" />
        <canvas id="box-layer"></canvas>
      </div>
    </section>
    <section>
      <h2>Depth</h2>
      <img id="depth" class="media" alt="Depth view" />
    </section>
    <aside>
      <section id="semantic-panel" style="display:none">
        <h2>Semantic BEV</h2>
        <img id="semantic" class="media" alt="Semantic BEV map" />
      </section>
      <section>
        <h2>State</h2>
        <div class="metrics">
          <div class="metric"><span>Pose</span><strong id="pose">-</strong></div>
          <div class="metric"><span>Heading</span><strong id="heading">-</strong></div>
          <div class="metric"><span>Rays</span><strong id="rays">-</strong></div>
          <div class="metric"><span>Explored</span><strong id="explored">-</strong></div>
          <div class="metric"><span>Occupied</span><strong id="occupied">-</strong></div>
          <div class="metric"><span>Step</span><strong id="step">-</strong></div>
          <div class="metric"><span>Objects</span><strong id="objects">-</strong></div>
          <div class="metric"><span>Freshness</span><strong id="freshness">-</strong></div>
        </div>
        <div class="controls">
          <button onclick="sendAction('turn_left')">A</button>
          <button class="primary" onclick="sendAction('move_forward')">W</button>
          <button onclick="sendAction('turn_right')">D</button>
          <button onclick="sendAction('look_up')">Look Up</button>
          <button onclick="sendAction('move_back')">S</button>
          <button onclick="sendAction('look_down')">Look Down</button>
          <button class="wide" onclick="resetSim()">Reset</button>
          <button id="auto-btn" class="wide" onclick="toggleAuto()">Auto</button>
          <button onclick="sendAction('path_step')">Step</button>
          <button onclick="saveMemory()">Save</button>
          <button onclick="loadMemory()">Load</button>
          <button onclick="getState()">Refresh</button>
        </div>
        <div class="hint" id="memory">Keyboard: W forward, A/D turn, S back, Q/E look, R reset.</div>
      </section>
    </aside>
    <section id="bev-panel">
      <h2>BEV Memory</h2>
      <img id="bev" class="media" alt="BEV map" />
    </section>
  </main>
  <script>
    let busy = false;
    let autoTimer = null;
    let lastBoxes = [];
    const statusEl = document.getElementById("status");
    const rgbEl = document.getElementById("rgb");
    const boxLayer = document.getElementById("box-layer");

    function setStatus(text, bad=false) {
      statusEl.textContent = text;
      statusEl.className = bad ? "error" : "";
    }

    function drawBoxes() {
      const boxes = lastBoxes || [];
      if (!rgbEl.complete || !rgbEl.naturalWidth || !rgbEl.naturalHeight) return;
      boxLayer.width = rgbEl.naturalWidth;
      boxLayer.height = rgbEl.naturalHeight;
      const ctx = boxLayer.getContext("2d");
      ctx.clearRect(0, 0, boxLayer.width, boxLayer.height);
      ctx.lineWidth = Math.max(2, Math.round(boxLayer.width / 180));
      ctx.font = `${Math.max(11, Math.round(boxLayer.width / 18))}px ui-sans-serif, system-ui`;
      for (const box of boxes) {
        const x = Math.max(0, box.x);
        const y = Math.max(0, box.y);
        const w = Math.min(boxLayer.width - x, box.w);
        const h = Math.min(boxLayer.height - y, box.h);
        if (w <= 2 || h <= 2) continue;
        const color = box.color || "#61c6a7";
        const label = `${box.category} ${(box.score || 0).toFixed(2)}`;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.strokeRect(x, y, w, h);
        const metrics = ctx.measureText(label);
        const labelW = Math.ceil(metrics.width) + 8;
        const labelH = Math.max(16, Math.round(boxLayer.width / 14));
        const labelY = Math.max(0, y - labelH);
        ctx.fillRect(x, labelY, labelW, labelH);
        ctx.fillStyle = "#07100d";
        ctx.fillText(label, x + 4, labelY + labelH - 5);
      }
    }

    function applyState(data) {
      lastBoxes = data.grounding_boxes || [];
      rgbEl.onload = drawBoxes;
      rgbEl.src = "data:image/jpeg;base64," + data.rgb_jpeg;
      document.getElementById("depth").src = "data:image/png;base64," + data.depth_png;
      document.getElementById("bev").src = "data:image/png;base64," + data.bev_png;
      const semanticPanel = document.getElementById("semantic-panel");
      if (data.semantic_png) {
        document.getElementById("semantic").src = "data:image/png;base64," + data.semantic_png;
        semanticPanel.style.display = "";
      } else {
        semanticPanel.style.display = "none";
      }
      document.getElementById("pose").textContent = `${data.pose.x.toFixed(2)}, ${data.pose.y.toFixed(2)}`;
      document.getElementById("heading").textContent = `${data.pose.heading_deg.toFixed(1)} deg`;
      document.getElementById("rays").textContent = data.ray_count;
      document.getElementById("explored").textContent = data.bev.num_explored_cells;
      document.getElementById("occupied").textContent = data.bev.num_occupied_cells;
      document.getElementById("step").textContent = data.step;
      const memory = data.memory || {};
      document.getElementById("objects").textContent = `${memory.num_items || 0} (${memory.active_items || 0} active)`;
      document.getElementById("freshness").textContent = (memory.mean_freshness || 0).toFixed(3);
      const classes = memory.per_class ? Object.entries(memory.per_class).map(([k, v]) => `${k}:${v}`).join(", ") : "-";
      document.getElementById("memory").textContent = `Grounding: ${lastBoxes.length}. Memory: ${classes}. Keyboard: W forward, A/D turn, S back, Q/E look, R reset.`;
      setStatus(`ready: ${data.scene_name} / boxes ${lastBoxes.length}`);
      drawBoxes();
    }

    async function getState() {
      const res = await fetch("/api/state");
      if (!res.ok) throw new Error(await res.text());
      applyState(await res.json());
    }

    async function sendAction(action) {
      if (busy) return;
      busy = true;
      setStatus(action);
      try {
        const res = await fetch("/api/action", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({action})
        });
        if (!res.ok) throw new Error(await res.text());
        applyState(await res.json());
      } catch (err) {
        setStatus(String(err), true);
      } finally {
        busy = false;
      }
    }

    function toggleAuto() {
      const btn = document.getElementById("auto-btn");
      if (autoTimer) {
        clearInterval(autoTimer);
        autoTimer = null;
        btn.textContent = "Auto";
        setStatus("auto stopped");
        return;
      }
      btn.textContent = "Stop";
      autoTimer = setInterval(() => {
        if (!busy) sendAction("path_step");
      }, 550);
      sendAction("path_step");
    }

    async function resetSim() {
      if (busy) return;
      busy = true;
      setStatus("reset");
      try {
        const res = await fetch("/api/reset", {method: "POST"});
        if (!res.ok) throw new Error(await res.text());
        applyState(await res.json());
      } catch (err) {
        setStatus(String(err), true);
      } finally {
        busy = false;
      }
    }

    async function saveMemory() {
      const res = await fetch("/api/save_memory", {method: "POST"});
      if (!res.ok) throw new Error(await res.text());
      applyState(await res.json());
    }

    async function loadMemory() {
      const res = await fetch("/api/load_memory", {method: "POST"});
      if (!res.ok) throw new Error(await res.text());
      applyState(await res.json());
    }

    document.addEventListener("keydown", (event) => {
      if (event.repeat || busy) return;
      const key = event.key.toLowerCase();
      if (key === "w" || event.key === "ArrowUp") sendAction("move_forward");
      if (key === "a" || event.key === "ArrowLeft") sendAction("turn_left");
      if (key === "d" || event.key === "ArrowRight") sendAction("turn_right");
      if (key === "s" || event.key === "ArrowDown") sendAction("move_back");
      if (key === "q") sendAction("look_up");
      if (key === "e") sendAction("look_down");
      if (key === "r") resetSim();
      if (key === " ") {
        event.preventDefault();
        toggleAuto();
      }
    });

    window.addEventListener("resize", drawBoxes);

    getState().catch(err => setStatus(String(err), true));
  </script>
</body>
</html>
"""


class HabitatControlSession:
    def __init__(
        self,
        scene: Path,
        resolution: int,
        move_amount: float,
        turn_amount: float,
        scene_dataset_config: Path | None = None,
        bev_resolution: float = 0.05,
        grid_size: int = 240,
        sample_stride: int = 2,
        obstacle_dilation_cells: int = 2,
        semantic_categories: list[str] | None = None,
        semantic_confidence_saturation: float = 80.0,
        freshness_tau_steps: float = 20.0,
        negative_evidence_scale: float = 1.0,
        object_memory_missing_confidence_threshold: float = 0.35,
        object_memory_missing_missed_weight_threshold: float = 6.0,
        object_memory_stale_missed_weight_threshold: float = 2.5,
        semantic_prior_decay: bool = False,
        semantic_prior_decay_scale: float = 1.0,
        memory_path: Path | None = None,
        start_path_min_distance: float = 3.0,
        start_path_samples: int = 48,
        enable_oracle_metrics: bool = True,
    ) -> None:
        self.scene = scene
        self.resolution = resolution
        self.move_amount = move_amount
        self.turn_amount = turn_amount
        self.scene_dataset_config = scene_dataset_config
        self.bev_resolution = bev_resolution
        self.grid_size = grid_size
        self.sample_stride = sample_stride
        self.obstacle_dilation_cells = obstacle_dilation_cells
        self.semantic_categories = semantic_categories or ["wall", "door", "table", "chair"]
        self.semantic_confidence_saturation = semantic_confidence_saturation
        self.freshness_tau_steps = freshness_tau_steps
        self.negative_evidence_scale = max(0.0, float(negative_evidence_scale))
        self.object_memory_missing_confidence_threshold = float(object_memory_missing_confidence_threshold)
        self.object_memory_missing_missed_weight_threshold = float(object_memory_missing_missed_weight_threshold)
        self.object_memory_stale_missed_weight_threshold = float(object_memory_stale_missed_weight_threshold)
        self.semantic_prior_decay = bool(semantic_prior_decay)
        self.semantic_prior_decay_scale = max(0.0, float(semantic_prior_decay_scale))
        self.memory_path = memory_path
        self.start_path_min_distance = start_path_min_distance
        self.start_path_samples = start_path_samples
        self.enable_oracle_metrics = enable_oracle_metrics
        self.lock = threading.Lock()
        self.step_count = 0
        self.memory_step_count = 0
        self.last_payload: dict[str, Any] | None = None
        self.memory_origin_world_xz: tuple[float, float] | None = None
        self.last_evidence_pose: dict[str, float] | None = None
        self.last_loaded_bev_transform: dict[str, Any] | None = None

        self._setup_sim()
        self._reset_agent()
        self._reset_memory()

    def close(self) -> None:
        self.sim.close()

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.step_count = 0
            self._reset_agent()
            self._reset_memory()
            self.last_payload = self._state_payload()
            return self.last_payload

    def state(self) -> dict[str, Any]:
        with self.lock:
            if self.last_payload is None:
                self.last_payload = self._state_payload()
            return self.last_payload

    def action(self, action: str) -> dict[str, Any]:
        with self.lock:
            if action in {"move_forward", "turn_left", "turn_right", "look_up", "look_down"}:
                self.sim.step(action)
            elif action == "move_back":
                self._move_back()
            elif action == "path_step":
                self._path_step()
            else:
                raise ValueError(f"Unknown action: {action}")
            self.step_count += 1
            self.memory_step_count += 1
            self.last_payload = self._state_payload()
            return self.last_payload

    def _setup_sim(self) -> None:
        import habitat_sim
        from habitat_sim import SensorSubType, SensorType

        sensor_specs = [
            self._camera_spec(habitat_sim, SensorSubType, SensorType.COLOR, "rgb"),
            self._camera_spec(habitat_sim, SensorSubType, SensorType.DEPTH, "depth"),
        ]
        try:
            sensor_specs.append(
                self._camera_spec(habitat_sim, SensorSubType, SensorType.SEMANTIC, "semantic")
            )
        except Exception:
            pass

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(self.scene)
        sim_cfg.enable_physics = False
        if self.scene_dataset_config is not None:
            sim_cfg.scene_dataset_config_file = str(self.scene_dataset_config)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=self.move_amount)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=self.turn_amount)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=self.turn_amount)
            ),
            "look_up": habitat_sim.agent.ActionSpec(
                "look_up", habitat_sim.agent.ActuationSpec(amount=self.turn_amount)
            ),
            "look_down": habitat_sim.agent.ActionSpec(
                "look_down", habitat_sim.agent.ActuationSpec(amount=self.turn_amount)
            ),
        }

        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    def _camera_spec(self, habitat_sim, SensorSubType, sensor_type, uuid: str):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = SensorSubType.PINHOLE
        spec.resolution = [self.resolution, self.resolution]
        spec.position = [0.0, 1.5, 0.0]
        return spec

    def _reset_memory(self) -> None:
        agent_state = self.sim.get_agent(0).get_state()
        pose = self._pose_from_state(agent_state)
        config = DenseBEVConfig(
            grid_size=(self.grid_size, self.grid_size),
            resolution=self.bev_resolution,
            sample_stride=self.sample_stride,
            max_depth_m=6.0,
            obstacle_dilation_radius_cells=self.obstacle_dilation_cells,
        )
        origin = self.memory_origin_world_xz or (
            pose["x"] - (config.grid_size[0] // 2) * config.resolution,
            pose["y"] - (config.grid_size[1] // 2) * config.resolution,
        )
        self.bev = DenseBEVMapper(origin_world_xz=origin, config=config)
        self.oracle_free_mask = None
        pathfinder = getattr(self.sim, "pathfinder", None)
        if self.enable_oracle_metrics and pathfinder is not None and getattr(pathfinder, "is_loaded", False):
            self.oracle_free_mask = oracle_navmesh_mask(
                pathfinder=pathfinder,
                origin_world_xz=self.bev.origin_world_xz,
                grid_size=self.bev.config.grid_size,
                resolution=self.bev.config.resolution,
                height=float(np.asarray(agent_state.position, dtype=np.float32)[1]),
            )
        self.semantic_bev = None
        self.last_evidence_pose = None
        if self.scene_dataset_config is not None:
            self.semantic_bev = SemanticBEVAccumulator(
                mapper=self.bev,
                semantic_scene=self.sim.semantic_scene,
                categories=self.semantic_categories,
                confidence_saturation=self.semantic_confidence_saturation,
                freshness_tau_steps=self.freshness_tau_steps,
            )
        self.memory_store = ObjectMemoryStore(
            scene_id=str(self.scene),
            freshness_tau_steps=self.freshness_tau_steps,
            missing_confidence_threshold=self.object_memory_missing_confidence_threshold,
            missing_missed_weight_threshold=self.object_memory_missing_missed_weight_threshold,
            stale_missed_weight_threshold=self.object_memory_stale_missed_weight_threshold,
        )

    def _reset_agent(self) -> None:
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        self.autopilot_path = _sample_navigable_path(
            self.sim,
            min_distance_m=self.start_path_min_distance,
            max_samples=self.start_path_samples,
        )
        self.autopilot_index = 0
        if self.autopilot_path:
            state.position = self.autopilot_path[0]
            rotation = _rotation_toward(self.autopilot_path[0], _next_point(self.autopilot_path, 0))
            if rotation is not None:
                state.rotation = rotation
        else:
            pathfinder = getattr(self.sim, "pathfinder", None)
            if pathfinder is not None and getattr(pathfinder, "is_loaded", False):
                point = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
                if point.shape == (3,) and np.isfinite(point).all():
                    state.position = point
        try:
            agent.set_state(state, infer_sensor_states=True)
        except TypeError:
            agent.set_state(state)

    def _path_step(self) -> None:
        if not getattr(self, "autopilot_path", None) or self.autopilot_index + 1 >= len(self.autopilot_path):
            self.autopilot_path = _sample_navigable_path(
                self.sim,
                min_distance_m=self.start_path_min_distance,
                max_samples=self.start_path_samples,
            )
            self.autopilot_index = 0
        if not self.autopilot_path:
            self.sim.step("move_forward")
            return
        self.autopilot_index = min(self.autopilot_index + 1, len(self.autopilot_path) - 1)
        self._set_agent_pose(
            self.autopilot_path[self.autopilot_index],
            _next_point(self.autopilot_path, self.autopilot_index),
        )

    def _set_agent_pose(self, position: np.ndarray, look_at) -> None:
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        state.position = np.asarray(position, dtype=np.float32)
        rotation = _rotation_toward(state.position, look_at)
        if rotation is not None:
            state.rotation = rotation
        try:
            agent.set_state(state, infer_sensor_states=True)
        except TypeError:
            agent.set_state(state)

    def _move_back(self) -> None:
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        try:
            forward = np.asarray(state.rotation.transform_vector([0.0, 0.0, -1.0]), dtype=np.float32)
        except Exception:
            forward = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        state.position = np.asarray(state.position, dtype=np.float32) - forward * self.move_amount
        agent.set_state(state)

    def _state_payload(self) -> dict[str, Any]:
        observations = self.sim.get_sensor_observations()
        rgb = _rgb_array(observations.get("rgb"))
        depth = _valid_depth(observations.get("depth"))
        semantic = observations.get("semantic") if "semantic" in observations else None
        agent_state = self.sim.get_agent(0).get_state()
        sensor_state = agent_state.sensor_states.get("depth") or next(iter(agent_state.sensor_states.values()))
        pose = self._pose_from_state(agent_state)
        pose["sensor_pitch_deg"] = _pitch_deg_from_rotation(sensor_state.rotation)
        evidence_update = self._evidence_update_from_pose(pose)

        snapshot = self.bev.update_from_depth(
            depth=depth,
            agent_position_xyz=agent_state.position,
            sensor_position_xyz=sensor_state.position,
            sensor_rotation=sensor_state.rotation,
            hfov_deg=90.0,
        )
        semantic_report = None
        observability_report = None
        current_tracks: list[dict] = []
        if self.semantic_bev is not None and semantic is not None:
            semantic_update = self.semantic_bev.update_from_observation(
                depth=depth,
                semantic=semantic_array(semantic),
                sensor_position_xyz=np.asarray(sensor_state.position, dtype=np.float32),
                sensor_rotation=sensor_state.rotation,
                floor_y=float(np.asarray(agent_state.position, dtype=np.float32)[1]),
                hfov_deg=90.0,
                step=self.memory_step_count,
                evidence_weight=evidence_update["weight"],
                prior_decay_weight=(
                    evidence_update["negative_weight"] * self.semantic_prior_decay_scale
                    if self.semantic_prior_decay
                    else 0.0
                ),
            )
            semantic_report = self.semantic_bev.report()
            semantic_report["prior_decay_update"] = semantic_update
            seen_ids = self.semantic_bev.seen_ids_for_step(self.memory_step_count)
            expected_visible_ids = self.semantic_bev.expected_visible_ids(
                depth=depth,
                sensor_position_xyz=np.asarray(sensor_state.position, dtype=np.float32),
                sensor_rotation=sensor_state.rotation,
                hfov_deg=90.0,
            )
            current_tracks = [
                track
                for track in semantic_report.get("tracks", [])
                if int(track.get("semantic_id", -1)) in seen_ids
            ]
            observability = {
                int(item.semantic_id): _observability_label(
                    semantic_id=int(item.semantic_id),
                    seen_ids=seen_ids,
                    expected_visible_ids=expected_visible_ids,
                )
                for item in self.memory_store.items.values()
                if not str(item.source).startswith("prior_")
            }
            prior_expected_visible_ids = self._prior_expected_visible_item_ids(
                depth=depth,
                sensor_position_xyz=np.asarray(sensor_state.position, dtype=np.float32),
                sensor_rotation=sensor_state.rotation,
                floor_y=float(np.asarray(agent_state.position, dtype=np.float32)[1]),
                hfov_deg=90.0,
            )
            for item in self.memory_store.items.values():
                if str(item.source).startswith("prior_"):
                    observability[item.id] = "expected_visible_miss" if item.id in prior_expected_visible_ids else "not_observable"
            self.memory_store.update_from_tracks(
                current_tracks,
                current_step=self.memory_step_count,
                observability=observability,
                evidence_weight=evidence_update["weight"],
                negative_evidence_weight=evidence_update["negative_weight"],
            )
            observability_report = {
                "positive_ids": sorted(int(value) for value in seen_ids),
                "expected_visible_ids": sorted(int(value) for value in expected_visible_ids),
                "expected_visible_miss_ids": sorted(
                    [key for key, value in observability.items() if value == "expected_visible_miss"],
                    key=str,
                ),
                "not_observable_ids": sorted(
                    [key for key, value in observability.items() if value == "not_observable"],
                    key=str,
                ),
                "prior_expected_visible_ids": sorted(prior_expected_visible_ids),
            }
        else:
            self.memory_store.decay(current_step=self.memory_step_count)
        self.last_evidence_pose = pose
        grounding_boxes = _grounding_boxes_from_tracks(
            tracks=current_tracks,
            memory_items=self.memory_store.to_dict()["items"],
            depth=depth,
            sensor_position_xyz=np.asarray(sensor_state.position, dtype=np.float32),
            sensor_rotation=sensor_state.rotation,
            hfov_deg=90.0,
        )

        return {
            "step": self.step_count,
            "memory_step": self.memory_step_count,
            "scene": str(self.scene),
            "scene_name": self.scene.name,
            "pose": pose,
            "evidence_update": evidence_update,
            "ray_count": _sample_count(depth, self.bev.config.sample_stride),
            "bev": snapshot,
            "geometry_oracle": self._geometry_oracle_payload(),
            "semantic": _semantic_payload(semantic_report),
            "observability": observability_report or {},
            "memory": self.memory_store.summary(),
            "memory_items": self.memory_store.to_dict()["items"],
            "grounding_boxes": grounding_boxes,
            "rgb_jpeg": _image_to_base64(_rgb_image(rgb), "JPEG", quality=95),
            "depth_png": _image_to_base64(_depth_image(depth), "PNG"),
            "bev_png": _image_to_base64(_bev_image(self.bev), "PNG"),
            "oracle_png": (
                _image_to_base64(_oracle_image(self.oracle_free_mask, self.bev.trajectory), "PNG")
                if self.oracle_free_mask is not None
                else None
            ),
            "oracle_diff_png": (
                _image_to_base64(_oracle_diff_image(self.bev, self.oracle_free_mask), "PNG")
                if self.oracle_free_mask is not None
                else None
            ),
            "semantic_png": (
                _image_to_base64(_semantic_bev_image(self.semantic_bev, self.bev.trajectory), "PNG")
                if self.semantic_bev is not None
                else None
            ),
        }

    def save_memory(self) -> dict[str, Any]:
        with self.lock:
            if self.memory_path is None:
                raise RuntimeError("No --memory-path was provided")
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_store.save(self.memory_path)
            if self.last_payload is None:
                self.last_payload = self._state_payload()
            payload = dict(self.last_payload)
            payload["memory_saved_path"] = str(self.memory_path)
            return payload

    def save_bev_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        semantic_evidence = (
            self.semantic_bev.combined_evidence().astype(np.float32)
            if self.semantic_bev is not None
            else np.zeros((0, *self.bev.config.grid_size), dtype=np.float32)
        )
        live_semantic_evidence = (
            self.semantic_bev.evidence.astype(np.float32)
            if self.semantic_bev is not None
            else np.zeros((0, *self.bev.config.grid_size), dtype=np.float32)
        )
        prior_semantic_evidence = (
            self.semantic_bev.prior_evidence.astype(np.float32)
            if self.semantic_bev is not None
            else np.zeros((0, *self.bev.config.grid_size), dtype=np.float32)
        )
        metadata = {
            "scene": str(self.scene),
            "grid_size": list(self.bev.config.grid_size),
            "resolution": float(self.bev.config.resolution),
            "origin_world_xz": list(self.bev.origin_world_xz),
            "semantic_categories": list(self.semantic_categories),
            "semantic_evidence_shape": list(semantic_evidence.shape),
        }
        trajectory = np.asarray(self.bev.trajectory, dtype=np.int32) if self.bev.trajectory else np.empty((0, 2), dtype=np.int32)
        np.savez_compressed(
            path,
            metadata=json.dumps(metadata),
            occupancy_logodds=self.bev.occupancy_logodds.astype(np.float32),
            explored=self.bev.explored.astype(np.uint8),
            observation_count=self.bev.observation_count.astype(np.int32),
            trajectory=trajectory,
            semantic_evidence=semantic_evidence,
            live_semantic_evidence=live_semantic_evidence,
            prior_semantic_evidence=prior_semantic_evidence,
        )

    def load_bev_state(
        self,
        path: str | Path,
        load_semantic_evidence: bool = True,
        align: str = "source",
        keep_trajectory: bool = True,
        load_semantic_as_prior: bool = False,
    ) -> dict[str, Any]:
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"]))
        occupancy = np.asarray(data["occupancy_logodds"], dtype=np.float32)
        explored = np.asarray(data["explored"], dtype=np.uint8).astype(bool)
        observation_count = np.asarray(data["observation_count"], dtype=np.int32)
        trajectory = np.asarray(data["trajectory"], dtype=np.int32)
        if align not in {"source", "center"}:
            raise ValueError(f"Unsupported BEV load alignment: {align!r}")

        loaded_origin = tuple(float(value) for value in metadata["origin_world_xz"])
        current_grid_size = tuple(int(value) for value in self.bev.config.grid_size)
        current_origin = tuple(float(value) for value in self.bev.origin_world_xz)
        resolution = float(metadata.get("resolution", self.bev_resolution))
        paste_offset = (0, 0)

        if align == "center":
            target_shape = (
                max(int(current_grid_size[0]), int(occupancy.shape[0])),
                max(int(current_grid_size[1]), int(occupancy.shape[1])),
            )
            current_center = (
                current_origin[0] + 0.5 * current_grid_size[0] * self.bev.config.resolution,
                current_origin[1] + 0.5 * current_grid_size[1] * self.bev.config.resolution,
            )
            target_origin = (
                current_center[0] - 0.5 * target_shape[0] * resolution,
                current_center[1] - 0.5 * target_shape[1] * resolution,
            )
            paste_offset = (
                (target_shape[0] - int(occupancy.shape[0])) // 2,
                (target_shape[1] - int(occupancy.shape[1])) // 2,
            )
            target_occupancy = np.zeros(target_shape, dtype=np.float32)
            target_explored = np.zeros(target_shape, dtype=bool)
            target_observation_count = np.zeros(target_shape, dtype=np.int32)
            ox, oy = paste_offset
            sx, sy = occupancy.shape
            target_occupancy[ox : ox + sx, oy : oy + sy] = occupancy
            target_explored[ox : ox + sx, oy : oy + sy] = explored
            target_observation_count[ox : ox + sx, oy : oy + sy] = observation_count
            occupancy = target_occupancy
            explored = target_explored
            observation_count = target_observation_count
        else:
            target_shape = tuple(int(value) for value in occupancy.shape)
            target_origin = loaded_origin

        if tuple(occupancy.shape) != tuple(self.bev.config.grid_size) or align == "center":
            config = DenseBEVConfig(
                grid_size=tuple(int(value) for value in occupancy.shape),
                resolution=resolution,
                sample_stride=self.sample_stride,
                max_depth_m=6.0,
                obstacle_dilation_radius_cells=self.obstacle_dilation_cells,
            )
            self.grid_size = int(config.grid_size[0])
            self.bev_resolution = float(config.resolution)
            self.bev = DenseBEVMapper(
                origin_world_xz=target_origin,
                config=config,
            )
            if self.semantic_bev is not None:
                self.semantic_bev.mapper = self.bev
                if tuple(self.semantic_bev.evidence.shape[1:]) != tuple(self.bev.config.grid_size):
                    self.semantic_bev.evidence = np.zeros(
                        (len(self.semantic_bev.categories), *self.bev.config.grid_size),
                        dtype=np.float32,
                    )
                    self.semantic_bev.prior_evidence = np.zeros_like(self.semantic_bev.evidence)
        else:
            self.bev.origin_world_xz = target_origin

        self.bev.occupancy_logodds = occupancy.copy()
        self.bev.explored = explored.copy()
        self.bev.observation_count = observation_count.copy()
        if keep_trajectory:
            if align == "center" and paste_offset != (0, 0) and trajectory.size:
                trajectory = trajectory.copy()
                trajectory[:, 0] += int(paste_offset[0])
                trajectory[:, 1] += int(paste_offset[1])
            self.bev.trajectory = [tuple(int(value) for value in row) for row in trajectory.tolist()]
        else:
            self.bev.trajectory = []

        loaded_semantic = False
        if load_semantic_evidence and self.semantic_bev is not None and "semantic_evidence" in data.files:
            evidence = np.asarray(data["semantic_evidence"], dtype=np.float32)
            if evidence.shape == self.semantic_bev.evidence.shape:
                if load_semantic_as_prior:
                    self.semantic_bev.prior_evidence = evidence.copy()
                    self.semantic_bev.evidence = np.zeros_like(self.semantic_bev.evidence)
                else:
                    self.semantic_bev.evidence = evidence.copy()
                    self.semantic_bev.prior_evidence = np.zeros_like(self.semantic_bev.evidence)
                loaded_semantic = True
            elif align == "center" and evidence.ndim == 3 and evidence.shape[0] == self.semantic_bev.evidence.shape[0]:
                target_evidence = np.zeros_like(self.semantic_bev.evidence)
                ox, oy = paste_offset
                sx, sy = evidence.shape[1:]
                if ox >= 0 and oy >= 0 and ox + sx <= target_evidence.shape[1] and oy + sy <= target_evidence.shape[2]:
                    target_evidence[:, ox : ox + sx, oy : oy + sy] = evidence
                    if load_semantic_as_prior:
                        self.semantic_bev.prior_evidence = target_evidence
                        self.semantic_bev.evidence = np.zeros_like(self.semantic_bev.evidence)
                    else:
                        self.semantic_bev.evidence = target_evidence
                        self.semantic_bev.prior_evidence = np.zeros_like(self.semantic_bev.evidence)
                    loaded_semantic = True
        pathfinder = getattr(self.sim, "pathfinder", None)
        self.oracle_free_mask = None
        if self.enable_oracle_metrics and pathfinder is not None and getattr(pathfinder, "is_loaded", False):
            agent_state = self.sim.get_agent(0).get_state()
            self.oracle_free_mask = oracle_navmesh_mask(
                pathfinder=pathfinder,
                origin_world_xz=self.bev.origin_world_xz,
                grid_size=self.bev.config.grid_size,
                resolution=self.bev.config.resolution,
                height=float(np.asarray(agent_state.position, dtype=np.float32)[1]),
            )
        self.last_payload = None
        self.last_loaded_bev_transform = {
            "source_origin_world_xz": list(loaded_origin),
            "target_origin_world_xz": list(self.bev.origin_world_xz),
            "source_resolution": resolution,
            "target_resolution": float(self.bev.config.resolution),
            "paste_offset_cells": list(paste_offset),
            "align": align,
        }
        return {
            "path": str(path),
            "source_scene": metadata.get("scene"),
            "target_scene": str(self.scene),
            "loaded_semantic_evidence": loaded_semantic,
            "loaded_semantic_as_prior": bool(load_semantic_as_prior),
            "align": align,
            "kept_trajectory": bool(keep_trajectory),
            "source_grid_size": list(int(value) for value in metadata.get("grid_size", occupancy.shape)),
            "grid_size": list(self.bev.config.grid_size),
            "origin_world_xz": list(self.bev.origin_world_xz),
            "paste_offset_cells": list(paste_offset),
        }

    def load_object_memory(
        self,
        path: str | Path,
        source: str = "prior_A",
        align_to_loaded_bev: bool = True,
        reset_evidence: bool = False,
    ) -> dict[str, Any]:
        loaded = ObjectMemoryStore.load(path)
        transform = self.last_loaded_bev_transform if align_to_loaded_bev else None
        next_negative_id = -1
        imported = 0
        for old_id, item in loaded.items.items():
            new_id = f"{source}::{old_id}"
            while any(existing.semantic_id == next_negative_id for existing in self.memory_store.items.values()):
                next_negative_id -= 1
            item.id = new_id
            item.object_id = item.object_id or old_id
            item.semantic_id = next_negative_id
            next_negative_id -= 1
            item.source = source
            item.centroid_xz = _transform_xz_from_loaded_bev(item.centroid_xz, transform)
            if reset_evidence:
                item.missed_observation_count = 0
                item.negative_evidence_count = 0
                item.not_observable_count = 0
                item.missed_observation_weight = 0.0
                item.negative_evidence_weight = 0.0
                item.not_observable_weight = 0.0
            item.status = "active"
            self.memory_store.items[new_id] = item
            imported += 1
        self.last_payload = None
        return {
            "path": str(path),
            "source": source,
            "source_scene": loaded.scene_id,
            "target_scene": str(self.scene),
            "imported_items": imported,
            "align_to_loaded_bev": bool(align_to_loaded_bev),
            "reset_evidence": bool(reset_evidence),
        }

    def _prior_expected_visible_item_ids(
        self,
        depth: np.ndarray,
        sensor_position_xyz: np.ndarray,
        sensor_rotation,
        floor_y: float,
        hfov_deg: float,
        occlusion_margin_m: float = 0.25,
        min_projected_points: int = 2,
        min_unoccluded_fraction: float = 0.55,
        patch_radius_px: int = 2,
        min_patch_valid_fraction: float = 0.5,
    ) -> set[str]:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            return set()
        height, width = depth.shape
        fx = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
        fy = fx
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        sensor_xyz = np.asarray(sensor_position_xyz, dtype=np.float32).reshape(3)
        axes = _camera_axes(sensor_rotation)
        expected: set[str] = set()
        for item in self.memory_store.items.values():
            if not str(item.source).startswith("prior_"):
                continue
            center_xyz, height_range_y, size_xyz = _prior_visibility_shape(item.category, item.centroid_xz, floor_y)
            projected = [
                _project_world_point(point, sensor_xyz, axes, fx, fy, cx, cy, width, height)
                for point in _object_visibility_points(center_xyz, height_range_y, size_xyz)
            ]
            projected = [value for value in projected if value is not None]
            if len(projected) < max(1, int(min_projected_points)):
                continue
            unoccluded = 0
            for row, col, distance in projected:
                observed_depth = _patch_depth(depth, row, col, patch_radius_px, min_patch_valid_fraction)
                if observed_depth is not None and observed_depth + occlusion_margin_m >= distance:
                    unoccluded += 1
            if unoccluded >= max(2, int(np.ceil(len(projected) * float(min_unoccluded_fraction)))):
                expected.add(item.id)
        return expected

    def load_memory(self) -> dict[str, Any]:
        with self.lock:
            if self.memory_path is None:
                raise RuntimeError("No --memory-path was provided")
            if not self.memory_path.exists():
                raise FileNotFoundError(self.memory_path)
            self.memory_store = ObjectMemoryStore.load(self.memory_path)
            self.last_payload = self._state_payload()
            return self.last_payload

    def _geometry_oracle_payload(self) -> dict[str, Any]:
        if self.oracle_free_mask is None:
            return {"enabled": False}
        metrics = mapping_metrics(
            pred_free=self.bev.free_mask(),
            pred_occupied=self.bev.occupied_mask(),
            explored=self.bev.explored,
            oracle_free=self.oracle_free_mask,
            resolution=self.bev.config.resolution,
        )
        metrics["enabled"] = True
        return metrics

    def _pose_from_state(self, state) -> dict[str, float]:
        position = np.asarray(getattr(state, "position", [0.0, 0.0, 0.0]), dtype=np.float32)
        return {
            "x": float(position[0]),
            "y": float(position[2] if position.size > 2 else position[-1]),
            "heading_deg": _heading_deg_from_rotation(getattr(state, "rotation", None)),
        }

    def _evidence_update_from_pose(self, pose: dict[str, float]) -> dict[str, float | str]:
        if self.last_evidence_pose is None:
            return {
                "weight": 1.0,
                "negative_weight": 0.0,
                "translation_m": 0.0,
                "rotation_deg": 0.0,
                "novelty": 1.0,
                "translation_gate": 0.0,
                "reason": "initial",
            }
        dx = float(pose["x"]) - float(self.last_evidence_pose["x"])
        dz = float(pose["y"]) - float(self.last_evidence_pose["y"])
        translation = float(np.sqrt(dx * dx + dz * dz))
        rotation = abs(_angle_delta_deg(float(pose["heading_deg"]), float(self.last_evidence_pose["heading_deg"])))
        novelty = float(np.sqrt((translation / 0.35) ** 2 + (rotation / 30.0) ** 2))
        rise = 1.0 - float(np.exp(-novelty))
        fast_decay = float(np.exp(-(max(0.0, novelty - 2.0) ** 2) / 0.6))
        weight = float(np.clip(0.05 + 0.95 * rise * fast_decay, 0.05, 1.0))
        translation_gate = float(np.clip((translation - 0.12) / 0.38, 0.0, 1.0))
        negative_weight = float(np.clip(weight * translation_gate * self.negative_evidence_scale, 0.0, 1.0))
        return {
            "weight": weight,
            "negative_weight": negative_weight,
            "translation_m": translation,
            "rotation_deg": rotation,
            "novelty": novelty,
            "translation_gate": translation_gate,
            "reason": "motion_weighted",
        }


def _observability_label(semantic_id: int, seen_ids: set[int], expected_visible_ids: set[int]) -> str:
    if semantic_id in seen_ids:
        return "positive"
    if semantic_id in expected_visible_ids:
        return "expected_visible_miss"
    return "not_observable"


def _transform_xz_from_loaded_bev(
    centroid_xz: tuple[float, float],
    transform: dict[str, Any] | None,
) -> tuple[float, float]:
    if not transform:
        return (float(centroid_xz[0]), float(centroid_xz[1]))
    source_origin = transform.get("source_origin_world_xz") or [0.0, 0.0]
    target_origin = transform.get("target_origin_world_xz") or source_origin
    source_resolution = float(transform.get("source_resolution", 0.05))
    target_resolution = float(transform.get("target_resolution", source_resolution))
    paste_offset = transform.get("paste_offset_cells") or [0, 0]
    cell_x = (float(centroid_xz[0]) - float(source_origin[0])) / source_resolution + float(paste_offset[0])
    cell_y = (float(centroid_xz[1]) - float(source_origin[1])) / source_resolution + float(paste_offset[1])
    return (
        float(target_origin[0]) + cell_x * target_resolution,
        float(target_origin[1]) + cell_y * target_resolution,
    )


def _prior_visibility_shape(
    category: str,
    centroid_xz: tuple[float, float],
    floor_y: float,
) -> tuple[list[float], list[float], list[float]]:
    category = (category or "").lower()
    profiles = {
        "wall": (0.15, 2.25, [0.8, 2.1, 0.25]),
        "door": (0.10, 2.10, [0.9, 2.0, 0.18]),
        "bed": (0.20, 0.90, [1.8, 0.7, 2.0]),
        "chair": (0.20, 1.25, [0.7, 1.0, 0.7]),
        "table": (0.45, 1.15, [1.2, 0.7, 1.2]),
    }
    low, high, size = profiles.get(category, (0.25, 1.40, [0.8, 1.0, 0.8]))
    low_y = float(floor_y) + low
    high_y = float(floor_y) + high
    center_y = 0.5 * (low_y + high_y)
    center_xyz = [float(centroid_xz[0]), center_y, float(centroid_xz[1])]
    return center_xyz, [low_y, high_y], size


def _sample_navigable_path(sim, min_distance_m: float, max_samples: int, attempts: int = 80) -> list[np.ndarray]:
    pathfinder = getattr(sim, "pathfinder", None)
    if pathfinder is None or not getattr(pathfinder, "is_loaded", False):
        return []

    import habitat_sim

    best_points = []
    best_distance = -1.0
    for _ in range(attempts):
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        path.requested_end = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if not pathfinder.find_path(path):
            continue
        distance = float(path.geodesic_distance)
        points = [np.asarray(point, dtype=np.float32) for point in path.points]
        if distance > best_distance and len(points) >= 2:
            best_distance = distance
            best_points = points
        if distance >= min_distance_m and len(points) >= 2:
            return _resample_polyline(points, max_samples)
    return _resample_polyline(best_points, max_samples) if best_points else []


def _resample_polyline(points: list[np.ndarray], max_samples: int) -> list[np.ndarray]:
    if not points:
        return []
    if len(points) == 1 or max_samples <= 1:
        return points[:1]

    cumulative = [0.0]
    for prev, cur in zip(points[:-1], points[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(cur - prev)))
    total = cumulative[-1]
    if total <= 0.0:
        return points[:1]

    targets = np.linspace(0.0, total, max_samples)
    out = []
    segment = 0
    for target in targets:
        while segment + 1 < len(cumulative) and cumulative[segment + 1] < target:
            segment += 1
        if segment + 1 >= len(points):
            out.append(points[-1].copy())
            continue
        span = cumulative[segment + 1] - cumulative[segment]
        alpha = 0.0 if span <= 0.0 else (target - cumulative[segment]) / span
        out.append((1.0 - alpha) * points[segment] + alpha * points[segment + 1])
    return [np.asarray(point, dtype=np.float32) for point in out]


def _next_point(points: list[np.ndarray], idx: int):
    if not points:
        return None
    if idx + 1 < len(points):
        return points[idx + 1]
    if idx > 0:
        return points[idx - 1]
    return None


def _rotation_toward(position: np.ndarray, look_at):
    if look_at is None:
        return None
    direction = np.asarray(look_at - position, dtype=np.float32)
    norm = float(np.linalg.norm(direction[[0, 2]]))
    if norm <= 1e-6:
        return None
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    yaw = float(np.arctan2(-direction[0], -direction[2]))
    import quaternion

    return quaternion.from_rotation_vector([0.0, yaw, 0.0])


class Handler(BaseHTTPRequestHandler):
    session: HabitatControlSession
    outputs_root: Path = ROOT / "outputs"

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/state":
            self._send_json(self.session.state())
            return
        if self._send_output_asset():
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        try:
            if self.path == "/api/reset":
                self._send_json(self.session.reset())
                return
            if self.path == "/api/save_memory":
                self._send_json(self.session.save_memory())
                return
            if self.path == "/api/load_memory":
                self._send_json(self.session.load_memory())
                return
            if self.path == "/api/action":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b"{}"
                payload = json.loads(body.decode("utf-8"))
                self._send_json(self.session.action(str(payload.get("action", ""))))
                return
            self._send(404, b"not found", "text/plain")
        except Exception as exc:
            self._send(500, str(exc).encode("utf-8"), "text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._send(200, json.dumps(payload).encode("utf-8"), "application/json")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_output_asset(self) -> bool:
        parsed = urlparse(self.path)
        rel = unquote(parsed.path).lstrip("/")
        if not rel or rel.startswith("api/") or rel == "favicon.ico":
            return False
        candidate = (self.outputs_root / rel).resolve()
        try:
            candidate.relative_to(self.outputs_root.resolve())
        except ValueError:
            return False
        if not candidate.is_file():
            return False
        content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        self._send(200, candidate.read_bytes(), content_type)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Habitat-Sim remote-control web UI.")
    parser.add_argument("--scene", required=True, help="Path to a Habitat-Sim loadable .glb scene.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=43901)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--move-amount", type=float, default=0.25)
    parser.add_argument("--turn-amount", type=float, default=15.0)
    parser.add_argument("--scene-dataset-config")
    parser.add_argument("--bev-resolution", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--obstacle-dilation-cells", type=int, default=2)
    parser.add_argument("--semantic-categories", default="wall,door,table,chair")
    parser.add_argument("--semantic-confidence-saturation", type=float, default=80.0)
    parser.add_argument("--freshness-tau-steps", type=float, default=20.0)
    parser.add_argument("--memory-path")
    parser.add_argument("--start-path-min-distance", type=float, default=3.0)
    parser.add_argument("--start-path-samples", type=int, default=48)
    parser.add_argument("--disable-oracle-metrics", action="store_true")
    args = parser.parse_args()

    ensure_conda_nvidia_egl_vendor()
    scene = Path(args.scene).expanduser().resolve()
    if not scene.exists():
        raise FileNotFoundError(scene)
    scene_dataset_config = Path(args.scene_dataset_config).expanduser().resolve() if args.scene_dataset_config else None
    if scene_dataset_config is not None and not scene_dataset_config.exists():
        raise FileNotFoundError(scene_dataset_config)
    semantic_categories = [item.strip().lower() for item in args.semantic_categories.split(",") if item.strip()]
    memory_path = Path(args.memory_path).expanduser().resolve() if args.memory_path else None

    session = HabitatControlSession(
        scene=scene,
        resolution=args.resolution,
        move_amount=args.move_amount,
        turn_amount=args.turn_amount,
        scene_dataset_config=scene_dataset_config,
        bev_resolution=args.bev_resolution,
        grid_size=args.grid_size,
        sample_stride=args.sample_stride,
        obstacle_dilation_cells=args.obstacle_dilation_cells,
        semantic_categories=semantic_categories,
        semantic_confidence_saturation=args.semantic_confidence_saturation,
        freshness_tau_steps=args.freshness_tau_steps,
        memory_path=memory_path,
        start_path_min_distance=args.start_path_min_distance,
        start_path_samples=args.start_path_samples,
        enable_oracle_metrics=not args.disable_oracle_metrics,
    )
    Handler.session = session
    server = HTTPServer((args.host, args.port), Handler)
    print(f"RSC-Nav Habitat control UI: http://{args.host}:{args.port}/")
    print(f"Scene: {scene}")
    try:
        server.serve_forever()
    finally:
        session.close()


def ensure_conda_nvidia_egl_vendor() -> None:
    prefix = Path(sys.prefix)
    vendor_dir = prefix / "etc" / "glvnd" / "egl_vendor.d"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    vendor_file = vendor_dir / "10_nvidia.json"
    if vendor_file.exists():
        return
    vendor_file.write_text(
        '{\n'
        '    "file_format_version" : "1.0.0",\n'
        '    "ICD" : {\n'
        '        "library_path" : "libEGL_nvidia.so.0"\n'
        "    }\n"
        "}\n",
        encoding="utf-8",
    )


def _rgb_array(raw_rgb) -> np.ndarray:
    if raw_rgb is None:
        raise RuntimeError("Habitat-Sim did not return an RGB observation")
    rgb = np.asarray(raw_rgb)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    return rgb.astype(np.uint8)


def _valid_depth(raw_depth) -> np.ndarray:
    if raw_depth is None:
        raise RuntimeError("Habitat-Sim did not return a depth observation")
    depth = np.asarray(raw_depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if not (np.isfinite(depth) & (depth > 0)).any():
        raise RuntimeError("Habitat-Sim returned no positive finite depth values")
    return depth


def _rgb_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb)


def _depth_image(depth: np.ndarray) -> Image.Image:
    valid = np.isfinite(depth)
    norm = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        low = float(np.nanmin(depth[valid]))
        high = float(np.nanmax(depth[valid]))
        if high > low:
            norm = (depth - low) / (high - low)
    return Image.fromarray(np.uint8(np.clip(norm, 0, 1) * 255))


def _bev_image(bev: DenseBEVMapper) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 6))
    state = bev.occupancy_state().T
    cmap = mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#333333"])
    ax.imshow(state, origin="lower", cmap=cmap, vmin=0, vmax=2, alpha=0.86)
    ax.set_xlim(-0.5, bev.config.grid_size[0] - 0.5)
    ax.set_ylim(-0.5, bev.config.grid_size[1] - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if bev.trajectory:
        xs, ys = zip(*bev.trajectory)
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.8)
        ax.plot(xs[-1], ys[-1], color="#1f77b4", marker="*", markersize=12)
    fig.tight_layout(pad=0)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).copy()


def _semantic_bev_image(semantic_bev: SemanticBEVAccumulator, trajectory) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 6))
    state = semantic_bev.semantic_state()
    colors = ["#d9d9d9"] + [
        SEMANTIC_COLORS.get(category, "#9467bd")
        for category in semantic_bev.categories
    ]
    ax.imshow((state + 1).T, origin="lower", cmap=mcolors.ListedColormap(colors), vmin=0, vmax=len(colors) - 1)
    ax.set_xticks([])
    ax.set_yticks([])
    if trajectory:
        xs, ys = zip(*trajectory)
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.8)
        ax.plot(xs[-1], ys[-1], color="#1f77b4", marker="*", markersize=12)
    fig.tight_layout(pad=0)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).copy()


def _oracle_image(oracle_free: np.ndarray, trajectory) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 6))
    state = np.where(oracle_free.T, 1, 2)
    cmap = mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#333333"])
    ax.imshow(state, origin="lower", cmap=cmap, vmin=0, vmax=2, alpha=0.86)
    ax.set_xticks([])
    ax.set_yticks([])
    if trajectory:
        xs, ys = zip(*trajectory)
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.8)
        ax.plot(xs[-1], ys[-1], color="#1f77b4", marker="*", markersize=12)
    fig.tight_layout(pad=0)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).copy()


def _oracle_diff_image(bev: DenseBEVMapper, oracle_free: np.ndarray) -> Image.Image:
    diff = np.zeros(bev.config.grid_size, dtype=np.int8)
    diff[np.logical_and(bev.free_mask(), oracle_free)] = 1
    diff[np.logical_and(bev.free_mask(), ~oracle_free)] = 2
    diff[np.logical_and(bev.occupied_mask(), oracle_free)] = 3
    diff[np.logical_and(bev.occupied_mask(), ~oracle_free)] = 4
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#f4a261", "#4ea8de", "#333333"])
    ax.imshow(diff.T, origin="lower", cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks([])
    ax.set_yticks([])
    if bev.trajectory:
        xs, ys = zip(*bev.trajectory)
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.8)
        ax.plot(xs[-1], ys[-1], color="#1f77b4", marker="*", markersize=12)
    fig.tight_layout(pad=0)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).copy()


def _grounding_boxes_from_tracks(
    tracks: list[dict],
    memory_items: list[dict],
    depth: np.ndarray,
    sensor_position_xyz: np.ndarray,
    sensor_rotation,
    hfov_deg: float,
    occlusion_margin_m: float = 0.35,
    min_unoccluded_fraction: float = 0.35,
) -> list[dict]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        return []
    height, width = depth.shape
    if height <= 0 or width <= 0:
        return []

    memory_by_semantic_id = {
        int(item.get("semantic_id")): item
        for item in memory_items
        if item.get("semantic_id") is not None
    }
    fx = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    sensor_xyz = np.asarray(sensor_position_xyz, dtype=np.float32).reshape(3)
    axes = _camera_axes(sensor_rotation)
    boxes = []

    for track in tracks:
        center_xyz = track.get("gt_center_xyz")
        height_range = track.get("height_range_y")
        if center_xyz is None or height_range is None:
            centroid_xz = track.get("centroid_xz")
            if not centroid_xz:
                continue
            center_xyz, height_range, size_xyz = _prior_visibility_shape(
                str(track.get("category", "")),
                (float(centroid_xz[0]), float(centroid_xz[1])),
                float(sensor_xyz[1] - 1.5),
            )
        else:
            size_xyz = track.get("sizes_xyz")

        projected = [
            _project_world_point(point, sensor_xyz, axes, fx, fy, cx, cy, width, height)
            for point in _object_visibility_points(center_xyz, height_range, size_xyz)
        ]
        projected = [item for item in projected if item is not None]
        if len(projected) < 2:
            continue

        visible = []
        for row, col, distance in projected:
            observed_depth = _patch_depth(depth, row, col, radius=2, min_valid_fraction=0.4)
            if observed_depth is None or observed_depth + occlusion_margin_m >= distance:
                visible.append((row, col, distance))
        if len(visible) < max(1, int(np.ceil(len(projected) * float(min_unoccluded_fraction)))):
            continue

        rows = [row for row, _, _ in projected]
        cols = [col for _, col, _ in projected]
        pad = max(3, int(round(min(width, height) * 0.02)))
        x0 = int(np.clip(min(cols) - pad, 0, width - 1))
        y0 = int(np.clip(min(rows) - pad, 0, height - 1))
        x1 = int(np.clip(max(cols) + pad, 0, width - 1))
        y1 = int(np.clip(max(rows) + pad, 0, height - 1))
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            continue

        semantic_id = int(track.get("semantic_id", -1))
        memory_item = memory_by_semantic_id.get(semantic_id, {})
        confidence = float(memory_item.get("confidence", track.get("confidence", 0.0)))
        freshness = float(memory_item.get("freshness", track.get("freshness", 1.0)))
        score = float(np.clip(0.7 * confidence + 0.3 * freshness, 0.0, 1.0))
        category = str(track.get("category", "object"))
        boxes.append(
            {
                "semantic_id": semantic_id,
                "object_id": track.get("object_id"),
                "category": category,
                "confidence": round(confidence, 4),
                "freshness": round(freshness, 4),
                "score": round(score, 4),
                "status": memory_item.get("status", "active"),
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
                "color": SEMANTIC_COLORS.get(category, "#61c6a7"),
            }
        )

    boxes.sort(key=lambda item: item["score"], reverse=True)
    return boxes[:12]


def _semantic_payload(report: dict | None) -> dict:
    if report is None:
        return {
            "enabled": False,
            "observed_target_instances": 0,
            "semantic_cells": 0,
            "mean_freshness": 0.0,
            "per_class_cells": {},
        }
    return {
        key: value
        for key, value in report.items()
        if key != "tracks"
    }


def _sample_count(depth: np.ndarray, stride: int) -> int:
    stride = max(1, int(stride))
    sampled = depth[::stride, ::stride]
    return int((np.isfinite(sampled) & (sampled > 0)).sum())


def _image_to_base64(image: Image.Image, fmt: str, **save_kwargs) -> str:
    buffer = BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _heading_deg_from_rotation(rotation) -> float:
    if rotation is None:
        return 0.0
    try:
        vector = np.asarray(rotation.transform_vector([0.0, 0.0, -1.0]), dtype=np.float32)
    except Exception:
        return 0.0
    if vector.size < 3 or not np.isfinite(vector).all():
        return 0.0
    return float(np.degrees(np.arctan2(vector[0], -vector[2])))


def _pitch_deg_from_rotation(rotation) -> float:
    if rotation is None:
        return 0.0
    try:
        vector = np.asarray(rotation.transform_vector([0.0, 0.0, -1.0]), dtype=np.float32)
    except Exception:
        return 0.0
    if vector.size < 3 or not np.isfinite(vector).all():
        return 0.0
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return 0.0
    vector = vector / norm
    return float(np.degrees(np.arcsin(float(np.clip(vector[1], -1.0, 1.0)))))


def _angle_delta_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    main()
