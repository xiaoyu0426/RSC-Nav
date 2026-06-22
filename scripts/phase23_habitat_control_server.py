from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dense_bev_mapper import DenseBEVConfig, DenseBEVMapper
from object_memory_store import ObjectMemoryStore
from semantic_bev_memory import SEMANTIC_COLORS, SemanticBEVAccumulator, semantic_array


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
      <h2>RGB</h2>
      <img id="rgb" class="media" alt="RGB view" />
    </section>
    <section>
      <h2>BEV Memory</h2>
      <img id="bev" class="media" alt="BEV map" />
    </section>
    <aside>
      <section>
        <h2>Depth</h2>
        <img id="depth" class="media" alt="Depth view" />
      </section>
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
          <button onclick="saveMemory()">Save</button>
          <button onclick="loadMemory()">Load</button>
          <button onclick="getState()">Refresh</button>
        </div>
        <div class="hint" id="memory">Keyboard: W forward, A/D turn, S back, Q/E look, R reset.</div>
      </section>
    </aside>
  </main>
  <script>
    let busy = false;
    const statusEl = document.getElementById("status");

    function setStatus(text, bad=false) {
      statusEl.textContent = text;
      statusEl.className = bad ? "error" : "";
    }

    function applyState(data) {
      document.getElementById("rgb").src = "data:image/jpeg;base64," + data.rgb_jpeg;
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
      document.getElementById("memory").textContent = `Memory: ${classes}. Keyboard: W forward, A/D turn, S back, Q/E look, R reset.`;
      setStatus(`ready: ${data.scene_name}`);
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
    });

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
        memory_path: Path | None = None,
        start_path_min_distance: float = 3.0,
        start_path_samples: int = 12,
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
        self.memory_path = memory_path
        self.start_path_min_distance = start_path_min_distance
        self.start_path_samples = start_path_samples
        self.lock = threading.Lock()
        self.step_count = 0

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
            return self._state_payload()

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._state_payload()

    def action(self, action: str) -> dict[str, Any]:
        with self.lock:
            if action in {"move_forward", "turn_left", "turn_right", "look_up", "look_down"}:
                self.sim.step(action)
            elif action == "move_back":
                self._move_back()
            else:
                raise ValueError(f"Unknown action: {action}")
            self.step_count += 1
            return self._state_payload()

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
        origin = (
            pose["x"] - (config.grid_size[0] // 2) * config.resolution,
            pose["y"] - (config.grid_size[1] // 2) * config.resolution,
        )
        self.bev = DenseBEVMapper(origin_world_xz=origin, config=config)
        self.semantic_bev = None
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
        )

    def _reset_agent(self) -> None:
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        path_positions = _sample_navigable_path(
            self.sim,
            min_distance_m=self.start_path_min_distance,
            max_samples=self.start_path_samples,
        )
        if path_positions:
            state.position = path_positions[0]
            rotation = _rotation_toward(path_positions[0], _next_point(path_positions, 0))
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

        snapshot = self.bev.update_from_depth(
            depth=depth,
            agent_position_xyz=agent_state.position,
            sensor_position_xyz=sensor_state.position,
            sensor_rotation=sensor_state.rotation,
            hfov_deg=90.0,
        )
        semantic_report = None
        if self.semantic_bev is not None and semantic is not None:
            self.semantic_bev.update_from_observation(
                depth=depth,
                semantic=semantic_array(semantic),
                sensor_position_xyz=np.asarray(sensor_state.position, dtype=np.float32),
                sensor_rotation=sensor_state.rotation,
                floor_y=float(np.asarray(agent_state.position, dtype=np.float32)[1]),
                hfov_deg=90.0,
                step=self.step_count,
            )
            semantic_report = self.semantic_bev.report()
            self.memory_store.update_from_tracks(
                semantic_report.get("tracks", []),
                current_step=self.step_count,
            )
        else:
            self.memory_store.decay(current_step=self.step_count)

        return {
            "step": self.step_count,
            "scene": str(self.scene),
            "scene_name": self.scene.name,
            "pose": pose,
            "ray_count": _sample_count(depth, self.bev.config.sample_stride),
            "bev": snapshot,
            "semantic": _semantic_payload(semantic_report),
            "memory": self.memory_store.summary(),
            "rgb_jpeg": _image_to_base64(_rgb_image(rgb), "JPEG", quality=86),
            "depth_png": _image_to_base64(_depth_image(depth), "PNG"),
            "bev_png": _image_to_base64(_bev_image(self.bev), "PNG"),
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
            return self._state_payload()

    def load_memory(self) -> dict[str, Any]:
        with self.lock:
            if self.memory_path is None:
                raise RuntimeError("No --memory-path was provided")
            if not self.memory_path.exists():
                raise FileNotFoundError(self.memory_path)
            self.memory_store = ObjectMemoryStore.load(self.memory_path)
            return self._state_payload()

    def _pose_from_state(self, state) -> dict[str, float]:
        position = np.asarray(getattr(state, "position", [0.0, 0.0, 0.0]), dtype=np.float32)
        return {
            "x": float(position[0]),
            "y": float(position[2] if position.size > 2 else position[-1]),
            "heading_deg": _heading_deg_from_rotation(getattr(state, "rotation", None)),
        }


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

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/state":
            self._send_json(self.session.state())
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
    parser.add_argument("--start-path-samples", type=int, default=12)
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


if __name__ == "__main__":
    main()
