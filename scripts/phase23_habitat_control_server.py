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

from bev_memory import BEVMemory
from observation_adapter import HabitatObservationAdapter


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
      <section>
        <h2>State</h2>
        <div class="metrics">
          <div class="metric"><span>Pose</span><strong id="pose">-</strong></div>
          <div class="metric"><span>Heading</span><strong id="heading">-</strong></div>
          <div class="metric"><span>Rays</span><strong id="rays">-</strong></div>
          <div class="metric"><span>Explored</span><strong id="explored">-</strong></div>
          <div class="metric"><span>Occupied</span><strong id="occupied">-</strong></div>
          <div class="metric"><span>Step</span><strong id="step">-</strong></div>
        </div>
        <div class="controls">
          <button onclick="sendAction('turn_left')">A</button>
          <button class="primary" onclick="sendAction('move_forward')">W</button>
          <button onclick="sendAction('turn_right')">D</button>
          <button onclick="sendAction('look_up')">Look Up</button>
          <button onclick="sendAction('move_back')">S</button>
          <button onclick="sendAction('look_down')">Look Down</button>
          <button class="wide" onclick="resetSim()">Reset</button>
        </div>
        <div class="hint">Keyboard: W forward, A/D turn, S back, Q/E look, R reset.</div>
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
      document.getElementById("pose").textContent = `${data.pose.x.toFixed(2)}, ${data.pose.y.toFixed(2)}`;
      document.getElementById("heading").textContent = `${data.pose.heading_deg.toFixed(1)} deg`;
      document.getElementById("rays").textContent = data.ray_count;
      document.getElementById("explored").textContent = data.bev.num_explored_cells;
      document.getElementById("occupied").textContent = data.bev.num_occupied_cells;
      document.getElementById("step").textContent = data.step;
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
    def __init__(self, scene: Path, resolution: int, move_amount: float, turn_amount: float) -> None:
        self.scene = scene
        self.resolution = resolution
        self.move_amount = move_amount
        self.turn_amount = turn_amount
        self.lock = threading.Lock()
        self.step_count = 0

        self._setup_sim()
        self._reset_memory()
        self._reset_agent()

    def close(self) -> None:
        self.sim.close()

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.step_count = 0
            self._reset_memory()
            self._reset_agent()
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
        self.adapter = HabitatObservationAdapter(
            scene_id=str(self.scene),
            episode_id="phase23_remote_control",
            hfov_deg=90.0,
            max_rays=17,
            max_depth=10.0,
        )

    def _camera_spec(self, habitat_sim, SensorSubType, sensor_type, uuid: str):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = SensorSubType.PINHOLE
        spec.resolution = [self.resolution, self.resolution]
        spec.position = [0.0, 1.5, 0.0]
        return spec

    def _reset_memory(self) -> None:
        self.bev_grid_size = (160, 160)
        self.bev_resolution = 0.1
        self.bev_origin = (0.0, 0.0)
        self.bev = BEVMemory(
            grid_size=self.bev_grid_size,
            resolution=self.bev_resolution,
            origin_world_xy=self.bev_origin,
        )

    def _reset_agent(self) -> None:
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        pathfinder = getattr(self.sim, "pathfinder", None)
        if pathfinder is not None and getattr(pathfinder, "is_loaded", False):
            point = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
            if point.shape == (3,) and np.isfinite(point).all():
                state.position = point
        agent.set_state(state)
        pose = self._pose_from_state(agent.get_state())
        self.bev_origin = (
            pose["x"] - (self.bev_grid_size[0] // 2) * self.bev_resolution,
            pose["y"] - (self.bev_grid_size[1] // 2) * self.bev_resolution,
        )
        self.bev = BEVMemory(
            grid_size=self.bev_grid_size,
            resolution=self.bev_resolution,
            origin_world_xy=self.bev_origin,
        )

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
        pose = self._pose_from_state(agent_state)

        raw = {
            "frame_id": f"phase23_remote_control_{self.step_count:05d}",
            "time": self.step_count,
            "scene_id": str(self.scene),
            "episode_id": "phase23_remote_control",
            "pose": pose,
            "rgb": rgb,
            "depth": depth,
            "semantic": semantic,
        }
        frame = self.adapter.to_frame(raw)
        self.bev.update_from_frame(frame)

        return {
            "step": self.step_count,
            "scene": str(self.scene),
            "scene_name": self.scene.name,
            "pose": pose,
            "ray_count": len(frame.rays),
            "bev": self.bev.snapshot(),
            "rgb_jpeg": _image_to_base64(_rgb_image(rgb), "JPEG", quality=86),
            "depth_png": _image_to_base64(_depth_image(depth), "PNG"),
            "bev_png": _image_to_base64(_bev_image(self.bev), "PNG"),
        }

    def _pose_from_state(self, state) -> dict[str, float]:
        position = np.asarray(getattr(state, "position", [0.0, 0.0, 0.0]), dtype=np.float32)
        return {
            "x": float(position[0]),
            "y": float(position[2] if position.size > 2 else position[-1]),
            "heading_deg": _heading_deg_from_rotation(getattr(state, "rotation", None)),
        }


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
    args = parser.parse_args()

    ensure_conda_nvidia_egl_vendor()
    scene = Path(args.scene).expanduser().resolve()
    if not scene.exists():
        raise FileNotFoundError(scene)

    session = HabitatControlSession(
        scene=scene,
        resolution=args.resolution,
        move_amount=args.move_amount,
        turn_amount=args.turn_amount,
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


def _bev_image(bev: BEVMemory) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 6))
    state = bev.occupancy_state().T
    cmap = mcolors.ListedColormap(["#d9d9d9", "#ffffff", "#333333"])
    ax.imshow(state, origin="lower", cmap=cmap, vmin=0, vmax=2, alpha=0.86)
    ax.set_xlim(-0.5, bev.grid_size[0] - 0.5)
    ax.set_ylim(-0.5, bev.grid_size[1] - 0.5)
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
