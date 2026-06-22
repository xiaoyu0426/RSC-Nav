# Phase 2.2 Target Codex Handoff

You are running on the target Linux development machine. Your goal is to run
Phase 2C live Habitat-Sim validation and return a compact result bundle. The
source machine cannot push from here; assume the human will fetch this branch on
the target machine.

## Constraints

- Do not push to any remote.
- Prefer local fixes on the target machine when they are environment-only.
- Keep all output under `outputs/phase22_remote/`, `outputs/phase22_live/`,
  `outputs/phase22_sim/`, or `outputs/phase22/`.
- Return only the final tarball path and a short summary.

## First Command

From the repository root, run:

```bash
bash scripts/phase22_remote_linux_run.sh
```

For the current goal, prefer a real scene path. If a Habitat scene is already
available, run:

```bash
RSCNAV_HABITAT_SCENE=/abs/path/to/scene.glb bash scripts/phase22_remote_linux_run.sh
```

If no local scene is available, allow the runner to download Habitat test scenes:

```bash
RSCNAV_DOWNLOAD_TEST_SCENES=1 bash scripts/phase22_remote_linux_run.sh
```

If HuggingFace is blocked, the runner falls back to a direct mirror download for
`apartment_1.glb`. Override the mirror with:

```bash
RSCNAV_TEST_SCENE_URL=https://.../apartment_1.glb bash scripts/phase22_remote_linux_run.sh
```

The runner also writes an NVIDIA EGL GLVND vendor JSON inside the active conda
environment when `libEGL_nvidia.so.0` is present. This fixes machines where the
system GLVND directory exposes only Mesa without requiring sudo.

On multi-GPU machines, the runner automatically retries the live rendering smoke
with a small matrix of `CUDA_VISIBLE_DEVICES=<gpu_index>` and Habitat-Sim
`SimulatorConfiguration.gpu_device_id`. To restrict the search, set:

```bash
RSCNAV_CUDA_DEVICE_TRIES="0 1 2 3" RSCNAV_HABITAT_SCENE=/abs/path/to/scene.glb bash scripts/phase22_remote_linux_run.sh
```

If Habitat-Sim should try a different internal device list than the visible CUDA
device list, also set:

```bash
RSCNAV_CUDA_DEVICE_TRIES="0 1 2 3" RSCNAV_HABITAT_GPU_DEVICE_TRIES="0 1 2 3" RSCNAV_HABITAT_SCENE=/abs/path/to/scene.glb bash scripts/phase22_remote_linux_run.sh
```

The script creates a tarball like:

```text
outputs/phase22_remote/<host>_<timestamp>.tar.gz
```

That tarball is the artifact to bring back.

## If The Runner Fails

Inspect these files first:

```text
outputs/phase22_remote/<host>_<timestamp>/environment_report.txt
outputs/phase22_remote/<host>_<timestamp>/logs/runner.log
outputs/phase22_remote/<host>_<timestamp>/logs/*stderr.log
outputs/phase22_remote/<host>_<timestamp>/logs/*.exit
```

Common cases:

- Conda unavailable: the runner should install Miniforge under
  `$HOME/.rscnav/miniforge3`.
- `habitat_sim` import fails: rerun the runner after checking conda channel
  access to `conda-forge` and `aihabitat`.
- EGL context fails: verify NVIDIA Linux driver, `libEGL_nvidia`, GLVND vendor
  JSON, and `nvidia-smi`. This is the key Phase 2C live-rendering gate.
- Scene smoke skipped: set `RSCNAV_HABITAT_SCENE` to an existing `.glb` scene or
  allow the runner to download Habitat test scenes.

## Success Criteria

Report success only when:

- `logs/import_check.exit` is `0`
- `logs/pip_check.exit` is `0`
- `logs/contract_smoke.exit` is `0`
- `logs/none_live_smoke.exit` is `0`
- `logs/scene_live_smoke.exit` is `0`
- `repo_outputs/phase22_live/phase22_habitat_live_scene_log.json` exists
- `repo_outputs/phase22_live/rgb.png`, `depth.png`, and `bev_overlay.png` exist
- `logs/none_live_smoke.selected_habitat_gpu_device_id` exists
- `logs/scene_live_smoke.selected_habitat_gpu_device_id` exists

If `none_live_smoke` fails at EGL, report it as an environment blocker, not a
project-code blocker.
