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

If a Habitat scene is already available, prefer:

```bash
RSCNAV_HABITAT_SCENE=/abs/path/to/scene.glb bash scripts/phase22_remote_linux_run.sh
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
- `logs/scene_live_smoke.exit` is `0` if a scene is available

If `none_live_smoke` fails at EGL, report it as an environment blocker, not a
project-code blocker.

