# Phase 2.2 Remote Linux Runner

This is the low-transfer workflow for Phase 2C live Habitat-Sim validation on a
remote Linux development machine.

## One Command

After fetching the branch on the Linux machine, run from the repository root:

```bash
bash scripts/phase22_remote_linux_run.sh
```

The script writes all diagnostics and results under:

```text
outputs/phase22_remote/
```

Bring back only the generated tarball:

```text
outputs/phase22_remote/<host>_<timestamp>.tar.gz
```

If Codex is available on the target Linux machine, give it
`docs/phase22_target_codex_handoff.md` and ask it to run the handoff.

## Optional Inputs

If the Linux machine already has a Habitat scene, provide it:

```bash
RSCNAV_HABITAT_SCENE=/abs/path/to/scene.glb bash scripts/phase22_remote_linux_run.sh
```

If no scene is provided, the runner tries to download Habitat test scenes and
uses the first `.glb` it finds. Test scenes are enough for live RGB-D/EGL smoke;
they may not provide semantic labels.

Useful knobs:

```bash
RSCNAV_ENV_NAME=rscnav-habitat22
RSCNAV_CONDA_PREFIX=$HOME/.rscnav/miniforge3
RSCNAV_HABITAT_LAB_DIR=$HOME/.rscnav/habitat-lab
RSCNAV_HABITAT_DATA=$HOME/.rscnav/habitat_data
RSCNAV_DOWNLOAD_TEST_SCENES=0
RSCNAV_SKIP_SETUP=1
```

## What It Does

- Collects OS, GPU, EGL/GLVND, git, and environment diagnostics.
- Installs Miniforge under `$HOME/.rscnav/miniforge3` if no conda/mamba is found.
- Creates or reuses `rscnav-habitat22`.
- Installs `habitat-sim headless`, pins `numpy=1.26.4` and `pillow=10.4.0`, and installs `habitat-lab` stable.
- Runs import checks, `pip check`, Phase 2.2 adapter contract smoke, no-scene live smoke, and optional scene live smoke.
- Bundles logs and selected output artifacts into one small `.tar.gz`.

## Pass Criteria

- `logs/import_check.exit` is `0`.
- `logs/pip_check.exit` is `0`.
- `logs/contract_smoke.exit` is `0`.
- `logs/none_live_smoke.exit` is `0` for EGL/live rendering readiness.
- `logs/scene_live_smoke.exit` is `0` when a scene path or downloaded test scene is available.
