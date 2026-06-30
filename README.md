# RSC-Nav

RSC-Nav studies long-term semantic-spatial memory augmentation for embodied navigation.

Current repository stage:

- Phase 0: protocol and data contracts.
- Phase 1: minimal memory core smoke test.
- Phase 2: synthetic BEV memory projection smoke test.
- Phase 2.1: unified ObservationFrame adapter smoke test.
- Phase 2.2: Habitat RGB-D / semantic adapter contract smoke test.
- Phase 2 close-out: live Habitat BEV / semantic BEV / object memory update evidence.

## Phase 2 Evidence Page

The main Phase 2 visual summary is hosted from the development machine:

- [Phase 2 curated evidence](http://39.101.65.229:43901/negfix_ab_index.html)

Development-machine file:

```text
/workspace/yujiexiao/RSC_Nav/outputs/phase213_episode_runs/negfix_ab_index.html
```

This page links the curated A / B / A->B runs, GIFs, `summary.html` reports,
prior/live semantic-evidence curves, and Phase 2 documentation. Large `outputs/`
artifacts are intentionally kept outside Git.

## Phase Documents

Stage-specific execution notes are collected here:

- [Phase document index](docs/phase_docs/README.md)
- [Phase 2 execution document](RSC-Nav_Phase2_阶段性执行文档.md)
- [Phase 2 BEV and semantic memory notes](docs/phase_docs/phase2_bev_semantic_memory.md)

## PowerShell UTF-8 Setup

Windows PowerShell 5.1 may decode UTF-8 Markdown without BOM as GBK/ANSI,
which makes Chinese text look noisy in `Get-Content` output. Before running
repo commands in PowerShell, load the project UTF-8 defaults for the current
process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\scripts\powershell_utf8.ps1
```

The script sets console input/output, `$OutputEncoding`, Python UTF-8
environment variables, and common PowerShell `-Encoding` defaults to UTF-8.

## Phase 0+1 Smoke Test

Run:

```powershell
python scripts\phase01_smoke_test.py
```

Outputs:

- `outputs/phase01/phase01_smoke_log.json`
- `outputs/phase01/phase01_smoke_visualization.png`

The smoke test demonstrates:

```text
write -> retrieve -> perturb -> stale retrieval -> reconfigured retrieval
```

It is synthetic by design and does not require Habitat. Habitat integration starts after the memory core and schema are stable.

## Phase 2 BEV Smoke Test

Run:

```powershell
python scripts\phase02_bev_smoke_test.py
```

Outputs:

- `outputs/phase02/phase02_log.json`
- `outputs/phase02/bev_occupancy.png`
- `outputs/phase02/bev_semantic.png`
- `outputs/phase02/bev_memory_overlay.png`
- `outputs/phase02/bev_update_sequence.png`

This smoke test demonstrates:

```text
synthetic observation + pose
-> BEV projection
-> occupancy / explored map
-> semantic evidence map
-> long-term memory retrieval
```

It is still synthetic. The purpose is to stabilize the BEV memory interface before replacing synthetic observations with Habitat RGB-D, pose, and semantic detector outputs.

## Phase 2.1 Observation Adapter Smoke Test

Run:

```powershell
python scripts\phase21_observation_adapter_test.py
```

Outputs:

- `outputs/phase21/phase21_log.json`
- `outputs/phase21/observation_frame_debug.png`
- `outputs/phase21/adapter_bev_overlay.png`

This smoke test demonstrates:

```text
synthetic / mock Habitat-like observation
-> ObservationFrame
-> BEVMemory.update_from_frame(...)
-> long-term memory retrieval
```

## Phase 2.2 Habitat Adapter Contract Smoke Test

Run:

```powershell
python scripts\phase22_habitat_adapter_contract_test.py
```

Outputs:

- `outputs/phase22/phase22_log.json`
- `outputs/phase22/observation_frame.json`
- `outputs/phase22/habitat_like_inputs.png`
- `outputs/phase22/habitat_adapter_bev_overlay.png`

This test uses Habitat-style `rgb`, `depth`, `semantic`, and `pose` arrays,
but does not require Habitat to be installed. It verifies the project-side
contract that real Habitat frames must satisfy before entering BEV memory.

When `habitat-sim` is installed, run the live no-scene simulator smoke:

```powershell
wsl -d RSCNav-Ubuntu-22.04 -u root -- bash -lc "cd /mnt/e/WangLab/RSC_VLN && /opt/conda/bin/mamba run -n rscnav-habitat22 python scripts/phase22_habitat_sim_none_smoke.py"
```

This script verifies whether the current machine can create a headless
Habitat-Sim RGB-D context. On the current WSL2 setup, Habitat imports work but
live rendering is blocked by an EGL/CUDA device mismatch; see
`docs/phase22_habitat_wsl2_setup.md`.

For WSL2 + conda + Habitat setup notes, see:

- `docs/phase22_habitat_wsl2_setup.md`
- `envs/rscnav-habitat22.yml`
