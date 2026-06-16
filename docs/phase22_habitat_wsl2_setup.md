# Phase 2.2 WSL2 + Habitat Setup

This document records the conservative environment path for Phase 2.2:

```text
Habitat RGB-D / semantic / pose
-> ObservationFrame
-> BEVMemory.update_from_frame(...)
-> SemanticSpatialMemory retrieval
```

Phase 2.2 should not include ObjectNav training, Habitat baselines, landmark
graph retrieval, or policy learning. Those belong to later phases.

## Current Local Check

On 2026-06-16, this Windows host had:

- Conda available: `conda 24.9.2`
- NVIDIA GPU available on Windows: RTX 3060, 12 GB VRAM
- NVIDIA driver reported by `nvidia-smi`: 591.86
- CUDA runtime reported by `nvidia-smi`: 13.1
- No WSL distro currently installed, based on `wsl -l -v`

Later on 2026-06-16, `RSCNav-Ubuntu-22.04` was created from the official
Ubuntu Jammy WSL rootfs and used for Phase 2.2 setup:

- Ubuntu 22.04.5 LTS under WSL2
- WSL kernel: `6.6.87.2-microsoft-standard-WSL2`
- WSL `nvidia-smi` works and sees RTX 3060 / CUDA 13.1
- Miniforge installed at `/opt/conda`
- Conda env created: `rscnav-habitat22`
- `habitat-sim 0.3.3` headless installed
- `habitat-lab 0.3.3` installed from the stable branch at `/opt/habitat-lab`
- Project contract smoke passed inside WSL:

```bash
cd /mnt/e/WangLab/RSC_VLN
/opt/conda/bin/mamba run -n rscnav-habitat22 python scripts/phase22_habitat_adapter_contract_test.py
```

The real-time Habitat-Sim no-scene smoke currently fails at EGL context
creation:

```text
Platform::WindowlessEglApplication::tryCreateContext():
unable to find CUDA device 0 among 1 EGL devices in total
WindowlessContext: Unable to create windowless context
```

Diagnosis:

- CUDA is visible in WSL through `/usr/lib/wsl/lib`.
- `ldconfig` sees `libcuda.so.1` and other WSL NVIDIA CUDA libraries.
- EGL vendor config currently exposes Mesa only:
  `/usr/share/glvnd/egl_vendor.d/50_mesa.json`.
- No NVIDIA EGL vendor library is exposed in the WSL instance, so
  Habitat-Sim headless cannot match EGL device 0 to CUDA device 0.

This blocks live Habitat-Sim rendering in the current WSL2 setup, but it does
not block the project-side `ObservationFrame` / BEV contract smoke.

The missing WSL distro means real Habitat-Sim in WSL2 cannot be run from this
workspace yet. The repository now includes a contract-level Phase 2.2 smoke test
that does not require Habitat installation:

```powershell
python scripts\phase22_habitat_adapter_contract_test.py
```

## Recommended WSL2 Path

1. Install Ubuntu under WSL2, preferably Ubuntu 22.04 LTS for a stable research
   environment.
2. Confirm GPU visibility inside WSL:

```bash
nvidia-smi
```

3. Install Miniforge or Miniconda inside WSL.
4. Create the project environment:

```bash
conda env create -f envs/rscnav-habitat22.yml
conda activate rscnav-habitat22
```

5. Install Habitat-Sim with the official conda channel combination. Start with
   the stable release and use the headless variant for WSL2/offscreen rendering:

```bash
conda install habitat-sim headless -c conda-forge -c aihabitat
```

6. Install Habitat-Lab from the matching stable branch only after Habitat-Sim
   imports cleanly:

```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
```

7. Return to this project and run:

```bash
python scripts/phase22_habitat_adapter_contract_test.py
```

When a real scene and dataset are available, add a separate real-Habitat smoke
that reads one simulator frame and feeds the same `HabitatObservationAdapter`.

If the EGL/CUDA mismatch persists, use one of these routes:

- Run live Habitat-Sim on native Linux where NVIDIA EGL is available.
- Use a Docker/NVIDIA runtime setup that exposes NVIDIA EGL correctly.
- Generate Habitat observations on another machine and use the current adapter
  path with precomputed RGB-D / semantic / pose frames.
- Continue Phase 3 development against the Phase 2.2 contract outputs until
  live rendering hardware is available.

## Hardware And Data Questions To Confirm

- Which Ubuntu version should be installed in WSL2?
- Does `nvidia-smi` work inside WSL after installation?
- Are HM3D/HM3D-SEM or OVON dataset licenses and download paths available?
- Should the project be copied into the WSL Linux filesystem for faster I/O?
- Is a CPU-only fallback required for machines without NVIDIA GPU access?

## Notes From Upstream Docs

Habitat-Sim's official README lists conda as the recommended installation path,
with `conda-forge` and `aihabitat` channels, and documents a headless build
option for EGL/offscreen use. Habitat-Lab's README recommends installing
Habitat-Sim first, then installing Habitat-Lab from the stable branch.
