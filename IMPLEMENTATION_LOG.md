# RSC-Nav Implementation Log

> 本文档记录实际实现过程。目标是说明技术流程、阶段产出和当前状态，不记录过细流水账。

---

## 2026-06-15: Phase 0 + Phase 1 Memory Smoke Test

### 目标

在接入 Habitat / HM3D 之前，先完成最小记忆机制闭环：

```text
write -> retrieve -> perturb -> stale retrieval -> reconfigured retrieval
```

该阶段验证的不是完整导航能力，而是 RSC-Nav 的长期语义-空间记忆是否具备最基本的构建、检索和自适应更新能力。

### Phase 0: Data Contract

新增 schema：

```text
schemas/memory_item_schema.json
schemas/episode_schema.json
schemas/perturbation_schema.json
```

冻结了三个核心数据契约：

- `MemoryItem`：语义/地标记忆单元，包括位置、置信度、新鲜度、负证据、状态等字段。
- `Episode`：repeated-use proxy episode 的基本结构。
- `Perturbation`：对象移动、消失、置信度下降等记忆冲突事件。

### Phase 1: Memory Core

新增核心代码：

```text
src/rsc_nav_memory.py
```

实现了最小长期语义-空间记忆：

- 写入语义对象记忆。
- 目标条件检索。
- `confirm / weaken / relocate / overwrite` 的规则式更新雏形。
- `active / stale / missing / relocated` 记忆状态。
- 检索分数分解：

```text
semantic match
spatial proximity
confidence
freshness
status penalty
```

### Smoke Test

新增脚本：

```text
scripts/phase01_smoke_test.py
```

运行方式：

```powershell
python scripts\phase01_smoke_test.py
```

测试设置：

```text
初始记忆:
  sofa at (2, 2)
  table at (5, 3)
  lamp at (8, 8)

扰动:
  sofa 从 (2, 2) relocation 到 (8, 7)

对照:
  carried-stale: 不修正旧记忆
  carried-reconfigured: 降权旧 sofa，写入新 sofa
```

输出：

```text
outputs/phase01/phase01_smoke_log.json
outputs/phase01/phase01_smoke_visualization.png
```

关键结果：

```text
Stale top-1: sofa_001 (2, 2)
Adaptive top-1: sofa_004 (8, 7)
```

说明：

- 不做 adaptive update 时，检索仍指向旧 sofa 位置。
- 启用 reconfiguration 后，旧 sofa 被标记为 `relocated` 并降权，新 sofa 成为 top-1 检索结果。
- 该结果验证了最小因果闭环：新观测可以改变长期语义-空间记忆状态，并影响后续目标检索。

### 当前状态

Phase 0/1 已完成本地最小闭环。

当前实现仍是 synthetic grid，不是 Habitat RGB-D 投影结果。正式 BEV memory 将在 Phase 2 开始实现。

---

## 2026-06-15: Phase 2 Synthetic BEV Memory

### 目标

在接入 Habitat / HM3D 前，先完成一个可解释的 synthetic BEV memory 原型：

```text
synthetic RGB-D-like observation + pose
-> BEV projection
-> occupancy / explored map
-> semantic evidence map
-> long-term semantic-spatial memory
-> goal-conditioned retrieval visualization
```

该阶段的目标不是训练导航策略，而是把 Phase 1 的“长期语义-空间记忆”放到一个更接近 embodied navigation 的地图载体中。也就是说，记忆不再只是手写坐标表，而是由 agent pose 和局部观测逐步投影、累积、可视化得到。

### 新增实现

新增代码：

```text
src/bev_memory.py
scripts/phase02_bev_smoke_test.py
```

`src/bev_memory.py` 实现了最小 BEV 记忆结构：

- `AgentPose`：agent 在平面网格中的位置和朝向。
- `SyntheticObservation`：模拟 RGB-D / semantic detector 的局部观测。
- `BEVMemory`：维护 occupancy log-odds、explored cells、semantic evidence、trajectory。
- ray projection：用简化 ray endpoint 和 Bresenham line 更新 free / occupied cells。
- semantic projection：把语义观测投影到 BEV grid，并保留 label、confidence、time、source view。

`scripts/phase02_bev_smoke_test.py` 负责串联：

```text
synthetic observations
-> BEVMemory.update_from_observation(...)
-> SemanticSpatialMemory.observe(...)
-> retrieve(goal_label="sofa")
-> write log and figures
```

### 输出

运行方式：

```powershell
python scripts\phase02_bev_smoke_test.py
```

输出文件：

```text
outputs/phase02/phase02_log.json
outputs/phase02/bev_occupancy.png
outputs/phase02/bev_semantic.png
outputs/phase02/bev_memory_overlay.png
outputs/phase02/bev_update_sequence.png
```

关键结果：

```text
Explored cells: 33
Occupied cells: 6
Semantic cells: 4
Sofa retrieval top-1: sofa_004 (7, 9)
```

### 可视化说明

- `bev_occupancy.png`：展示 unknown / explored free / occupied cells，以及 agent 运动轨迹。
- `bev_semantic.png`：展示语义证据被投影到 BEV 后的位置和置信度。
- `bev_memory_overlay.png`：把占据地图、语义证据、长期记忆节点和 goal-conditioned retrieval 结果叠加到同一张图。
- `bev_update_sequence.png`：展示从 t=1 到 t=4，agent 逐步探索空间并累积语义证据的过程。

### 当前状态

Phase 2 synthetic BEV memory 已完成最小闭环。

当前实现仍然是 synthetic observation，不是 Habitat 真实 RGB-D / semantic segmentation 输出。下一步应进入 Phase 2+ / Phase 3 的接口化工作：把 synthetic observation 替换为 Habitat episode 中的 RGB-D、pose、semantic detector 输出，并保留同一套 BEV memory / long-term memory API。

---

## 2026-06-15: Phase 2.1 Observation Interface

### 目标

在进入 Habitat 之前，先把 Phase 2 的输入从 source-specific synthetic observation 改造成统一观测契约。

该阶段只做到 Habitat 前一层：不安装 Habitat，不加载真实 3D scene，不训练导航策略，也不进入 Phase 3 的 Landmark Retrieval。目标是让 BEV memory 后续可以同时接 synthetic、mock Habitat 和真实 Habitat 输入。

### 新增实现

新增代码：

```text
src/observation_types.py
src/observation_adapter.py
scripts/phase21_observation_adapter_test.py
```

`src/observation_types.py` 定义统一观测结构：

- `AgentPose`
- `CameraIntrinsics`
- `ObservationRay`
- `ObservationFrame`
- `SyntheticObservation`

`src/observation_adapter.py` 实现两个 adapter：

- `SyntheticObservationAdapter`：将 Phase 2 synthetic observation 转为 `ObservationFrame`。
- `MockHabitatObservationAdapter`：将 Habitat-like dict 转为 `ObservationFrame`，用于在不安装 Habitat 的情况下提前固定字段和接口形状。

`src/bev_memory.py` 新增：

```text
BEVMemory.update_from_frame(frame)
```

旧入口仍保留：

```text
BEVMemory.update_from_observation(observation)
```

旧入口内部会先通过 adapter 转成 `ObservationFrame`，再调用 `update_from_frame(...)`，因此 Phase 2 脚本保持兼容。

### Smoke Test

运行方式：

```powershell
python scripts\phase21_observation_adapter_test.py
```

输出文件：

```text
outputs/phase21/phase21_log.json
outputs/phase21/observation_frame_debug.png
outputs/phase21/adapter_bev_overlay.png
```

关键结果：

```text
Unified frames: 4
Mock Habitat frame rays: 3
Sofa retrieval top-1: sofa_004 (7, 9)
```

### 可视化说明

- `observation_frame_debug.png`：展示统一 `ObservationFrame` 中的 agent pose、heading、projected rays 和 semantic endpoints；标题中包含 mock Habitat adapter 生成的 rgb / depth shape。
- `adapter_bev_overlay.png`：展示 BEV memory 已经只通过 `ObservationFrame` 更新，同时仍能产生长期语义-空间记忆节点和 sofa goal-conditioned retrieval。

### 当前状态

Phase 2.1 已完成到 Habitat 前一层。

当前已完成：

```text
source-specific observation
-> adapter
-> ObservationFrame
-> BEVMemory.update_from_frame(...)
-> SemanticSpatialMemory
-> retrieval / visualization
```

下一步如果继续向真实模拟器推进，应进入 Phase 2.2：安装并初始化 Habitat 环境，读取真实 RGB-D / pose / semantic evidence，再转换为同一个 `ObservationFrame`。原始 Phase 3 仍保留为 Landmark Retrieval，不与 Habitat 接入混淆。

---

## 2026-06-16: Phase 2.2 Habitat Adapter Contract

### 目标

在真实 Habitat-Sim / Habitat-Lab 安装完成前，先补齐项目侧的 Phase 2.2 接口契约：

```text
Habitat-style rgb / depth / semantic / pose
-> HabitatObservationAdapter
-> ObservationFrame
-> BEVMemory.update_from_frame(...)
-> SemanticSpatialMemory retrieval
```

该阶段仍不进入 Phase 3 Landmark Retrieval，也不启动 ObjectNav baseline 或训练。真实 Habitat 运行需要 WSL2 Linux 发行版、匹配的 Habitat-Sim / Habitat-Lab 环境，以及可用 scene dataset。

### 新增实现

新增/修改代码：

```text
src/observation_adapter.py
scripts/phase22_habitat_adapter_contract_test.py
docs/phase22_habitat_wsl2_setup.md
envs/rscnav-habitat22.yml
```

`src/observation_adapter.py` 新增：

- `HabitatObservationAdapter`：从 Habitat 风格的 `rgb`、`depth`、`semantic`、`pose` 字段构造统一 `ObservationFrame`。
- depth column sampling：从 depth 图像按水平 FOV 采样 rays。
- semantic id mapping：把 semantic id 转成 object label，并生成 `hit_type="object"` 的 `ObservationRay`。
- fallback intrinsics：没有显式 camera intrinsics 时，根据 image shape 和 HFOV 估计 pinhole intrinsics。

`src/bev_memory.py` 修复：

- 出界 ray 不再直接丢弃；会裁剪到网格内最后一个 cell，并继续更新 free / explored cells。
- 补齐 `Optional` 类型导入，便于 `typing.get_type_hints(...)` 等检查。

`scripts/phase02_bev_smoke_test.py` 和 `scripts/phase21_observation_adapter_test.py` 新增最小断言，避免只生成图而不检查关键 contract。

### Smoke Test

运行方式：

```powershell
python scripts\phase22_habitat_adapter_contract_test.py
```

输出文件：

```text
outputs/phase22/phase22_log.json
outputs/phase22/observation_frame.json
outputs/phase22/habitat_like_inputs.png
outputs/phase22/habitat_adapter_bev_overlay.png
```

### 当前环境状态

本机 Windows 侧探测结果：

```text
conda 24.9.2
NVIDIA GeForce RTX 3060, 12 GB VRAM
NVIDIA driver 591.86
CUDA version reported by nvidia-smi: 13.1
```

但 `wsl -l -v` 显示当前没有已安装的 Linux 发行版，因此不能在 WSL2 内直接安装并运行真实 Habitat-Sim。真实 Habitat smoke test 需要先安装 WSL2 Ubuntu，并确认 WSL 内 `nvidia-smi` 可用。

### 当前状态

Phase 2 项目侧闭环已完成到：

```text
synthetic BEV memory
-> unified ObservationFrame
-> mock Habitat dict
-> Habitat-style RGB-D / semantic adapter contract
-> BEV update / long-term memory retrieval
```

剩余外部依赖：

- 安装 WSL2 Linux distribution。
- 在 WSL 内创建 `rscnav-habitat22` conda 环境。
- 安装匹配版本的 Habitat-Sim / Habitat-Lab。
- 准备 Habitat test scene 或 HM3D/HM3D-SEM 数据路径。
- 增加真实 simulator one-frame smoke test。

---

## 2026-06-16: WSL2 Habitat Environment Attempt

### 已完成

在 Windows 端由当前 Codex 指挥 WSL2 搭建环境：

```text
RSCNav-Ubuntu-22.04
Ubuntu 22.04.5 LTS
WSL2 kernel 6.6.87.2-microsoft-standard-WSL2
RTX 3060 visible through nvidia-smi
Miniforge: /opt/conda
conda env: rscnav-habitat22
Python: CPython 3.9.23
habitat-sim: 0.3.3 headless
habitat-lab: 0.3.3
```

PowerShell conda 启动报错也已修复：

- `conda config --set auto_activate_base false`
- 在用户 PowerShell profile 中为 conda hook 前置 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`

### 验证结果

WSL conda 环境中通过：

```bash
python -c "import habitat_sim, habitat; print(habitat_sim.__version__, habitat.__version__)"
python scripts/phase22_habitat_adapter_contract_test.py
```

Phase 2.2 contract smoke 输出：

```text
outputs/phase22/phase22_log.json
outputs/phase22/observation_frame.json
outputs/phase22/habitat_like_inputs.png
outputs/phase22/habitat_adapter_bev_overlay.png
```

### 当前阻塞

真实 Habitat-Sim 实时渲染 smoke：

```bash
python scripts/phase22_habitat_sim_none_smoke.py
```

当前失败于 EGL context：

```text
Platform::WindowlessEglApplication::tryCreateContext():
unable to find CUDA device 0 among 1 EGL devices in total
WindowlessContext: Unable to create windowless context
```

诊断：

- WSL 内 `nvidia-smi` 可见 RTX 3060。
- CUDA library 通过 `/usr/lib/wsl/lib` 暴露。
- 但 EGL vendor 当前只有 Mesa：`/usr/share/glvnd/egl_vendor.d/50_mesa.json`。
- Habitat-Sim headless 需要 NVIDIA EGL 与 CUDA device 对齐；当前 WSL2 图形栈未提供该条件。

### 状态判定

Phase 2.2 项目侧和 WSL conda/Habitat import 已完成。

真实 simulator live rendering 需要进一步硬件/系统层处理：

- 原生 Linux / 双系统 NVIDIA EGL；
- 或 Docker + NVIDIA runtime 暴露 EGL；
- 或先使用预计算 Habitat observations 继续 Phase 3。
