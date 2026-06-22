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

---

## 2026-06-22: Phase 2.3 / 2.4 Real Habitat-Sim Live BEV Evaluation

### 当前目标

当前 live 目标已收敛为：

```text
真实 Habitat-Sim headless live rendering
-> RGB / depth / pose
-> allocentric BEV semantic-spatial memory
-> 墙面、门、桌子、椅子的位置、类别、置信度和 freshness
-> oracle map / semantic GT / object stability 自动验收
-> 支撑后续长期记忆复用与更新实验
```

当前子目标顺序：

```text
1. dense depth + pose 几何 BEV 稳定性自动验收
2. Habitat semantic GT 上限版
3. object memory 的 confidence / freshness / ID stability
```

### 远端开发机环境

开发机 workspace：

```text
ssh yujiexiao@39.101.65.229 -p 1040
~/RSC_Nav
conda env: rscnav-habitat22
```

约束：

```text
开发机禁止 push，仅允许 pull / fast-forward 更新。
本机提交后通过 git bundle 传到开发机，再在开发机 git pull --ff-only。
```

已验证真实 headless rendering：

```text
scene: /workspace/yujiexiao/.rscnav/habitat_data/versioned_data/habitat_test_scenes/apartment_1.glb
renderer: NVIDIA A800-SXM4-80GB
Habitat-Sim RGB / depth rendering: pass
```

### 新增实现

相关 commits：

```text
d526e6a Add dense BEV geometry evaluation
1aec25d Support Habitat quaternion sensor rotations
6a34a92 Fix BEV metric safe division
d866aaf Use navigable paths for BEV geometry eval
```

新增文件：

```text
src/dense_bev_mapper.py
scripts/phase24_bev_geometry_eval.py
```

`src/dense_bev_mapper.py` 实现：

- dense depth deprojection：depth + camera intrinsics + sensor pose -> world points。
- allocentric BEV grid：世界坐标系固定，agent 移动时地图不随视角旋转。
- ray carving：从 agent cell 到 depth endpoint 更新 free cells。
- height-band occupied update：根据相对地面高度筛选墙/障碍证据。
- occupancy log-odds、explored mask、observation count、confidence map。
- Habitat navmesh oracle mask。
- 自动指标：free-space IoU、precision / recall、occupied precision / recall、occupied boundary Chamfer。

`scripts/phase24_bev_geometry_eval.py` 实现：

- 在真实 Habitat-Sim 场景中执行评估轨迹。
- 默认使用 navmesh shortest-path waypoints，而不是随机动作序列，以避免轨迹撞墙或覆盖过短。
- 输出 RGB / depth sample frames。
- 输出 `ours_bev.png`、`oracle_bev.png`、`diff_bev.png`、`confidence.png`、`metrics.json`、`summary.html`。

### 验收结果

短动作序列首次验收：

```text
remote: ~/RSC_Nav/outputs/phase24_bev_eval/apartment_1_20260622-192134/
local:  outputs/phase24_bev_eval/apartment_1_20260622-192134/

free_iou_observed: 0.1149
free_recall_observed: 1.0000
occupied_precision_observed: 1.0000
occupied_recall_observed: 0.2325
observed_cells: 298
```

结论：几何链路跑通，但轨迹太短，起点靠近 navmesh 边界，不能作为稳定验收依据。

path-mode 评估：

```text
remote: ~/RSC_Nav/outputs/phase24_bev_eval/apartment_1_path_20260622-192448/
local:  outputs/phase24_bev_eval/apartment_1_path_20260622-192448/
tar:    outputs/phase24_bev_eval/rscnav_phase24_apartment_1_path_20260622-192448.tar.gz

scene: /workspace/yujiexiao/.rscnav/habitat_data/versioned_data/habitat_test_scenes/apartment_1.glb
trajectory_mode: path
path_min_distance_m: 3.0
rgbd_resolution: 160
bev_resolution: 0.05
grid_size: 240
sample_stride: 2
```

关键指标：

```text
free_iou_observed:             0.8223
free_precision:                0.8237
free_recall_observed:          0.9980
occupied_precision_observed:   0.9501
occupied_recall_observed:      0.1540
occupied_boundary_chamfer_m:   0.2989
pred_free_cells:               11278
pred_occupied_cells:           381
observed_cells:                11659
oracle_free_observed_cells:    9309
oracle_obstacle_observed_cells:2350
```

保存的验收图：

```text
outputs/phase24_bev_eval/apartment_1_path_20260622-192448/ours_bev.png
outputs/phase24_bev_eval/apartment_1_path_20260622-192448/oracle_bev.png
outputs/phase24_bev_eval/apartment_1_path_20260622-192448/diff_bev.png
outputs/phase24_bev_eval/apartment_1_path_20260622-192448/confidence.png
outputs/phase24_bev_eval/apartment_1_path_20260622-192448/summary.html
outputs/phase24_bev_eval/apartment_1_path_20260622-192448/metrics.json
```

### 当前结论

几何 BEV 第一阶段结论：

```text
pass with reservations
```

已证明：

- 真实 Habitat-Sim headless live rendering 可用。
- depth + pose -> allocentric BEV 的坐标链路可运行。
- agent 沿 navmesh path 移动时，BEV free-space 在世界坐标中相对稳定。
- confidence map 能随观测累积。
- 自动验收产物和结果图已保存。

尚未完成：

- occupied / wall recall 偏低，说明当前 height-band endpoint update 对墙/障碍召回不足。
- `apartment_1.glb` 没有 semantic annotations，不能用于桌子、椅子、门的严格 semantic GT 自动验收。
- 还没有 object-level track、confidence/freshness、ID stability。

### Semantic GT 数据源探测

已在开发机下载 Habitat-Sim 官方 MP3D example scene：

```text
python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene --data-path /workspace/yujiexiao/.rscnav/habitat_data
```

数据路径：

```text
scene:   /workspace/yujiexiao/.rscnav/habitat_data/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.glb
dataset: /workspace/yujiexiao/.rscnav/habitat_data/versioned_data/mp3d_example_scene_1.1/mp3d.scene_dataset_config.json
navmesh: /workspace/yujiexiao/.rscnav/habitat_data/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.navmesh
house:   /workspace/yujiexiao/.rscnav/habitat_data/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.house
```

语义探测结果：

```text
pathfinder_loaded: true
levels: 1
regions: 10
objects: 187
num_categories: 31
wall: 10
door: 9
table: 4
chair: 11
sofa: 2
bed: 2
```

semantic sensor smoke：

```text
semantic_sensor_shape: 64 x 64
semantic_sensor_unique_count: 2 in one random view
```

结论：

```text
MP3D example scene 可作为 semantic GT 上限版的第一测试场景。
它包含当前目标类别 wall / door / table / chair，并可通过 semantic sensor 输出 instance id。
```

### Semantic GT 上限版验收

新增/更新：

```text
scripts/phase24_bev_geometry_eval.py
src/dense_bev_mapper.py
```

新增能力：

```text
depth + semantic sensor instance id + pose
-> semantic GT BEV projection
-> per-class BEV cells
-> per-instance visible track
-> confidence / first_seen / last_seen
-> semantic_tracks.json
```

运行场景：

```text
scene:   /workspace/yujiexiao/.rscnav/habitat_data/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.glb
dataset: /workspace/yujiexiao/.rscnav/habitat_data/versioned_data/mp3d_example_scene_1.1/mp3d.scene_dataset_config.json
categories: wall, door, table, chair
trajectory_mode: path
path_min_distance_m: 4.0
```

输出：

```text
remote: ~/RSC_Nav/outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/
local:  outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/
tar:    outputs/phase24_bev_eval/rscnav_phase24_mp3d_semantic_20260622-193635.tar.gz
```

保存的验收图：

```text
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/ours_bev.png
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/oracle_bev.png
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/diff_bev.png
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/confidence.png
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/semantic_bev.png
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/semantic_confidence.png
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/frame_000_semantic.png
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/frame_019_semantic.png
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/frame_038_semantic.png
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/semantic_tracks.json
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/summary.html
outputs/phase24_bev_eval/mp3d_semantic_20260622-193635/metrics.json
```

几何指标：

```text
free_iou_observed:             0.5652
free_precision:                0.5814
free_recall_observed:          0.9530
occupied_precision_observed:   0.7446
occupied_recall_observed:      0.1664
occupied_boundary_chamfer_m:   0.1459
observed_cells:                24882
```

语义指标：

```text
indexed_target_instances: 34
observed_target_instances: 29
semantic_cells: 4589
wall_cells: 3019
door_cells: 279
table_cells: 247
chair_cells: 1044
mean_centroid_error_m: 2.2008
```

语义上限版当前结论：

```text
pass as semantic projection smoke
not yet pass as final object stability metric
```

已证明：

- Habitat semantic sensor 可在开发机 headless 环境中渲染目标类别 instance id。
- wall / door / table / chair 能投影到同一个 allocentric BEV。
- 语义 evidence、confidence、visible instance track、first_seen / last_seen 能保存为验收产物。
- 验收图和 JSON 已保存。

需要改进：

- 当前 centroid error 使用可见表面点云中心对比完整 object AABB 中心；对墙、门、桌椅的部分可见视角偏差较大，不能作为最终 object stability 指标。
- 下一轮应使用 semantic instance id 的跨帧稳定性、footprint IoU、dominant id purity、ID switch / fragmentation 作为 object memory 评价。
- MP3D 场景几何 free IoU 低于 apartment_1，说明几何占用层仍需做多高度 bin、边界/墙面连续性和更合理的 obstacle recall。

### Object Freshness / Stability Smoke

新增 commit：

```text
1a05ad7 Add semantic object freshness metrics
3b2d11d Resample two-point Habitat paths
```

目的：

```text
把 semantic GT visible instance tracks 升级为 object-memory 雏形：
semantic_id
category
centroid_xz
confidence
freshness
age_steps
visible_steps
visibility_segments
fragmentation_count
first_seen_step / last_seen_step
```

修复：

```text
Habitat shortest path 有时只返回 start/end 两点。
旧版 _resample_polyline 在两点路径上没有插值，导致 freshness 复验只跑 2 帧。
已修复为两点路径也插值成 max_steps + 1 个 waypoints。
```

复验输出：

```text
remote: ~/RSC_Nav/outputs/phase24_bev_eval/mp3d_freshness_resampled_20260622-194233/
local:  outputs/phase24_bev_eval/mp3d_freshness_resampled_20260622-194233/
tar:    outputs/phase24_bev_eval/rscnav_phase24_mp3d_freshness_resampled_20260622-194233.tar.gz
```

几何指标：

```text
free_iou_observed:             0.5437
free_recall_observed:          0.9965
occupied_precision_observed:   0.9592
occupied_recall_observed:      0.0909
occupied_boundary_chamfer_m:   0.1416
observed_cells:                13503
```

object memory smoke 指标：

```text
indexed_target_instances: 34
observed_target_instances: 18
semantic_cells: 4511
wall_cells: 3167
door_cells: 3
table_cells: 130
chair_cells: 1211
mean_centroid_error_m: 1.5096
mean_fragmentation_count: 0.5
id_switches_upper_bound: 0
mean_freshness: 0.8181
```

示例：

```text
semantic_id 95 wall:
visible_steps: 0..38
confidence: 1.0
freshness: 1.0
fragmentation_count: 0

semantic_id 141 chair:
visible_steps: 0..36
confidence: 1.0
freshness: 0.9048
fragmentation_count: 0

semantic_id 171 wall:
visible_steps include gaps
visibility_segments: 3
fragmentation_count: 2
```

当前结论：

```text
object freshness smoke passed in GT-instance upper-bound mode.
```

已证明：

- object memory 的最小字段可以从 Habitat semantic GT 自动生成。
- confidence / freshness / visible_steps / fragmentation 可以保存并自动验收。
- 在 GT semantic id 条件下，ID switch upper bound 为 0。

尚未完成：

- 还没有独立的长期 memory store 持久化/重载/reuse/update 实验。
- 还没有 detector-driven object association，因此真实 ID switch / merge / split 仍未验收。
- door/table/chair 的覆盖依赖路径和视角，后续需要主动选择能覆盖目标类别的 semantic path。

### Geometry Occupied / Wall Recall Iteration

新增 commit：

```text
78fc5f8 Add BEV obstacle dilation metrics
```

问题：

```text
旧 mapper 只在 depth endpoint 的单个 BEV cell 加 occupied evidence。
墙面和障碍在 BEV 中过细、断裂，导致 occupied / wall recall 偏低。
```

修改：

```text
DenseBEVConfig.obstacle_dilation_radius_cells
--obstacle-dilation-cells
free_f1_observed
occupied_f1_observed
```

方法：

```text
只对判定为 obstacle 的 depth endpoint 做小半径 BEV dilation。
free-space ray carving 逻辑保持不变。
```

apartment_1 dilation sweep：

```text
remote: ~/RSC_Nav/outputs/phase24_bev_eval/apartment_dilation_sweep_20260622-194705/
local:  outputs/phase24_bev_eval/apartment_dilation_sweep_20260622-194705/
tar:    outputs/phase24_bev_eval/rscnav_phase24_apartment_dilation_sweep_20260622-194705.tar.gz
```

结果：

```text
radius 0:
  free_iou_observed:           0.8223
  free_f1_observed:            0.9025
  occupied_precision_observed: 0.9501
  occupied_recall_observed:    0.1540
  occupied_f1_observed:        0.2651
  boundary_chamfer_m:          0.2989

radius 1:
  free_iou_observed:           0.8554
  free_f1_observed:            0.9221
  occupied_precision_observed: 0.9479
  occupied_recall_observed:    0.3850
  occupied_f1_observed:        0.5476
  boundary_chamfer_m:          0.2894

radius 2:
  free_iou_observed:           0.8961
  free_f1_observed:            0.9452
  occupied_precision_observed: 0.9375
  occupied_recall_observed:    0.6329
  occupied_f1_observed:        0.7557
  boundary_chamfer_m:          0.2730
```

MP3D semantic scene radius 2 复验：

```text
remote: ~/RSC_Nav/outputs/phase24_bev_eval/mp3d_dilation2_20260622-194954/
local:  outputs/phase24_bev_eval/mp3d_dilation2_20260622-194954/
tar:    outputs/phase24_bev_eval/rscnav_phase24_mp3d_dilation2_20260622-194954.tar.gz
```

结果：

```text
free_iou_observed:             0.6458
free_recall_observed:          0.9329
free_f1_observed:              0.7848
occupied_precision_observed:   0.8947
occupied_recall_observed:      0.5616
occupied_f1_observed:          0.6900
occupied_boundary_chamfer_m:   0.1285
```

结论：

```text
radius 2 is the current default.
```

理由：

- apartment_1 上 occupied recall 从 0.1540 提升到 0.6329，occupied F1 从 0.2651 提升到 0.7557。
- MP3D semantic scene 上 occupied recall 达到 0.5616，occupied F1 达到 0.6900。
- precision 仍保持较高，apartment_1 为 0.9375，MP3D 为 0.8947。
- 可视化未出现整图过度涂黑，墙/障碍边界更连续。

### Phase 2.5 Persistent Object Memory Store

新增 commit：

```text
0ec6177 Add persistent object memory store eval
```

新增文件：

```text
src/object_memory_store.py
scripts/phase25_object_memory_store_eval.py
```

目标：

```text
把 Phase 2.4 的 semantic_tracks.json 从一次性验收输出升级为可持久化 object memory：
save -> load -> retrieve -> decay -> replay/update
```

`ObjectMemoryItem` 字段：

```text
id
semantic_id
object_id
category
centroid_xz
confidence
freshness
first_seen_step
last_seen_step
visible_steps
footprint_cells
fragmentation_count
status
source
```

运行输入：

```text
metrics: outputs/phase24_bev_eval/mp3d_dilation2_20260622-194954/metrics.json
```

远端验收输出：

```text
remote: ~/RSC_Nav/outputs/phase25_object_memory/mp3d_store_20260622-195605/
local:  outputs/phase25_object_memory/mp3d_store_20260622-195605/
tar:    outputs/phase25_object_memory/rscnav_phase25_mp3d_store_20260622-195605.tar.gz
```

保存文件：

```text
object_memory.json
object_memory_decayed.json
object_memory_replayed.json
object_memory_eval.json
object_memory_plot.png
summary.html
```

关键结果：

```text
initial:
  num_items: 18
  per_class: wall 5, door 3, chair 9, table 1
  mean_confidence: 0.8563
  mean_freshness: 0.8181
  active_items: 13
  stale_items: 3
  missing_items: 2

reload_equal: true

decay +30 steps:
  mean_freshness: 0.1825
  active_items: 0
  stale_items: 14
  missing_items: 4

replay/update same tracks:
  created: 0
  updated: 18
  mean_confidence: 0.8646
  mean_freshness: 0.7782
  active_items: 13
  stale_items: 3
  missing_items: 2
```

结论：

```text
Persistent object memory store smoke passed.
```

已证明：

- wall / door / table / chair 的 object memory 可以从 Habitat semantic GT tracks 自动生成。
- memory 可以持久化到 JSON 并完整 reload。
- freshness 会随时间衰减，状态会从 active 转向 stale / missing。
- 重放同一 semantic tracks 会 update 已有 memory，而不是创建重复对象。
- retrieval 可按类别返回带 score / confidence / freshness / status / centroid 的对象列表。

尚未完成：

- 还没有把 object memory store 接回 live web control UI。
- 还没有跨 episode 的真实 reload + navigation reuse 实验。
- 还没有 detector-driven association，因此 ID switch / merge / split 仍处于 GT upper-bound 阶段。

## Phase 2.6 Live Dense Semantic Memory Control

时间：2026-06-22

提交：

```text
8e0b318 Integrate dense semantic memory into live control
d175aa8 Choose navigable path starts for live control
```

目标：

```text
把离线验收通过的 dense depth BEV、Habitat semantic GT accumulator、persistent object memory store
接回真实 Habitat-Sim headless live control UI，使手控/网页闭环和自动验收使用同一套空间记忆核心。
```

实现内容：

- `scripts/phase23_habitat_control_server.py` 从旧的 sparse ray BEV 切换为 `DenseBEVMapper`。
- live `/api/state` 每步使用 depth + sensor pose 更新 allocentric BEV。
- 传入 `--scene-dataset-config` 时启用 Habitat semantic sensor，生成 semantic BEV、tracks、confidence、freshness。
- 接入 `ObjectMemoryStore`，网页显示对象数量、active 数、mean freshness、per-class memory。
- 增加 `/api/save_memory` 与 `/api/load_memory`，保存/加载 `live_object_memory.json`。
- reset 时从 navmesh 采样一条可走路径，把 agent 放在路径起点并朝向下一个 waypoint，减少随机贴墙出生。

当前开发机 live UI：

```text
url:  http://39.101.65.229:43901/
pid:  2119397
env:  rscnav-habitat22
scene: /workspace/yujiexiao/.rscnav/habitat_data/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.glb
dataset: /workspace/yujiexiao/.rscnav/habitat_data/versioned_data/mp3d_example_scene_1.1/mp3d.scene_dataset_config.json
```

远端验收输出：

```text
remote: ~/RSC_Nav/outputs/phase26_live_control/mp3d_live_pathstart_20260622-201033/
local:  outputs/phase26_live_control/mp3d_live_pathstart_20260622-201033/
```

保存文件：

```text
rgb.jpg
depth.png
bev.png
semantic_bev.png
state.json
live_object_memory.json
live_smoke_summary.json
server.log
server.pid
```

第二轮 live smoke 结果：

```text
step: 31

BEV:
  explored_cells: 7249
  free_cells: 2828
  occupied_cells: 4421
  mean_confidence: 0.8933

Semantic GT BEV:
  indexed_target_instances: 34
  observed_target_instances: 8
  semantic_cells: 1507
  per_class_cells:
    wall: 863
    door: 373
    table: 180
    chair: 91
  mean_freshness: 0.7868
  mean_fragmentation_count: 0.75
  id_switches_upper_bound: 0

Object memory:
  num_items: 8
  per_class:
    wall: 3
    door: 1
    table: 2
    chair: 2
  mean_confidence: 0.8058
  mean_freshness: 0.7868
  active_items: 5
  stale_items: 2
  missing_items: 1
```

结论：

```text
真实 Habitat-Sim headless live rendering 闭环已接入 dense semantic object memory。
网页手控时，BEV 会随本体移动持续累积几何/语义空间记忆，并能保存 object memory JSON。
```

仍需迭代：

- 当前 semantic BEV 仍有边界碎片和稀疏噪声，需要用自动指标约束稳定性。
- mean_centroid_error 仍偏高，不能作为最终 object localization 结论。
- 当前语义仍是 Habitat semantic GT upper-bound；尚未接入 detector-driven association。
- 需要增加 live 自动巡航/路径回放模式，用固定轨迹持续生成可比较的验收图。

## Phase 2.7 Live Path-Step Auto Evaluation

时间：2026-06-22

提交：

```text
e99bd14 Add live memory stability evaluator
ae19fd4 Add live path-step evaluation mode
b0b0e59 Evaluate mature live object stability
```

目标：

```text
把 Phase 2.6 的 live UI smoke 升级为自动验收：
server 提供 path_step 自动巡航动作，evaluator 驱动 live server、保存 checkpoint 图、
记录 object history，并计算 geometry / semantic class coverage / object memory / stability gates。
```

新增/修改：

```text
scripts/phase27_live_control_eval.py
scripts/phase23_habitat_control_server.py
```

关键机制：

- `/api/action` 新增 `path_step`：沿 navmesh path waypoint 推进，并在每个 waypoint 触发真实 Habitat sensor observation。
- `/api/state` 新增 `memory_items`：导出每个 object memory item 的 category、centroid、confidence、freshness、status。
- evaluator 默认使用 `trajectory-mode=path` 与 36 个 `path_step`。
- evaluator 保存 `states_compact.json`、`object_history.json`、`metrics.json`、`summary.html` 和每个 checkpoint 的 RGB/depth/BEV/semantic BEV 图。
- stability gate 改为 mature-tail window：保留 `total_drift_m` 作为发现过程诊断，但 pass/fail 使用最后 6 个观测窗口的 `tail_drift_m`，避免把长墙面逐步被发现时的 centroid 扩展误判为最终记忆不稳定。

远端验收输出：

```text
remote: ~/RSC_Nav/outputs/phase27_live_eval/mp3d_live_tailstable_20260622-202038/
local:  outputs/phase27_live_eval/mp3d_live_tailstable_20260622-202038/
url:    http://39.101.65.229:43901/
```

保存文件：

```text
eval/metrics.json
eval/live_eval_summary.json
eval/object_history.json
eval/states_compact.json
eval/summary.html
eval/step_0000_*.png
eval/step_0006_*.png
eval/step_0012_*.png
eval/step_0018_*.png
eval/step_0024_*.png
eval/step_0030_*.png
eval/step_0036_*.png
eval/step_0036_final_*.png
```

最终自动验收：

```text
passed: true

criteria:
  geometry_ok: true
  semantic_ok: true
  covered_all_classes: true
  memory_ok: true
  stability_ok: true
  max_mean_step_drift_m: 0.45
  max_tail_drift_m: 0.8
  stability_window: 6

final_step: 36

class_coverage:
  wall: 7
  door: 6
  table: 1
  chair: 6

final_memory:
  num_items: 20
  mean_confidence: 0.8476
  mean_freshness: 0.3812
  active_items: 9
  stale_items: 10
  missing_items: 1

final_semantic:
  observed_target_instances: 20
  semantic_cells: 6249
  per_class_cells:
    wall: 5644
    door: 103
    table: 9
    chair: 493
  mean_centroid_error_m: 1.2648
  mean_fragmentation_count: 0.45
  id_switches_upper_bound: 0

final_bev:
  explored_cells: 18557
  free_cells: 9770
  occupied_cells: 8787
  mean_confidence: 0.8907

object_stability:
  tracked_items: 20
  mean_step_drift_m: 0.0525
  max_total_drift_m: 4.9131
  max_tail_drift_m: 0.0076
  mean_confidence: 0.8135
  mean_freshness: 0.7050
```

结论：

```text
当前 live 目标的自动验收链路已经成立：
真实 Habitat-Sim headless live server 可以通过 path_step 自动巡航覆盖 wall / door / table / chair，
BEV semantic-spatial memory 会持续累积对象位置、类别、confidence、freshness，
并通过 geometry / semantic GT / object stability 指标自动保存图像和 JSON 证据。
```

仍需保留的科学问题：

- 当前 semantic signal 仍是 Habitat semantic GT upper-bound，不是 detector-driven semantic perception。
- 物体/墙面的 centroid 在首次发现到逐步扩展阶段会有较大 `total_drift_m`，但 mature tail 已稳定；后续报告中需要明确区分 discovery drift 与 stabilized memory drift。
- `chair` 与 `table` 类在本 MP3D example scene 中像素/格子占比偏小，后续应在 HM3D-Sem / ReplicaCAD 或更多场景上复验。
- 当前 active/stale/missing 由 freshness threshold 定义，需要在长期记忆复用实验中进一步标定。

## Phase 2.8 Live Oracle Geometry Gate

时间：2026-06-22

提交：

```text
8b2cb79 Add oracle geometry gate to live eval
```

目标：

```text
把 live evaluator 的 geometry gate 从“BEV 非空”升级为 navmesh oracle map 对齐指标，
使当前 live 自动验收同时覆盖 oracle map / semantic GT / object stability 三个证据来源。
```

实现内容：

- `phase23_habitat_control_server.py` 在 reset 后缓存当前 BEV origin 对应的 navmesh oracle free mask。
- `/api/state` 新增 `geometry_oracle`，返回 `free_iou_observed`、`occupied_f1_observed`、`occupied_boundary_chamfer_m` 等指标。
- `phase27_live_control_eval.py` 新增 oracle gate：
  - `oracle_enabled == true`
  - `free_iou_observed >= 0.2`
  - `occupied_f1_observed >= 0.05`
- 修正远端 43901 旧进程占端口问题后复跑，确认请求命中新 server payload。

远端验收输出：

```text
remote: ~/RSC_Nav/outputs/phase27_live_eval/mp3d_live_oracle_clean_20260622-202641/
local:  outputs/phase27_live_eval/mp3d_live_oracle_clean_20260622-202641/
url:    http://39.101.65.229:43901/
```

保存文件：

```text
eval/metrics.json
eval/live_eval_summary.json
eval/object_history.json
eval/states_compact.json
eval/summary.html
eval/step_0036_final_bev.png
eval/step_0036_final_semantic_bev.png
```

最终自动验收：

```text
passed: true

criteria:
  geometry_ok: true
  bev_nonempty: true
  oracle_enabled: true
  semantic_ok: true
  covered_all_classes: true
  memory_ok: true
  stability_ok: true

oracle geometry:
  free_iou_observed: 0.7043
  free_precision: 0.8464
  free_recall_observed: 0.8076
  free_f1_observed: 0.8265
  occupied_precision_observed: 0.7473
  occupied_recall_observed: 0.7951
  occupied_f1_observed: 0.7705
  occupied_boundary_chamfer_m: 0.1033

semantic/object:
  wall: 8
  door: 3
  table: 2
  chair: 6
  num_items: 19
  active_items: 11
  mean_confidence: 0.7565
  mean_freshness: 0.8488

object stability:
  tracked_items: 19
  mean_step_drift_m: 0.0477
  max_tail_drift_m: 0.7262
```

结论：

```text
当前真实 Habitat-Sim live 闭环已具备 oracle map / semantic GT / object stability 三类自动验收证据。
这比 Phase 2.7 更完整：geometry 不再只靠非空图像，而是明确对齐 navmesh oracle。
```

下一轮建议：

- 将 oracle gate 的阈值升级为多场景统计阈值，而不是单场景 smoke 阈值。
- 为 `summary.html` 增加 oracle diff 图，便于肉眼定位 free/occupied 错误区域。
- 把 live object memory 的 save/load/replay 做成跨 episode 验收，进入长期记忆复用实验。

## Phase 2.9 Live Memory Reuse / Reload Evaluation

时间：2026-06-22

提交：

```text
e10ff38 Add live memory reuse evaluator
57d0a60 Allow live server without oracle metrics
a82ad2e Treat replay discoveries as new memory
```

目标：

```text
验证 live object memory 不只是当场显示，而是可以 save -> reset -> load -> replay/update。
这为后续长期记忆复用实验提供最小闭环。
```

实现内容：

- `phase23_habitat_control_server.py` 新增 `memory_step`，作为跨 reset 单调递增的 object-memory 时间轴。
  - UI/episode `step` 可以 reset 到 0。
  - `memory_step` 不随 reset 回退，用于 semantic tracks、ObjectMemoryStore update/decay。
- `phase23_habitat_control_server.py` 新增 `--disable-oracle-metrics`。
  - oracle geometry eval 仍可启用。
  - memory reuse eval 可跳过 oracle mask 初始化，避免不必要的启动等待。
- 新增 `scripts/phase28_live_memory_reuse_eval.py`。
  - 第一次 path_step 巡航后 save memory。
  - reset 后验证不是完整沿用旧 memory。
  - load memory 后验证旧 IDs 全部恢复。
  - replay path_step 后验证旧 IDs 保留并有对象被 update。
  - 新发现对象允许作为 new memory，只有重复 ID 才视为 duplicate。

远端验收输出：

```text
remote: ~/RSC_Nav/outputs/phase29_live_memory_reuse/mp3d_reuse_pass_20260622-203909/
local:  outputs/phase29_live_memory_reuse/mp3d_reuse_pass_20260622-203909/
url:    http://39.101.65.229:43901/
```

保存文件：

```text
eval/metrics.json
eval/live_memory_reuse_summary.json
eval/timeline_compact.json
eval/summary.html
eval/037_saved_memory_step_0036_mem_0084_*.png
eval/039_loaded_memory_step_0000_mem_0084_*.png
eval/052_replay_saved_memory_step_0012_mem_0096_*.png
server/live_object_memory.json
```

最终自动验收：

```text
passed: true

criteria:
  reset_did_not_keep_full_memory: true
  load_retained_ok: true
  replay_retained_ok: true
  duplicate_ok: true
  update_ok: true
  replay_active_ok: true
  memory_step_monotonic: true

saved:
  step: 36
  memory_step: 84
  num_items: 8
  per_class:
    wall: 4
    door: 1
    table: 2
    chair: 1
  mean_confidence: 0.8144
  mean_freshness: 0.6786

loaded:
  step: 0
  memory_step: 84
  num_items: 10
  retained_after_load:
    count: 8
    ratio: 1.0

replay:
  step: 12
  memory_step: 96
  num_items: 14
  retained_after_replay:
    count: 8
    ratio: 1.0
  duplicate_item_ids: []
  updated_ids:
    chair_141
    door_24
    table_132
    wall_0
    wall_125
    wall_155
    wall_29
  new_after_replay:
    chair_102
    chair_110
    table_31
    wall_171
    wall_53
    wall_95
```

结论：

```text
live object memory 已通过真实 Habitat-Sim save/reset/load/replay 自动验收。
旧 memory IDs 在 load 与 replay 后 100% 保留；replay 会更新旧对象，也能接纳新发现对象；
memory_step 保证 freshness/update 时间轴跨 reset 不倒退。
```

仍需保留的科学问题：

- 当前 save/load/replay 仍在同一 MP3D example scene 内完成，还不是多场景或跨任务泛化。
- replay 后新出现对象被接纳为 new memory；后续 detector-driven 版本需要更严格的 association/merge/split 判据。
- `freshness` 已有跨 reset 时间轴，但长期真实时间/多 episode 时间尺度还需要实验标定。

## Phase 3.0 Live Oracle Visual Evidence

时间：2026-06-22

提交：

```text
dfb25b7 Save oracle visual layers in live eval
```

目标：

```text
把 oracle geometry gate 从纯数值验收补强为可审计图像验收：
保存 live BEV、oracle map、BEV-vs-oracle diff、semantic BEV 四类 checkpoint 图。
```

实现内容：

- `phase23_habitat_control_server.py` 在 oracle metrics enabled 时返回：
  - `oracle_png`
  - `oracle_diff_png`
- `phase27_live_control_eval.py` 自动保存 oracle/oracle_diff checkpoint 图。
- `summary.html` 增加 BEV / Oracle / Oracle Diff / Semantic BEV 四列。

远端验收输出：

```text
remote: ~/RSC_Nav/outputs/phase30_live_oracle_visuals/mp3d_oracle_visuals_20260622-204418/
local:  outputs/phase30_live_oracle_visuals/mp3d_oracle_visuals_20260622-204418/
url:    http://39.101.65.229:43901/
```

保存文件：

```text
eval/step_0000_oracle.png
eval/step_0000_oracle_diff.png
eval/step_0012_oracle.png
eval/step_0012_oracle_diff.png
eval/step_0024_oracle.png
eval/step_0024_oracle_diff.png
eval/step_0036_oracle.png
eval/step_0036_oracle_diff.png
eval/step_0036_final_bev.png
eval/step_0036_final_oracle.png
eval/step_0036_final_oracle_diff.png
eval/step_0036_final_semantic_bev.png
eval/metrics.json
eval/summary.html
```

最终自动验收：

```text
passed: true

oracle geometry:
  free_iou_observed: 0.7043
  free_f1_observed: 0.8265
  occupied_f1_observed: 0.7705
  occupied_boundary_chamfer_m: 0.1033

semantic/object:
  wall: 8
  door: 3
  table: 2
  chair: 6
  num_items: 19
  mean_confidence: 0.7565
  mean_freshness: 0.8488

object stability:
  tracked_items: 19
  mean_step_drift_m: 0.0477
  max_tail_drift_m: 0.7262
```

结论：

```text
当前 live 自动验收已经同时保存数值证据与图像证据。
Oracle Diff 图可以直接定位 free/occupied 与 navmesh oracle 的一致/错误区域，
比单独的 BEV 或纯 JSON 指标更适合作为后续迭代和论文实验记录。
```

### 下一步

优先级：

```text
1. 验证 MP3D example scene / HM3D-Sem / ReplicaCAD 中至少一个 semantic GT 数据源可用。
2. 在 evaluator 中加入 semantic sensor，上限验证 table / chair / door / wall 的 BEV 投影。
3. 增加 object memory track：category、centroid、footprint、confidence、freshness、last_seen_step。
4. 对 occupied recall 做第二轮几何改进：多高度 bin、端点膨胀、墙面连续性/边界更新。
```
