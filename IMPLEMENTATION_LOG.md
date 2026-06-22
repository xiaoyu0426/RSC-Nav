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

### 下一步

优先级：

```text
1. 验证 MP3D example scene / HM3D-Sem / ReplicaCAD 中至少一个 semantic GT 数据源可用。
2. 在 evaluator 中加入 semantic sensor，上限验证 table / chair / door / wall 的 BEV 投影。
3. 增加 object memory track：category、centroid、footprint、confidence、freshness、last_seen_step。
4. 对 occupied recall 做第二轮几何改进：多高度 bin、端点膨胀、墙面连续性/边界更新。
```
