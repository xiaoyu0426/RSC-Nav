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
