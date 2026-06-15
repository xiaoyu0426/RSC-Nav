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

## Next: Phase 2 BEV Memory

计划先做 synthetic BEV projector，再接 Habitat。

Phase 2 目标：

```text
RGB-D-like observation + pose
-> BEV projection
-> occupancy map
-> explored map
-> semantic evidence map
-> visualization
```

预期输出：

```text
outputs/phase02/bev_occupancy.png
outputs/phase02/bev_semantic.png
outputs/phase02/bev_memory_overlay.png
outputs/phase02/phase02_log.json
```

时间预估：

- synthetic Phase 2：3-5 天。
- Habitat 接入版：约 2-3 周。

