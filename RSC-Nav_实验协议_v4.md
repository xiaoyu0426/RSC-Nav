# RSC-Nav 实验协议 v4
*(Planning Protocol for Long-Term Semantic-Spatial Memory Navigation)*

---

## 0. 核心问题

RSC-Nav 要验证的不是“agent 会不会走路”，而是：

> 在同一环境被反复使用时，agent 能否构建、检索、复用并修正长期语义-空间记忆，从而减少重复探索并避免旧记忆误导？

实验必须围绕五个基本事实展开：

1. agent 只能通过第一视角 RGB-D、pose 和语义证据获得环境信息。
2. 同一环境中的空间结构、对象和地标具有可复用性。
3. 历史经验可能过期，需要被局部修正。
4. 过长历史不能直接全部塞进隐状态，必须被结构化、检索化。
5. 当环境整体变化过大时，系统不应把旧地图强行更新成新地图，而应识别 context mismatch 并进行 remapping。

v4 的核心边界是：

```text
same-context local change
-> adaptive memory update

global context change
-> context remapping
```

A->B 跨环境加载实验只作为 global mismatch stress test，不作为未来处理环境大变的最终更新方法。

context remapping 的默认语义是保留旧 context：检测到 mismatch 后创建或切换到新 `context_id`，新观测写入新 context；旧 `context_id` 下的 memory 不被覆写或删除，只在当前任务中降权、隔离或归档，未来回到旧环境时仍可检索。

---

## 1. 实验对象

RSC-Nav 的实验对象是一个长期记忆导航系统：

```text
RGB-D + pose + semantic evidence
-> allocentric BEV memory
-> semantic object memory
-> landmark-topological memory
-> goal-conditioned retrieval
-> waypoint / stop decision
-> adaptive update or context remapping
```

系统维护五类记忆或状态：

- occupancy / explored memory
- semantic evidence memory
- landmark-topological memory
- memory state: active / stale / missing / relocated
- context state: context_id / context_confidence / context_mismatch

每个 semantic / landmark memory item 至少包含：

```text
id
semantic_label or semantic_embedding
bev_position
confidence
freshness
last_seen_time
visit_count
negative_evidence_count
status
context_id or context_embedding
source_view_ids
```

`context_id` 是 memory isolation 的主键。context mismatch 后，系统应保留旧 `context_id` 的 memory，并为新环境创建新的 memory context。

---

## 2. 仿真环境

主实验在室内仿真中完成。

基础环境：

- Habitat-Lab / Habitat-Sim
- HM3D / HM3D-OVON / ObjectNav-compatible HM3D
- RGB-D observation
- oracle pose / depth as default
- simulator semantic labels or precomputed semantic evidence

默认不把 SLAM、开放词汇检测器、真实机器人部署作为主变量。CLIP 或其他开放词汇感知模型可以作为扩展设置；论文主线仍然可以使用 simulator semantic labels 来隔离长期记忆机制本身。

主变量只有一个：

> 是否存在可检索、可复用、可修正、可按 context 隔离的长期语义-空间记忆。

---

## 3. Episode 设计

使用 repeated-use proxy episodes 表示同一环境中的多次任务访问。

```text
scene_i:
  task_1: explore / find object_a / write memory
  task_2: find object_b / reuse memory
  task_3: find object_c near object_a / retrieve landmark relation
  task_4: conflict or scene-variant test / update memory
```

`task_1` 主要用于写入记忆。  
长期记忆收益从 `task_index >= 2` 开始统计。

reset / carried 对照必须共享：

- scene
- start pose
- task sequence
- goal specification
- max steps
- success radius
- action space
- semantic evidence source
- pose / depth source

---

## 4. 实验主张

### Claim 1: Memory Reuse

长期记忆应减少后续任务中的重复探索。

对照：

```text
memory reset
vs
memory carried
```

核心指标：

- Success Rate
- SPL
- Path Length
- Distance-to-Goal
- Memory Reuse Gain
- Exploration Efficiency

### Claim 2: Landmark-Conditioned Retrieval

地标和拓扑节点应提升目标检索和可解释性，尤其是 relation goal。

对照：

```text
BEV-only carried
vs
RSC-full carried
vs
RSC-full carried without landmark retrieval
```

核心指标：

- Retrieval Hit@K
- Anchor Hit@K
- target-region Hit@K
- irrelevant retrieval rate
- retrieval score decomposition

### Claim 3: Long-Horizon Memory Retention

当历史任务变长并加入无关任务时，目标相关记忆仍应能被检索。

任务形式：

```text
task_1: find sofa
task_2: find cabinet
task_3: find sink
task_4: find bed
task_5: find lamp near sofa
```

对照：

```text
short history
vs
long history with distractor tasks

goal-conditioned retrieval
vs
random / recency-only retrieval
```

实验层级：

- 主评估：retrieval-level，评估长历史和 distractor 条件下的目标相关记忆检索能力。
- 补充评估：navigation-level，在代表性设置中报告少量导航结果。

核心指标：

- Long-Horizon Memory Retention
- Retrieval Hit@K
- Anchor Hit@K
- Memory Reuse Gain under distractors
- stale / irrelevant retrieval rate

### Claim 4: Same-Context Adaptive Reconfiguration

当同一环境中的新观测与旧记忆局部冲突时，系统应降低旧记忆影响，并更新 memory state，而不是重置整张地图。

基础对照：

```text
carried-stale
vs
carried-adaptive
vs
memory-reset
```

关键消融：

```text
RSC-full carried without freshness penalty
RSC-full carried without status penalty
RSC-full carried without negative evidence
```

核心指标：

- Adaptation Success
- Stale Memory Error Rate
- Recovery Steps
- Relocated Object Hit@K
- Map Correction Latency
- wrong old-location stop rate

### Claim 5: Context Remapping Under Global Change

当环境整体变化过大时，系统应识别 context mismatch，避免旧环境记忆继续作为当前任务主上下文。

基础对照：

```text
forced-single-context
vs
context-remapping
```

核心指标：

- Remapping Trigger Accuracy
- Context Selection Accuracy
- Cross-Context Interference Rate
- Old-Context Retrieval Error
- New-Context Recovery Steps

Claim 5 不替代 Claim 4。A->B 可作为早期 stress test 或 failure-mode visualization，但最终方法应是 context remapping，而不是把 global change 全部交给 adaptive decay。

Claim 5 的关键不是“清空旧记忆”，而是“保留旧记忆但避免当前任务误用”。因此评估应关注 context selection、cross-context interference 和旧 context 回访时的可复用性。

---

## 5. 冲突实验分层

### Level 1: Semantic-Evidence Perturbation

最轻量、最可控，用于验证 memory update rule 的最小因果闭环。

```text
task_1:
  object_a observed at location_x

perturbation:
  object_a relocated / removed / confidence degraded in semantic evidence

task_2:
  search object_a or object_b near object_a
```

### Level 2: Simulation Scene Variant

在仿真观测层验证旧记忆修正，而不只停留在 metadata perturbation。

```text
scene_i_v1:
  object_a at location_x
  agent explores and writes memory

scene_i_v2:
  same global layout
  object_a relocated / removed / replaced
  agent receives new RGB-D / semantic observation
  memory must be corrected
```

核心对照：

```text
carried-stale on scene_i_v2
vs
carried-adaptive on scene_i_v2
vs
memory-reset on scene_i_v2
```

### Level 3: Global Context Stress / Remapping

用于验证整体环境变化不应被解释为同一 context 内的局部更新。

```text
scene_A:
  write memory context_A

scene_B:
  load or retrieve context_A
  detect global mismatch
  create / switch context_B
```

预期行为：

```text
context_A memory:
  retained and archived

context_B memory:
  newly written and selected for current task
```

核心对照：

```text
forced-single-context update
vs
context-remapping
```

A->B 衰减曲线或拼接 GIF 只属于该层的 stress visualization；它说明 prior/live 分离和旧 context weakening 是否存在，但不构成大变环境的最终处理方法。

---

## 6. 阶段性路线

阶段划分只作为实验依赖关系，不作为本文主张本身。

```text
Phase 0-1:
  data contract and memory core

Phase 2:
  BEV / semantic BEV / object memory substrate

Phase 3:
  landmark retrieval and topological memory

Phase 4:
  context remapping gate

Phase 5+:
  navigation evaluation, ablation, thesis packaging
```

Phase 2 的具体实现、可视化、A/B stress test 和阶段性结论单独记录在 Phase 2 阶段性执行文档中。实验协议只保留其在全局实验中的角色：

```text
Phase 2 provides stable semantic-spatial memory substrate.
Phase 3 uses it for retrieval.
Phase 4 decides when it belongs to the wrong context.
```

各阶段的执行细节、当前进度、待补事项和可视化产物统一维护在 `docs/phase_docs/`。

---

## 7. 最小毕业版本

必须完成：

1. Habitat/HM3D 仿真下 repeated-use episodes。
2. BEV + semantic + landmark memory。
3. reset / carried 对照。
4. BEV-only / RSC-full 对照。
5. stale / adaptive local update 对照。
6. long-history distractor retrieval 分析。
7. 至少一种 same-context local perturbation。
8. 至少一种 global context remapping stress test 或 remapping 对照。
9. 指标、消融、可视化和失败案例分析。

---

> **版本:** v4  
> **最后更新时间:** 2026-06-27
