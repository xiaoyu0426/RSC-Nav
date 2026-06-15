# RSC-Nav 实验协议 v3
*(First-Principles Experimental Protocol for RSC-Nav)*

---

## 0. 核心问题

RSC-Nav 要验证的不是“agent 会不会走路”，而是：

> 在同一环境被反复使用时，agent 能否构建、检索、复用并修正长期语义-空间记忆，从而减少重复探索并避免旧记忆误导？

实验必须围绕四个基本事实展开：

1. agent 只能通过第一视角观测获得环境信息。
2. 同一环境中的空间结构和语义对象具有可复用性。
3. 历史经验可能过期。
4. 过长历史不能直接全部塞进隐状态，必须被结构化、检索化。

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
-> conflict-aware memory update
```

系统维护四类记忆：

- occupancy / explored memory
- semantic evidence memory
- landmark-topological memory
- memory state: active / stale / missing / relocated

每个 semantic / landmark memory item 至少包含：

```text
id
semantic_label or embedding
bev_position
confidence
freshness
last_seen_time
visit_count
negative_evidence_count
status
source_view_ids
context_id or context_embedding
```

---

## 2. 仿真环境

主实验在室内仿真中完成。

基础环境：

- Habitat-Lab / Habitat-Sim
- HM3D / HM3D-OVON / ObjectNav-compatible HM3D
- RGB-D observation
- oracle pose / depth as default
- simulator semantic labels or precomputed semantic evidence

默认不把 SLAM、开放词汇检测器、真实机器人部署作为主变量。

主变量只有一个：

> 是否存在可检索、可复用、可修正的长期语义-空间记忆。

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

---

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
- Revisit Success
- node-to-BEV attention quality
- landmark relation failure rate

---

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

---

### Claim 4: Conflict-Aware Reconfiguration

当新观测与旧记忆冲突时，系统应降低旧记忆影响，并更新 memory state。

基础对照：

```text
carried-stale
vs
carried-reconfigured
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

---

## 5. 冲突实验分层

### Level 1: Semantic-Evidence Perturbation

最轻量、最可控。

```text
task_1:
  object_a observed at location_x

perturbation:
  object_a relocated / removed / confidence degraded in semantic evidence

task_2:
  search object_a or object_b near object_a
```

用于验证 memory update rule 的最小因果闭环。

---

### Level 2: Simulation Scene Variant

v3 新增主增强实验。

目标：在仿真观测层验证旧记忆修正，而不只停留在 metadata perturbation。该层级作为主增强实验，优先完成一种最小可控版本。

基本形式：

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

推荐变化：

- target object relocated
- target object removed
- landmark-related object relation changed
- distractor object inserted near old location

控制变量：

- same base scene
- same task sequence
- same start pose when feasible
- same max steps
- same success radius
- same action space
- same perception source
- same planner / policy

核心对照：

```text
carried-stale on scene_i_v2
vs
carried-reconfigured on scene_i_v2
vs
memory-reset on scene_i_v2
```

论文表述：

> controlled simulation scene-variant reconfiguration

不表述为真实动态世界部署。

---

## 6. Baselines

硬 baseline：

- BEV-only reset
- BEV-only carried
- RSC-full reset
- RSC-full carried
- RSC-full carried-stale
- RSC-full carried-reconfigured
- Reactive / frontier-only

关键消融：

- RSC-full carried without landmark retrieval
- RSC-full carried without freshness penalty
- RSC-full carried without status penalty
- RSC-full carried without negative evidence
- RSC-full carried with random retrieval
- RSC-full carried with recency-only retrieval

可选 baseline：

- Graph-only memory
- LSTM / implicit memory
- open-vocabulary perception variant

---

## 7. 指标

### 7.1 导航指标

- Success Rate
- SPL
- Distance-to-Goal
- Path Length
- Stop Accuracy

### 7.2 记忆复用指标

```text
Memory Reuse Gain = SPL_carried - SPL_reset
```

也报告：

- Success Gain
- Path Reduction
- Exploration Efficiency
- Revisit Success

### 7.3 检索指标

- Retrieval Hit@K
- Anchor Hit@K
- target-region Hit@K
- irrelevant retrieval rate
- retrieval score decomposition

retrieval score 至少拆成：

```text
semantic match
spatial proximity
confidence
freshness
status penalty
context match
```

### 7.4 重配置指标

- Adaptation Success
- Recovery Steps
- Stale Memory Error Rate
- Relocated Object Hit@K
- Map Correction Latency
- wrong old-location stop rate

---

## 8. 统计要求

所有主实验使用 paired comparison。

必须报告：

- number of scenes
- number of episode groups
- number of tasks
- number of target categories
- random seeds
- mean
- standard deviation or 95% confidence interval

`Memory Reuse Gain` 只在 `task_index >= 2` 上统计。

失败案例至少分为：

- timeout
- saw goal but failed stop
- wrong object stop
- returned to stale old location
- stale node ranked above corrected node
- map / projection failure
- landmark relation failure
- irrelevant-memory retrieval

---

## 9. 可视化

每组主实验至少输出：

- BEV occupancy / explored / semantic map
- landmark graph
- goal-to-node top-k retrieval
- retrieval score decomposition
- confidence / freshness / status curve
- reset vs carried trajectory
- stale vs reconfigured trajectory
- scene_v1 vs scene_v2 memory correction case

---

## 10. 实施阶段

### Phase 0: Protocol and Data Contract

- freeze episode schema
- freeze memory item schema
- freeze perturbation schema
- implement validation scripts
- generate small sample episodes

### Phase 1: Memory Core

- implement `MemoryItem`
- implement `SemanticEvidence`
- implement `LandmarkNode`
- implement `SemanticSpatialMemory`
- implement write / retrieve / weaken / relocate / overwrite
- run smoke test:

```text
write -> retrieve -> perturb -> stale retrieval -> reconfigured retrieval
```

### Phase 2: BEV Memory

- RGB-D to BEV projection
- occupancy map
- explored map
- semantic evidence map
- confidence / freshness / negative evidence update

### Phase 3: Landmark Retrieval

- keyframe node
- landmark node
- temporal / spatial edges
- node merge
- goal-to-node retrieval
- node-to-BEV attention

### Phase 4: Navigation Policy

主实验优先使用固定 planner / waypoint teacher，避免把结论混入策略学习能力。

可选扩展：

- imitation waypoint head
- learned stop head

### Phase 5: Main Experiments

- Claim 1: reset vs carried
- Claim 2: BEV-only vs RSC-full
- Claim 3: short history vs long history with distractors
- Claim 4: stale vs reconfigured

### Phase 6: Simulation Scene Variant

- construct one minimal scene_i_v1 / scene_i_v2 variant
- run one prioritized relocation / removal / relation-change test
- report stale-memory correction metrics
- add qualitative visual cases

### Phase 7: Thesis Packaging

- method equations
- protocol description
- main tables
- ablation tables
- visual cases
- failure analysis

---

## 11. 最小毕业版本

必须完成：

1. Habitat/HM3D 仿真下的 repeated-use episodes。
2. BEV + semantic + landmark memory。
3. reset / carried 对照。
4. BEV-only / RSC-full 对照。
5. stale / reconfigured 对照。
6. long-history distractor 分析。
7. 至少一种最小可控 scene-variant 仿真验证。
8. 指标、消融、可视化和失败案例分析。

---

## 12. 论文贡献表述

建议最终贡献写为：

1. 提出一种 RSC-inspired 的长期语义-空间记忆导航机制。
2. 设计目标条件的语义-地标检索与状态化记忆更新方法。
3. 构建 repeated-use proxy episodes 和 controlled simulation scene variants，用于评估长期记忆复用与重配置。
4. 通过消融和机制指标验证长期记忆、地标检索、长历史保持和冲突重配置的作用。

---

> **版本:** v3  
> **最后更新时间:** 2026-06-15
