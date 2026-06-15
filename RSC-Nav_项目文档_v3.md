# RSC-Nav 项目文档 v3
*(RSC-Nav: Retrosplenial-Inspired Long-Term Semantic-Spatial Memory for Embodied Navigation)*

---

## 1. 项目问题

RSC-Nav 研究的问题是：

> 在同一室内环境被反复使用时，具身 agent 能否构建、检索、复用并修正长期语义-空间记忆，从而减少重复探索、保持长历史中的关键空间线索，并避免旧记忆误导？

这不是一个单次 ObjectNav 问题，也不是单纯的反应式导航问题。

RSC-Nav 关注的是长期使用中的四个事实：

1. agent 的信息来自第一视角 RGB-D、pose 和语义证据。
2. 同一环境中的空间结构、对象和地标可以被复用。
3. 历史记忆可能过期，需要被修正。
4. 长历史不能全部依赖隐状态保存，必须被结构化并按目标检索。

---

## 2. 研究目标

本文目标是提出并验证一种 RSC-inspired 长期语义-空间记忆导航机制。

该机制回答四个问题：

1. **记忆如何构建？**  
   如何将第一视角观测转换为环境中心的 BEV、语义对象和地标拓扑记忆？

2. **记忆如何复用？**  
   后续任务到来时，agent 是否能利用历史探索经验减少重复探索？

3. **记忆如何检索？**  
   当前目标如何从长期记忆中检索相关对象、地标、区域和关系？

4. **记忆如何修正？**  
   当新观测与旧记忆冲突时，系统如何降低旧记忆影响并更新状态？

---

## 3. RSC 启发

本文不复现压后皮层的生物机制，而是将 RSC 相关空间认知功能抽象为可实现的计算机制。

### H1: 参考系转换

```text
egocentric observation + pose
-> allocentric BEV memory
```

第一视角观测被投影到环境中心或局部全局一致的空间记忆中。

### H2: 地标锚定

```text
semantic evidence + keyframes
-> landmark / topological memory
```

稳定对象、关键视角和可复访位置被保存为可检索地标。

### H3: 目标条件检索

```text
goal query
-> relevant semantic / landmark memory
-> BEV action context
-> waypoint / stop
```

当前目标先读取长期记忆，再决定探索区域、waypoint 或 stop。

### H4: 记忆重配置

```text
observation conflict + old memory
-> confidence / freshness / negative evidence / status update
-> corrected retrieval and navigation decision
```

系统不盲目信任旧地图，而是在局部冲突出现时修正相关记忆项。

---

## 4. 核心机制

RSC-Nav 维护四类长期记忆：

1. **Occupancy / explored memory**  
   记录可通行区域、障碍物、未知区域和已探索区域。

2. **Semantic evidence memory**  
   记录对象、类别、位置、置信度、新鲜度、访问次数和负证据。

3. **Landmark-topological memory**  
   记录地标节点、关键视角节点，以及它们之间的空间或时序连接。

4. **Memory state**  
   记录每个记忆单元的状态：

```text
active / stale / missing / relocated
```

整体流程：

```text
RGB-D + pose + semantic evidence
        |
        v
Allocentric BEV projection
        |
        +--> occupancy / explored memory
        +--> semantic evidence memory
        +--> landmark-topological memory
        +--> memory state
                  |
goal query --------+
        |
        v
goal-conditioned retrieval
        |
        v
waypoint / stop decision
        |
        v
new observation and conflict-aware update
```

---

## 5. 记忆项定义

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

MVP 阶段使用规则式更新。

```text
confirm:
  新观测再次支持旧记忆，提高 confidence，刷新 freshness / last_seen_time

weaken:
  历史节点在当前可见区域内应被看到却未出现，降低 confidence，增加 negative evidence

relocate:
  同类或同实例在新位置连续确认，旧节点标记 stale / relocated，新节点升权

overwrite:
  新证据稳定强于旧证据时，更新 semantic memory 或 landmark node
```

---

## 6. 实验主张

### Claim 1: Memory Reuse

长期记忆应提升后续任务效率。

```text
memory reset
vs
memory carried
```

### Claim 2: Landmark-Conditioned Retrieval

地标和拓扑记忆应比单一 BEV 记忆提供更稳定、更可解释的目标检索。

```text
BEV-only carried
vs
RSC-full carried
vs
RSC-full carried without landmark retrieval
```

### Claim 3: Long-Horizon Memory Retention

当历史任务变长并混入无关任务时，系统仍应检索到目标相关记忆。

```text
short history
vs
long history with distractor tasks

goal-conditioned retrieval
vs
random / recency-only retrieval
```

该 claim 优先作为 retrieval-level 实验：先评估长历史和 distractor 条件下的目标相关记忆检索能力，再在代表性设置中补充少量导航结果。

### Claim 4: Conflict-Aware Reconfiguration

当新观测与旧记忆冲突时，重配置机制应减少 stale memory 误导。

```text
carried-stale
vs
carried-reconfigured
```

---

## 7. 仿真验证

主实验在室内仿真环境中进行：

- Habitat-Lab / Habitat-Sim
- HM3D / HM3D-OVON / ObjectNav-compatible HM3D
- RGB-D observation
- oracle pose / depth
- simulator semantic labels or precomputed semantic evidence

默认隔离 SLAM 和开放词汇检测噪声，把论文主变量集中在长期记忆机制上。

### 7.1 Repeated-Use Proxy Episodes

```text
scene_i:
  task_1: explore / find object_a / write memory
  task_2: find object_b / reuse memory
  task_3: find object_c near object_a / retrieve landmark relation
  task_4: conflict or scene-variant test / update memory
```

`task_1` 用于写入记忆。  
长期记忆收益从 `task_index >= 2` 开始统计。

### 7.2 Semantic-Evidence Perturbation

```text
task_1:
  object_a observed at location_x

perturbation:
  object_a relocated / removed / confidence degraded in semantic evidence

task_2:
  search object_a or object_b near object_a
```

该实验用于验证更新规则的最小因果闭环。

### 7.3 Controlled Simulation Scene Variants

v3 加入受控仿真场景变体验证。

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
- landmark-related relation changed
- distractor object inserted near old location

执行分层：

- Level 1：semantic-evidence perturbation，作为 B 档必做的最小因果闭环。
- Level 2：controlled simulation scene variant，作为主增强实验，优先完成一种最小可控版本。

该实验验证：RSC-Nav 不只在 metadata perturbation 下工作，也能在受控仿真观测变化下修正旧语义-空间记忆。

---

## 8. Baselines and Ablations

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

## 9. 主要指标

常规导航指标：

- Success Rate
- SPL
- Distance-to-Goal
- Path Length
- Stop Accuracy

长期记忆指标：

- Memory Reuse Gain
- Revisit Success
- Exploration Efficiency
- Long-Horizon Memory Retention

检索指标：

- Retrieval Hit@K
- Anchor Hit@K
- target-region Hit@K
- irrelevant retrieval rate
- retrieval score decomposition

重配置指标：

- Adaptation Success
- Stale Memory Error Rate
- Recovery Steps
- Relocated Object Hit@K
- Map Correction Latency
- wrong old-location stop rate

---

## 10. 研究边界

B 档硕士主实验验证长期记忆机制，不承诺解决所有真实部署问题。

本文不做：

- 真实机器人多天部署。
- 完整真实动态世界建图。
- 完整 GOAT-Bench 排榜。
- 完整 3D scene graph 系统。
- 端到端大模型智能体。

本文优先做：

- Habitat/HM3D 室内仿真。
- oracle pose / depth。
- simulator semantic labels 或预提取 semantic evidence。
- repeated-use proxy episodes。
- controlled simulation scene variants。
- 规则式长期记忆更新。

---

## 11. 实施路线

### Phase 0: Protocol and Data Contract

- freeze episode schema
- freeze memory item schema
- freeze perturbation schema
- implement validation scripts
- generate sample episodes

### Phase 1: Memory Core

- implement `MemoryItem`
- implement `SemanticEvidence`
- implement `LandmarkNode`
- implement `SemanticSpatialMemory`
- run:

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
- temporal / spatial adjacency
- goal-to-node retrieval
- node-to-BEV attention

### Phase 4: Navigation Policy

主实验优先使用固定 planner / waypoint teacher，避免把结论混入策略学习能力。

可选扩展：

- imitation waypoint head
- learned stop head

### Phase 5: Main Experiments

- reset / carried
- BEV-only / RSC-full
- short history / long history with distractors
- carried-stale / carried-reconfigured

### Phase 6: Simulation Scene Variant

- construct scene_i_v1 / scene_i_v2
- run relocation / removal / relation-change tests
- report stale-memory correction metrics
- add qualitative visual cases

### Phase 7: Thesis Packaging

- method equations
- protocol description
- main tables
- ablation tables
- visualization
- failure analysis

---

## 12. 最小毕业版本

硕士论文最小完成版本：

1. 一个可复现的 RSC-Nav 原型系统。
2. Habitat/HM3D 仿真下的 repeated-use episodes。
3. BEV + semantic + landmark memory。
4. reset / carried 对照实验。
5. BEV-only / RSC-full 对照实验。
6. long-history distractor 分析。
7. stale / reconfigured 对照实验。
8. 至少一种 controlled simulation scene variant。
9. 指标、消融、可视化和失败案例分析。

---

## 13. 论文贡献

建议论文贡献表述为：

1. 提出一种 RSC-inspired 的长期语义-空间记忆导航机制。
2. 设计目标条件的语义-地标检索与状态化记忆更新方法。
3. 构建 repeated-use proxy episodes 和 controlled simulation scene variants，用于评估长期记忆复用与重配置。
4. 通过消融和机制指标验证长期记忆、地标检索、长历史保持和冲突重配置的作用。

---

> **版本:** v3  
> **最后更新时间:** 2026-06-15
