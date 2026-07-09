# RSC-Nav 项目文档 v4
*(RSC-Nav: Retrosplenial-Inspired Long-Term Semantic-Spatial Memory for Embodied Navigation)*

---

## 1. 项目问题

RSC-Nav 研究的问题是：

> 在同一室内环境被反复使用时，具身 agent 能否构建、检索、复用并修正长期语义-空间记忆，从而减少重复探索、保持长历史中的关键空间线索，并避免旧记忆误导？

这不是一个单次 ObjectNav 问题，也不是单纯的反应式导航问题。RSC-Nav 关注的是长期使用中的四个事实：

1. agent 的信息来自第一视角 RGB-D、pose 和语义证据。
2. 同一环境中的空间结构、对象和地标可以被复用。
3. 历史记忆可能过期，需要被修正或重映射。
4. 长历史不能全部依赖隐状态保存，必须被结构化并按目标检索。

v4 在 v3 主线基础上补充一个关键边界：局部变化和整体环境变化不应使用同一种更新解释。

```text
same-context local change
-> adaptive update

global context change
-> context remapping
```

也就是说，物体被移走、遮挡、重新出现或局部语义证据冲突，应由 adaptive update 处理；当当前观测与旧环境整体不匹配时，应触发 context remapping，而不是把旧地图强行慢慢衰减成新地图。

---

## 2. 研究目标

本文目标是提出并验证一种 RSC-inspired 长期语义-空间记忆导航机制。

该机制回答五个问题：

1. **记忆如何构建？**  
   如何将第一视角观测转换为环境中心的 BEV、语义对象和地标拓扑记忆？

2. **记忆如何复用？**  
   后续任务到来时，agent 是否能利用历史探索经验减少重复探索？

3. **记忆如何检索？**  
   当前目标如何从长期记忆中检索相关对象、地标、区域和关系？

4. **记忆如何局部修正？**  
   当同一环境中的新观测与旧记忆冲突时，系统如何利用 confidence、freshness、negative evidence 和 status 更新相关记忆项？

5. **记忆如何全局重映射？**  
   当当前观测与旧环境整体不一致时，系统如何识别 context mismatch，避免旧 context 干扰当前任务？

---

## 3. RSC / Hippocampal 启发

本文不复现 RSC 或 hippocampus 的生物机制，而是将相关空间认知功能抽象为可实现的计算机制。

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

### H4: 局部记忆重配置

```text
observation conflict + old memory within the same context
-> confidence / freshness / negative evidence / status update
-> corrected retrieval and navigation decision
```

系统不盲目信任旧地图，而是在局部冲突出现时修正相关记忆项。

### H5: 全局 Context Remapping

```text
large geometric-semantic mismatch
-> create or switch memory context
-> reduce stale cross-context interference
```

这一部分仅作为 hippocampal remapping 和 RSC context-dependent spatial representation 的计算启发，不声称复现生物机制。

---

## 4. 核心机制

RSC-Nav 维护五类长期结构：

1. **Occupancy / explored memory**  
   记录可通行区域、障碍物、未知区域和已探索区域。

2. **Semantic evidence memory**  
   记录对象、类别、位置、置信度、新鲜度、访问次数、负证据，以及 prior/live evidence。

3. **Landmark-topological memory**  
   记录地标节点、关键视角节点，以及它们之间的空间、时序和共视关系。

4. **Memory state**  
   记录每个记忆单元的状态：

```text
active / stale / missing / relocated
```

5. **Context state**  
   记录记忆属于哪个环境上下文，并支持在整体不匹配时隔离旧 context：

```text
context_id / context_confidence / context_mismatch
```

context mismatch 后，系统应创建或切换到新的 `context_id`，但不覆写旧环境的 memory。旧 context 作为可再检索的历史记忆保留，只是在当前任务中被降权或隔离，避免跨环境误导。

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
        +--> memory and context state
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
new observation
        |
        +--> same-context adaptive update
        +--> global context remapping when mismatch is large
```

---

## 5. 记忆更新原则

长期记忆更新应区分三类证据：

```text
positive observation:
  当前确实看到了该对象或地标

not observable:
  当前没看到，但视角、高度、遮挡或覆盖条件不足以判断

expected-visible miss:
  历史位置理论上应可见，却连续没有看到
```

只有 expected-visible miss 才应产生 negative evidence。这样可以避免因为 pitch 改变、遮挡或视锥没有覆盖对象高度，就把“没看到”误判为“不存在”。

对于同一 context 内的局部冲突，系统进行 adaptive update：

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

对于整体环境变化，系统不应把 adaptive update 当作最终解决方案，而应进行 context remapping。A->B 这类跨环境加载实验可作为 stress test，用来观察 prior/live 分离和旧 context 影响是否会下降，但不作为环境大变的最终更新方法。

context remapping 的基本语义是：

```text
old context:
  keep memory
  archive / downweight for current task
  remain retrievable if context is revisited

new context:
  create new context_id
  write new observations into new memory
  retrieve primarily within current context
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

### Claim 4: Same-Context Adaptive Reconfiguration

当同一环境中的新观测与旧记忆局部冲突时，adaptive update 应减少 stale memory 误导。

```text
carried-stale
vs
carried-adaptive
vs
memory-reset
```

关键实验应使用同环境对象删除、移动、遮挡或语义证据扰动，而不是用完全不同的 A/B 环境替代。

### Claim 5: Context Remapping Under Global Change

当环境整体变化过大时，系统应识别 context mismatch，避免旧环境记忆继续作为当前任务主上下文。

```text
forced-single-context
vs
context-remapping
```

A->B 可以作为该 claim 的早期 stress test 或 failure-mode visualization，但真正的主张应是 remapping 后的 context selection 和 cross-context interference 降低。

该 claim 不要求删除旧环境记忆；相反，旧环境 memory 应被保留为 `context_A`，新环境写入 `context_B`。评估重点是当前任务能否选对 context，而不是把 A 的记忆擦掉。

---

## 7. 仿真验证

主实验在室内仿真环境中进行：

- Habitat-Lab / Habitat-Sim
- HM3D / HM3D-OVON / ObjectNav-compatible HM3D
- RGB-D observation
- oracle pose / depth
- simulator semantic labels or precomputed semantic evidence

默认隔离 SLAM 和开放词汇检测噪声，把论文主变量集中在长期记忆机制上。CLIP 或其他开放词汇模型可以作为扩展感知设置；即使不引入真实开放词汇识别，也不影响本文验证长期语义-空间记忆机制本身。

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

### 7.2 Same-Context Perturbation

```text
task_1:
  object_a observed at location_x

perturbation:
  object_a relocated / removed / confidence degraded in semantic evidence

task_2:
  search object_a or object_b near object_a
```

这是验证 adaptive update 的核心实验形式。

### 7.3 Global Context Stress Test

```text
scene_A:
  write memory context_A

scene_B:
  load context_A as prior
  observe mismatch
  evaluate forced-single-context vs remapping
```

该实验验证整体环境变化下是否需要 context remapping。仅使用 A->B 衰减曲线时，它只能作为 stress test，不构成最终方法主张。

---

## 8. 预期贡献

1. 提出一种 RSC-inspired 的长期语义-空间记忆导航框架。
2. 将第一视角 RGB-D / pose / semantic evidence 组织为可复用的 BEV、语义对象和地标拓扑记忆。
3. 设计目标条件的地标检索机制，用于长期记忆复用和导航上下文注入。
4. 区分 same-context adaptive update 与 global context remapping，避免把局部更新和整体环境切换混为一种机制。
5. 通过 repeated-use episodes、same-context perturbation、long-horizon distractors 和 context stress test 验证长期记忆的收益、稳定性和失效边界。

---

> **版本:** v4  
> **最后更新时间:** 2026-06-27
