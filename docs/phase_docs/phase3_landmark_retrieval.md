# Phase 3: Landmark Retrieval
*(M1: Object Memory -> Landmark Nodes -> Goal Query Top-K Retrieval)*

## 阶段定位

Phase 3 的目标是在 Phase 2 的 semantic-spatial memory substrate 上构建最小可用的 landmark retrieval，使长期记忆可以按目标查询，而不只是被动显示在 BEV / semantic BEV 上。

M1 最小闭环：

```text
object memory
-> landmark nodes
-> goal query
-> top-k retrieval
-> retrieval score breakdown
-> BEV region projection / visualization
```

Phase 3 是后续 VGGT-NLMap Semantic BEV Bridge 的前置入口：外部导入的 NLMap objects 必须能落到 RSC landmark nodes，并能被 goal query 检索。

## 当前状态

状态：M1 初步实现已完成，已通过 fixture 验收、Phase 2.13 真实 object memory 导入验收和独立子 agent 审核。

已完成：

- `src/landmark_retrieval.py`：landmark node 构建、alias-aware merge、context / status-aware top-k retrieval。
- `scripts/phase3_landmark_retrieval_eval.py`：fixture 验收、外部 object memory 导入、score breakdown、HTML/SVG 可视化报告。
- fixture 验收：9/9 checks passed，覆盖 exact query、alias query、alias merge、context selection、missing/stale 降权、unknown query abstain。
- Phase 2.13 object memory smoke：36 个 object -> 31 个 landmark nodes，5/5 auto top-1 label checks passed，并生成 top-k retrieval 和 BEV 可视化。

仍待补：

- 更严格的 schema validator。
- 与 Phase 2 runner 的正式集成入口。
- 更强的真实场景人工标注验收，例如 Hit@K、invalid retrieval rate、context leakage。
- relation query / keyframe graph / topology edges。
- Phase 5 navigation policy 接入。

## M1 范围

M1 必须完成：

- 从现有 object memory 生成 landmark nodes。
- 支持从 `objects.json` / object memory 导入 landmark nodes。
- 支持简单 goal query。
- 输出 top-k retrieval。
- 输出 retrieval score decomposition。
- 将 top-k landmarks 投影回 BEV 区域。
- 生成机器可读 report 和 HTML 可视化。

M1 不做：

- 完整 navigation policy。
- waypoint / stop decision。
- 复杂拓扑规划。
- learned retrieval model。
- full context remapping。
- 真实机器人控制。

## 输入

来自 RSC-Nav Phase 2：

- semantic BEV。
- object memory items。
- trajectory / keyframes。
- memory state: confidence / freshness / status。
- context_id。

来自 Bridge / 外部前端的后续输入：

- `objects.json`。
- `rsc_memory_init.json`。
- object label / semantic score / 3D or BEV position。
- source frame ids。

输入 object memory item 最小字段：

```text
id
label
bev_position: [x, y]
confidence
freshness
status: active / stale / missing / relocated
context_id
source: live / prior / imported
source_view_ids
```

## 输出

M1 输出文件：

```text
landmark_nodes.json
topk_retrieval.json
retrieval_score_breakdown.json
retrieval_report.html
retrieval_bev.png
```

`landmark_nodes.json` 最小字段：

```text
id
label
aliases
bev_position
confidence
freshness
status
context_id
source_object_ids
visit_count
last_seen_step
```

`topk_retrieval.json` 最小字段：

```text
query
context_id
top_k
results:
  - landmark_id
    label
    bev_position
    final_score
    score_breakdown
    status
    confidence
    freshness
```

## Landmark Node 构建规则

M1 使用规则式构建，不引入学习模型。

基本规则：

```text
one active object memory item
-> one landmark node
```

合并规则：

```text
same / alias label
+ same context_id
+ BEV distance < merge_radius
-> merge into one landmark node
```

默认合并字段：

- position: confidence-weighted average。
- confidence: max or weighted mean。
- freshness: max。
- status: 取更可靠状态，优先级为 active > stale > missing。
- source_object_ids: union。

M1 可先使用保守参数：

```text
merge_radius = 0.5m to 1.0m
top_k = 5
```

## Goal Query 支持范围

M1 支持三类简单 query。

### 1. Object Query

示例：

```text
chair
find chair
where is the bed
```

目标：按 label / alias 检索 object landmark。

### 2. Category Query

示例：

```text
seat
furniture
door
```

目标：用 alias table 或轻量文本匹配，将 query 映射到候选 labels。

M1 可先用手写 alias table：

```text
seat -> chair / sofa
desk -> table
mug -> cup
entrance -> door
```

### 3. Imported Object Query

示例：

```text
NLMap imported cup
imported chair
```

目标：验证外部 object memory 能被同一 retrieval pipeline 使用。

M1 暂不要求复杂关系 query：

```text
lamp near sofa
cup on table
chair beside door
```

relation goal 放到 Phase 3 后续增强或 Phase 5 navigation 前。

## Retrieval Score v1

M1 使用可解释规则打分：

```text
final_score =
  w_semantic * semantic_match
+ w_confidence * confidence
+ w_freshness * freshness
+ w_status * status_score
+ w_context * context_match
+ w_source * source_bonus
```

初始权重建议：

```text
w_semantic = 0.40
w_confidence = 0.20
w_freshness = 0.15
w_status = 0.15
w_context = 0.10
w_source = 0.00
```

M1 暂不强制使用 spatial proximity，因为单个 object query 没有明确目标位置；后续 relation query 或 navigation query 再加入 spatial proximity。

状态分数建议：

```text
active: 1.0
stale: 0.5
relocated: 0.4
missing: 0.0
```

context_match：

```text
same context_id: 1.0
unknown context: 0.5
different context_id: 0.0
```

semantic_match v1：

```text
exact label match: 1.0
alias match: 0.8
substring / normalized match: 0.6
no match: 0.0
```

后续可替换为 CLIP / sentence embedding，但 M1 先保持规则可解释。

## BEV Projection

top-k retrieval 结果必须能投影到 BEV。

M1 输出：

- landmark center。
- top-k rank marker。
- label。
- status color。
- confidence size or alpha。

推荐颜色：

```text
active: green
stale: yellow
missing: gray
relocated: orange
query target: red outline
```

## 验收标准

M1 通过需要满足以下条件。

### A. 数据转换验收

- 能从现有 object memory 生成 `landmark_nodes.json`。
- 每个 active object 至少生成一个 landmark node，除非被合理 merge。
- `landmark_nodes.json` 包含 label、position、confidence、freshness、status、context_id。
- 支持从外部 `objects.json` 或 `rsc_memory_init.json` 导入并生成 landmark nodes。

### B. 检索功能验收

- 输入至少 5 个常见 object queries，可以返回 top-k。
- exact label query 的 top-1 应命中对应 label。
- alias query 的 top-k 应包含对应目标 label。
- missing / stale object 不应排在同类 active object 前面，除非 active object 置信度显著更低。
- different context_id 的 landmark 默认不应进入当前 context top-k 前列。

### C. 可解释性验收

- 每个 retrieval result 必须输出 score breakdown。
- score breakdown 至少包含：

```text
semantic_match
confidence
freshness
status_score
context_match
final_score
```

- HTML report 能展示 query、top-k、分数拆解和 BEV 标注。

### D. Bridge 接入验收

- 能读取一个模拟 NLMap object list。
- 能把 imported objects 写成 RSC landmark nodes。
- imported landmark 能被 goal query 检索。
- imported landmark 能投影到 BEV。

### E. 回归安全验收

- Phase 2 object memory / semantic BEV 原有输出不被破坏。
- 不改变 Phase 2 negative evidence update 语义。
- 不要求 navigation policy 才能运行 Phase 3 retrieval。

## 最小测试集

建议构造一个 deterministic fixture：

```text
objects:
  chair active confidence=0.9
  chair stale confidence=0.8
  table active confidence=0.7
  bed missing confidence=0.4
  door active confidence=0.6
  sofa active confidence=0.85 context_id=context_A
  sofa active confidence=0.95 context_id=context_B
```

测试 queries：

```text
chair
seat
table
door
bed
sofa in context_A
sofa in context_B
```

预期：

- `chair` top-1 为 active chair。
- `seat` top-k 包含 chair / sofa。
- `bed` 可返回 missing bed，但 status_score 为 0，排名应低。
- context_A query 不应优先返回 context_B sofa。

## 关键评估指标

M1 先报告 retrieval-level 指标：

- Retrieval Hit@1。
- Retrieval Hit@K。
- Alias Hit@K。
- Stale-over-active error rate。
- Context leakage rate。
- invalid / irrelevant retrieval rate。

M1 不报告 SPL / Success Rate，因为还没有接 navigation policy。

## 产物清单

代码产物建议：

```text
src/landmark_memory.py
src/landmark_retrieval.py
scripts/phase3_landmark_retrieval_eval.py
scripts/phase3_landmark_retrieval_report.py
```

输出产物建议：

```text
outputs/phase3_landmark_retrieval/
  landmark_nodes.json
  topk_retrieval.json
  retrieval_score_breakdown.json
  retrieval_bev.png
  retrieval_report.html
  metrics.json
```

当前验收产物：

```text
outputs/phase3_landmark_retrieval/m1_fixture/retrieval_report.html
outputs/phase3_landmark_retrieval/from_phase213_bed_door/retrieval_report.html
```

## 后续增强

M1 之后再做：

- keyframe node。
- temporal / spatial / co-visible edges。
- relation query: `object near landmark`。
- node-to-BEV attention。
- embedding-based semantic match。
- graph retrieval。
- integration with Phase 5 waypoint / stop decision。

## 与 VGGT-NLMap Semantic BEV Bridge 的关系

Bridge Stage A/B 依赖 RSC-Nav Phase 3 M1：

```text
NLMap objects
-> RSC object memory import
-> landmark nodes
-> goal query top-k retrieval
```

因此 M1 的验收目标不是完整导航，而是提供一个稳定的外部 semantic map 接入口。随后 Bridge Stage A/B 负责把 NLMap-style semantic map 转为 occupancy BEV / semantic BEV / RSC object memory，并复用 M1 retrieval。Bridge Stage C 再用 VGGT / DUST3R 替换 RGB-D / pose 几何来源。

统一执行顺序：

```text
1. RSC-Nav Phase 3 M1
2. Bridge Stage A/B
3. Bridge Stage C
4. RSC-Nav 后续阶段
```

> **最后更新时间:** 2026-06-29
