# M3.5: Semantic Representation Alignment
*(Semantic BEV / Object Memory / Landmark Graph Representation Audit)*

## 阶段定位

M3.5 是从 RGB 输入链路进入导航规划前的表示审计阶段。

它的目标不是继续替换 detection backend，也不是提前做 Phase 5 navigation policy，而是回答一个更基础的问题：

```text
RSC-Nav 输入给后续 planner 的语义空间记忆，到底应该如何表示？
```

当前主线已经具备：

```text
Habitat RGB-D / RGB sequence
-> OWLv2 / GroundingDINO / SAM grounding evidence
-> depth / pose / geometry projection
-> object inventory
-> semantic BEV MVP
-> RSC object memory
-> Phase 3 landmark retrieval
```

M3.5 需要把这些产物整理成论文可解释、代码可复用、实验可消融的统一表示。

## 当前状态

状态：准备开始。

前置阶段结论：

```text
M2.5 open-vocabulary grounding adapter:
  MVP passed
  best default branch = GroundingDINO box + full coverage + multi-view/confidence filter
  SAM = optional localization refinement, not default evidence source

M3 RGB geometry frontend:
  VGGT MVP 已跑通
  但当前主线仍可继续使用 Habitat RGB-D / oracle pose 隔离语义记忆变量
```

因此 M3.5 可以暂时基于当前已产出的 Habitat oracle geometry + GroundingDINO semantic evidence 开展表示对齐，不必等待 RGB-only geometry 完全成熟。

## 核心问题

当前 semantic BEV 可视化容易被误解为“彩色地图就是语义地图”。M3.5 需要明确：

```text
PNG / GIF 只是展示层。
真正的 semantic map 应该是可计算的多层状态：
  geometry occupancy
  semantic evidence / confidence
  object instances
  landmark graph
  temporal update state
```

如果这一层不先定清楚，Phase 5 API planner 会出现输入摇摆：

```text
planner should read:
  a color image?
  a class-id grid?
  an object inventory?
  a landmark graph?
  all of them with priority?
```

M3.5 的任务就是把这个输入契约定下来。

## 建议表示：G / S / O / L

### G: Geometry BEV

用于传统导航和可通行性判断。

内容：

```text
unknown / free / occupied
explored mask
trajectory
optional frontier / visited count
```

特点：

```text
低语义、低噪声、高实时性。
即使 semantic evidence 错误，底层 navigation 仍主要依赖 G。
```

### S: Semantic Evidence BEV

用于表示每个 BEV cell 上的语义证据。

推荐内部表示：

```text
semantic_logits or semantic_confidence[C, H, W]
semantic_label[H, W] as visualization / argmax view
semantic_freshness[H, W]
semantic_source_count[H, W]
semantic_uncertainty[H, W]
```

注意：

```text
墙、地板、天花板等背景类应低权重或作为 geometry prior。
landmark 类对象应高权重，因为它们对任务规划更有用。
```

### O: Object Memory / Object Inventory

用于维护可复用对象实例。

核心字段：

```text
object_id
label / aliases
position_3d
bev_position
confidence
freshness
source_view_ids
positive_observation_count
missed_observation_count
negative_evidence_count
status: active / stale / missing
context_id
```

说明：

```text
O 是当前 RSC-Nav 的主记忆层。
M2.5 的 grounding_candidates 已经可以初始化 O。
Phase 2 negative evidence update 主要也作用在 O。
```

### L: Landmark / Topology Graph

用于任务规划和 goal query。

节点：

```text
landmark object nodes
optional room / area nodes
optional door / passage nodes
optional frontier nodes
```

边：

```text
near / connected / visible-from / same-room / reachable-via
```

说明：

```text
Phase 3 M1 已完成 O -> L 的最小版本。
M3.5 需要定义 L 如何作为 Phase 5 planner 的主要语义输入。
```

## 与常见论文表示的对齐方向

M3.5 需要审计并参考的表示范式：

```text
1. dense semantic occupancy grid
   每个 BEV cell 存占用和类别置信度。

2. object-centric semantic map
   以对象实例和位置作为主要语义单元。

3. topological semantic graph
   以 room / object / doorway / route node 组织地图。

4. open-vocabulary feature map
   每个区域存 CLIP / VLM feature，可按文本 query 检索。

5. temporal semantic memory
   地图状态随 observation history、freshness、negative evidence 更新。
```

当前 RSC-Nav 更接近：

```text
geometry BEV
+ object-centric semantic evidence
+ temporal object memory
+ landmark retrieval graph
```

这不是缺点。它符合本项目的主线：长期可复用、可更新、可规划的室内语义空间记忆。

## M3.5 MVP 验收标准

### 文档验收

```text
1. 完成 semantic representation audit 文档。
2. 明确 G / S / O / L 四层字段和用途。
3. 明确哪些字段进入 Phase 5 planner，哪些只用于可视化或诊断。
4. 明确与常见 paper 表示的相同点、差异点和合理性。
```

### 代码 / 产物验收

```text
1. 生成一个 representation bundle：
   geometry_bev
   semantic_evidence_bev
   object_memory
   landmark_graph
   planner_context.json

2. 生成一个 HTML 审计页：
   Traditional BEV
   Semantic Evidence BEV
   Object Inventory Projection
   Landmark Graph / Top-K Retrieval
   Confidence / Freshness / Evidence Count

3. 当前 M2.5 best branch 能导出该 bundle。
4. Phase 3 retrieval 可从 bundle 中读取 L 或 O。
```

### 决策验收

```text
明确 Phase 5 planner 的最小输入：
  goal query
  G: geometry BEV summary
  O: active/stale landmark candidates
  L: top-k landmark nodes and relations
  optional S: compact semantic evidence summary

不把完整彩色 PNG 当作唯一 planner 输入。
PNG 只作为人类检查和可视化材料。
```

## 当前不做

```text
1. 不训练导航模型。
2. 不立即改写 Phase 5 planner。
3. 不重新大规模调 GroundingDINO / SAM。
4. 不把 VGGT RGB-only geometry 设为默认输入。
5. 不声称 dense semantic segmentation 已经完成。
```

## 第一批执行任务

```text
M3.5A: Literature / Representation Audit
  汇总 semantic BEV / semantic map / object-centric map / topological graph 的常见表示。

M3.5B: RSC-Nav Representation Contract
  固定 G / S / O / L 字段和 JSON / NPZ 输出约定。

M3.5C: Bundle Exporter MVP
  从 M2.5 best run 导出 representation bundle。

M3.5D: Audit HTML
  可视化 G / S / O / L，并标注哪些是 planner input、哪些是 diagnostic view。

M3.5E: Phase 5 Handoff
  形成 planner_context.json，为 API planner / waypoint scoring 做输入准备。
```

## Log

### 2026-07-03: 阶段启动

启动原因：

```text
M2.5 RGB input chain 已达到 MVP pass：
  RGB / RGB-D -> open-vocabulary grounding -> object inventory -> semantic BEV -> RSC memory -> Phase 3 retrieval -> oracle validation

当前主要瓶颈不再是链路是否可跑，而是语义空间记忆表示是否足够规范、可解释、可对齐论文表示。
```

当前结论：

```text
1. 继续局部调 SAM 的边际收益有限。
2. 需要先固定表示层，再进入 Phase 5 planner。
3. G / S / O / L 四层表示暂定为 M3.5 的主工作对象。
```

下一步：

```text
先做 M3.5A：对照项目文档中的内部参考文献和近期 semantic map / BEV / open-vocabulary map 工作，
整理常见表示方式，并据此审计当前 RSC-Nav 的 semantic BEV / object memory / landmark graph 是否对齐。
```

### 2026-07-03: M3.5A 初步表示审计

参考方向：

```text
NLMap:
  open-vocabulary queryable scene representation
  object / location query results injected into LLM planner

VLMaps:
  visual-language features fused into spatial / 3D map
  natural-language indexing of landmarks and obstacle maps

OpenScene:
  dense 3D point features aligned with text / image feature space
  arbitrary text query -> 3D heatmap / segmentation

ConceptGraphs:
  open-vocabulary 3D scene graph
  object nodes + semantic / spatial relations for planning

Open-Fusion:
  real-time open-vocabulary RGB-D mapping and queryable scene representation
```

初步结论：

```text
1. 大部分相关工作不是只维护一张彩色 BEV PNG。
2. 更常见的是组合表示：
   geometry / occupancy
   semantic feature or evidence
   object instances
   graph / relations
   query interface
3. RSC-Nav 当前 G/S/O/L 四层拆分与这些方向基本对齐。
4. 当前不足是 S 仍偏 object-centric evidence，不是 dense per-cell open-vocabulary feature map。
5. 这不阻塞 Phase 5 MVP；Phase 5 更应先消费 O/L/G 的结构化上下文，而不是完整 PNG。
```

产物：

```text
outputs/m35_semantic_representation_alignment/semantic_representation_audit_20260703.html
```

M3.5B 下一步：

```text
定义 representation_bundle schema：
  geometry_bev
  semantic_evidence
  object_memory
  landmark_graph
  planner_context

并从当前 M2.5 best run 导出一份 bundle。
```

### 2026-07-03: Paper-style semantic BEV figure

用户目标：

```text
继续把当前 RGB 输入链路优化到论文级对齐的 semantic BEV 表示。
希望看到更好看的、多圈遍历后的近似论文级图。
```

本次新增：

```text
scripts/m35_make_paper_semantic_bev_figure.py
```

图像结构：

```text
A. First-person RGB
B. Open-vocabulary grounding evidence
C. G: traditional geometry BEV
D. O: object inventory projection
E. S: semantic BEV evidence
F. L: landmark retrieval graph

top strip:
  candidates / landmarks / precision / recall / F1 / centroid error

bottom strip:
  representation contract
```

当前输出：

```text
Best quality baseline:
  outputs/m35_semantic_representation_alignment/paper_semantic_bev_best96_20260703/paper_semantic_bev_overview.html
  P/R/F1=0.429/0.474/0.450, err=0.277 m

Multi-pass traversal:
  outputs/m35_semantic_representation_alignment/paper_semantic_bev_2pass96_20260703/paper_semantic_bev_overview.html
  P/R/F1=0.292/0.400/0.337, err=0.267 m

Index:
  outputs/m35_semantic_representation_alignment/paper_semantic_bev_index_20260703.html
```

说明：

```text
best96 版指标更好，适合作为当前主结果候选图。
2-pass96 版满足多圈遍历展示需求，但当前检测分布导致 F1 低于 best96。
这说明“多走几圈”会增加覆盖和记忆展示，但不自动提升 open-vocabulary detector 的语义库存质量；
后续如果要把多圈版也做成主结果，需要调 route / representative frame / filtering threshold。
```

### 2026-07-03: M3.5B/C representation bundle MVP

目标：

```text
把当前 best96 RGB / grounding / BEV / object memory / retrieval 输出，
固化为 Phase5A API planner 可以读取的 G/S/O/L representation bundle。
```

本次新增：

```text
scripts/m35_export_representation_bundle.py
```

导出内容：

```text
geometry_bev_summary.json
semantic_evidence_summary.json
object_memory.json
landmark_graph.json
planner_context.json
bundle_manifest.json
representation_bundle_report.html
assets/
```

当前主产物：

```text
outputs/m35_semantic_representation_alignment/representation_bundle_best96_20260703/representation_bundle_report.html
outputs/m35_semantic_representation_alignment/representation_bundle_best96_20260703/planner_context.json
outputs/m35_semantic_representation_alignment/representation_bundle_best96_20260703/landmark_graph.json
```

运行输入：

```text
outputs/m25_open_vocab_grounding/full_env_groundingdino_t020_full96_views2_conf035_validation_20260703
```

运行摘要：

```text
status: passed
objects: 42
landmark_nodes: 41
lightweight_edges: 227
candidate_waypoints: 48
topk_for_goal(find bed): 5
GroundingDINO P/R/F1: 0.429 / 0.474 / 0.450
mean centroid error: 0.277 m
Phase3 retrieval: passed, 5/5 checks
```

M3.5B/C 设计决策：

```text
1. PNG / HTML 只作为 human audit view。
2. Phase5A planner 的主输入是 planner_context.json。
3. G/S/O/L 分别对应：
   G: geometry_bev_summary
   S: semantic_evidence_summary
   O: object_memory_summary
   L: element_topology_graph + topk_landmarks
4. wall / floor / ceiling 被标为 tier0_background，不作为 goal landmark。
5. door / corridor / connector 类被标为 tier1_connector，优先用于 stopover / transition。
6. bed / sofa / chair / table 等被标为 tier2_stable_landmark，作为主要 goal anchor。
7. cup / mug / book 等被标为 tier3_task_object，按 query 和置信度使用。
8. near edges 是轻量空间关系，不是完整拓扑规划图。
9. candidate_waypoints 是 landmark perimeter anchor，必须由 traditional BEV / navmesh 再验证可达。
```

Phase4-lite 占位：

```text
planner_context.json 已包含：
  current_context_id
  context_confidence
  remap_triggered
  mismatch_signals

当前 remap_triggered=false。
完整 mismatch score / context manager / A->B->A 回访验证仍属于后续 Phase4。
```

结论：

```text
M3.5B/C MVP 通过。
当前已经有稳定的 planner input contract，可以进入 Phase5A API semantic task planner 最小闭环。

进入 Phase5A 前不建议继续大改 perception 或完整 remapping；
只需把 planner_context.json 作为输入，先验证：
  RSC memory + landmark/topology context
  -> API task plan
  -> stopover waypoint ranking
  -> traditional planner execution
  -> trace / report
```
