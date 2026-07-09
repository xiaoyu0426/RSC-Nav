# Demo Concept: "I Want Water"

## 定位

本 demo 用来展示 RSC-Nav 的最终系统目标：

```text
first-person RGB observation
-> reusable semantic-spatial memory
-> evidence-aware update
-> context-aware remapping
-> semantic task planning
-> waypoint execution / replanning
```

它不是单纯展示 `find object`，而是展示机器人如何在室内环境中建立、复用、更新和规划语义空间记忆。

一句话目标：

> 机器人先用第一视角 RGB 建立室内语义记忆；之后用户说“我要喝水”，系统复用历史语义记忆规划到可能有水的地方；如果旧位置没有水，系统基于负证据降低置信度并重新规划；如果进入新环境，则触发 context remapping，保留旧环境记忆并创建或切换新 context。

## Demo 剧情

### 1. 首次进入新环境

机器人进入一个未知室内环境。初始没有可复用语义地图。

系统输入：

```text
first-person RGB / RGB-D frames
pose / depth / point cloud
```

系统在线或离线构建：

```text
traditional BEV / occupancy
semantic evidence
semantic BEV
RSC object memory
RSC landmark memory
candidate waypoints
```

理想探索策略：

```text
exploration interest =
  unexplored frontier
  + semantic uncertainty
  + landmark novelty
  + connector / door priority
  + memory coverage gain
```

当前 MVP 可先使用 scripted coverage loop，后续再替换为 interest-driven exploration。

### 2. 形成可复用语义记忆

探索完成后，系统生成当前环境的 memory context：

```text
context_id = home_A
traditional BEV
semantic BEV
object memory
landmark graph
top-k retrieval index
candidate waypoints
```

典型 object / landmark：

```text
table
chair
door
sofa
bed
bottle
cup
sink
fridge
kitchen counter
```

memory item 应包含：

```text
label
position
confidence
freshness
status: active / stale / missing
context_id
last_seen_step
negative_evidence_count
source_view_ids
```

### 3. 用户提出自然需求

用户不是直接说 `find bottle`，而是说：

```text
我要喝水
```

系统需要先做语义目标扩展：

```text
drink water
-> water bottle
-> cup
-> sink / faucet
-> fridge
-> kitchen counter
-> dining table
```

这一步可以由 API planner 或一个轻量 affordance expansion 模块完成。

### 4. 记忆检索与任务规划

系统从 object / landmark memory 中检索候选：

```text
bottle on table:
  confidence = 0.82
  freshness = 0.91
  status = active

cup near sink:
  confidence = 0.63
  freshness = 0.76
  status = active

sink in kitchen:
  confidence = 0.88
  status = active

fridge in kitchen:
  confidence = 0.95
  status = active
```

API semantic planner 读取：

```text
goal query
traditional BEV summary
semantic BEV summary
top-k landmarks
object memory states
element topology graph
candidate waypoints
candidate path summaries
```

示例 planner 输出：

```json
{
  "task_plan": [
    {"step": 1, "intent": "go_to_region", "target": "table_area"},
    {"step": 2, "intent": "verify_object", "target": "water_bottle"},
    {"step": 3, "intent": "stop_if_found", "target": "water_bottle"}
  ],
  "stopover_waypoints": ["wp_bottle_table_01"],
  "selected_waypoint_id": "wp_bottle_table_01",
  "stop_probability": 0.83,
  "reason": "A remembered bottle near the table is active, fresh, and close to the current route."
}
```

底层 traditional BEV / navmesh planner 负责可达路径和低层动作。

### 5. 到达旧位置但没有水瓶

机器人到达历史记忆中的水瓶位置，当前 RGB 没检测到 bottle。

系统不能直接删除记忆，而要分类当前没看到的原因：

```text
positive observation:
  当前确实看到 bottle。

not observable:
  当前没看到，但视角、高度、遮挡或覆盖条件不足以判断。

expected-visible miss:
  历史 bottle 位置在当前可见范围内，理论上应该看到，但没有看到。
```

只有 `expected-visible miss` 计为 negative evidence。

更新示例：

```text
step 1:
  status = active
  confidence = 0.82

expected-visible miss #1:
  confidence = 0.70
  status = active

expected-visible miss #2:
  confidence = 0.58
  status = stale

expected-visible miss #3:
  confidence = 0.42
  status = stale

expected-visible miss #4:
  confidence = 0.30
  status = missing
```

### 6. 自动二次规划

当原候选变 stale / missing 后，系统重新检索：

```text
exclude or downweight stale bottle
prefer active alternatives:
  cup near sink
  sink
  fridge
  kitchen counter
```

示例 replanning 输出：

```json
{
  "task_plan": [
    {"step": 1, "intent": "go_to_landmark", "target": "sink"},
    {"step": 2, "intent": "search_nearby", "target": "cup_or_faucet"},
    {"step": 3, "intent": "stop_if_affordance_found", "target": "drink_water"}
  ],
  "stopover_waypoints": ["wp_sink_02"],
  "selected_waypoint_id": "wp_sink_02",
  "stop_probability": 0.76,
  "reason": "The remembered bottle is stale; sink and cup landmarks remain active for the drink-water goal."
}
```

这一步是 demo 的核心价值：

```text
memory reuse
-> evidence-aware update
-> stale object downweighting
-> semantic replanning
```

### 7. 进入新环境时触发 context remapping

如果机器人被带到另一个 apartment 或整体不同房间，系统不能通过慢衰减洗掉旧地图。

应触发 context remapping：

```text
geometry overlap low
semantic landmark overlap low
expected landmarks absent globally
new landmark distribution appears
```

输出：

```text
old context:
  context_id = home_A
  memory retained
  inactive for current task
  available if revisited

new context:
  context_id = home_B
  write new observations
  retrieve primarily within current context
```

如果回到 `home_A`，系统应能切回旧 context 并复用旧记忆。

## 页面可视化设计

理想 demo 页面包含 5 个区域：

### A. First-Person Observation

- RGB frame / GIF / video。
- depth 或 estimated depth。
- detector boxes / masks。
- 当前 query 和当前 step。

### B. Geometry / BEV

- traditional BEV / occupancy。
- explored area。
- robot trajectory。
- selected waypoint。
- planned path。

### C. Semantic Memory

- semantic BEV。
- object memory table。
- confidence / freshness / status。
- stale / missing object 高亮。
- negative evidence count 曲线。

### D. Landmark and Planner

- top-k landmarks。
- element topology graph。
- task plan JSON。
- waypoint scores。
- selected waypoint。
- stop probability。
- replanning reason。

### E. Context Remapping

- current context_id。
- context confidence。
- mismatch score。
- remap_triggered。
- old / new context memory summary。

## 当前项目已有基础

当前已经实现或跑通的部分：

```text
Habitat RGB-D / oracle pose input
OWLv2 / GroundingDINO / SAM semantic grounding MVP
semantic candidates -> BEV / semantic BEV
RSC object memory
negative evidence update MVP
object memory -> landmark retrieval
G/S/O/L representation bundle
qwen3-max API semantic planner
Phase5A next-waypoint selection MVP
```

关键产物：

```text
outputs/m35_semantic_representation_alignment/paper_semantic_bev_index_20260703.html
outputs/m35_semantic_representation_alignment/representation_bundle_best96_20260703/representation_bundle_report.html
outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_20260703/qwen3_max_20case_noleak_report.html
```

当前可作为 demo 早期版本的能力：

```text
RGB/RGB-D observation
-> semantic BEV MVP
-> object / landmark memory
-> qwen3-max selects semantic waypoint
```

## 尚需实现的关键模块

### 1. Interest-Driven Exploration

当前多为 scripted coverage loop。

需要实现：

```text
frontier score
semantic uncertainty score
landmark novelty score
connector / door priority
coverage gain
```

MVP 可先使用规则式 exploration interest，不需要训练。

### 2. Affordance Expansion

当前多是 `find bed / find chair` 这类直接 object query。

需要支持：

```text
我要喝水
-> bottle / cup / sink / fridge / faucet / kitchen counter
```

MVP 可先用 API 生成候选类别和权重，并缓存为结构化 JSON。

### 3. Same-Context Adaptive Update Experiment

Phase 2 已有 negative evidence 更新机制，但还需要正式 demo：

```text
object exists in old memory
-> robot visits old location
-> object not found under expected-visible condition
-> confidence decreases
-> status active -> stale -> missing
-> replanning triggered
```

### 4. Replanning Loop

需要将 planner 调用从一次性改为循环：

```text
plan
execute to waypoint
verify target
update memory
if failed and alternatives exist:
  re-rank candidates
  replan
```

### 5. Context Remapping Gate

当前只有 Phase4-lite 字段。

需要实现：

```text
mismatch score
context manager
create / switch context_id
old-context retention
return-to-old-context reuse
```

### 6. Habitat / Navmesh Execution

当前 Phase5A 使用 BEV waypoint proxy。

后续需要：

```text
waypoint -> navigable point
shortest path query
path length
path points
execution trace
failure reason
```

这不是主创新点，但会让 demo 的导航闭环更真实。

## Demo-final backlog: 当前不足与下一步

面向最终 demo，当前必须显式追踪以下待优化事项：

```text
1. Interest-driven exploration
   scripted coverage loop 只是 MVP substitute。
   后续需要把 frontier / uncertainty / landmark novelty / affordance prior
   组合成主动探索策略，让机器人决定下一步该看哪里、走哪里、是否回访。

2. Lingbo vision/depth model trial
   尝试灵波 vision 和 depth 模型，作为当前 Habitat oracle RGB-D / semantic 前端的替代或补强。
   最小验收：
     detection / grounding 是否覆盖 table/chair/door/wall/cup/bottle
     depth 是否能稳定支持 3D centroid / BEV projection
     failure case 是否能被 object-memory confidence 表达

3. Real computed localization
   当前 Habitat demo 使用 exact simulator pose。
   真实世界需要 SLAM / VIO / wheel odometry / relocalization。
   BEV/object memory 的稳定性必须显式面对：
     localization drift
     relocalization jump
     camera-depth calibration
     map alignment uncertainty
   后续置信度应拆成 semantic confidence + pose/map confidence。

4. API planner demo-level refinement
   API planner 已能将“找水处/餐桌水杯/回到主人床边”落到现有 semantic waypoint。
   下一步不是把 planner 手工规则化，而是增强 demo 链路表现：
     natural-language instruction
     -> qwen3-max subgoal sequence
     -> waypoint arrival
     -> stop and look
     -> verify target evidence
     -> update memory / replan if missing
```

## 真机扩展理想路线

理想情况下，该 demo 最终可以从 Habitat 仿真扩展到真实室内环境。

真机扩展不是第一版 MVP 的硬要求，但它是项目最有展示力的长期方向：

```text
real RGB stream
-> RGB-only geometry frontend
-> open-vocabulary semantic grounding
-> semantic BEV / object memory
-> landmark retrieval
-> adaptive update / context remapping
-> task planning
-> robot navigation or human-in-the-loop waypoint execution
```

### Real-World v0: 手机 / 手持 RGB 回放

第一步不直接控制机器人，而是用手机拍摄室内 RGB 视频或图片序列。

输入：

```text
phone RGB sequence
optional IMU / ARKit / ARCore pose
optional manually marked scale
```

处理：

```text
VGGT / DUST3R / MASt3R
-> camera poses / depth / point cloud
-> align to local world frame
-> semantic grounding
-> BEV / semantic BEV
-> RSC object / landmark memory
```

验收：

```text
1. 能从真实 RGB 序列生成可浏览 semantic BEV。
2. 能生成 object memory 和 landmark graph。
3. 能对 "我要喝水" 这类目标输出候选位置和 waypoint。
4. 暂不要求机器人真实移动。
```

该版本适合作为论文或作品集的真实场景补充图。

### Real-World v1: 人在回路的 waypoint 验证

第二步仍不强求机器人自动驾驶。

系统输出：

```text
go to table area
verify bottle
if missing, go to sink area
```

由人手持相机按系统 waypoint 走过去，系统持续接收 RGB 并更新 memory。

验收：

```text
1. 系统能基于真实 RGB 更新 object confidence。
2. 旧位置没有水瓶时，能触发 expected-visible miss。
3. confidence / status 变化可视化。
4. 能重新选择下一个候选位置。
```

该版本已经能验证“真实 RGB 下的可复用语义记忆与二次规划”，但避开底盘控制和碰撞安全复杂度。

### Real-World v2: 移动机器人导航

第三步接入真实机器人底盘，例如：

```text
ROS / ROS2 mobile base
RGB-D camera or RGB camera
wheel odometry / visual odometry / SLAM
local costmap
Nav2 / move_base
```

RSC-Nav 的职责：

```text
semantic memory
goal grounding
landmark retrieval
waypoint / stopover selection
memory update
context remapping
```

传统机器人栈职责：

```text
localization
obstacle avoidance
local planning
velocity command
emergency stop
```

也就是说，RSC-Nav 不直接输出低层速度控制，而是输出：

```text
semantic stopover waypoint
target object / region
stop probability
replanning trigger
```

再由 ROS Nav2 / local planner 执行。

### 真机扩展的关键挑战

真实环境相比 Habitat 会引入：

```text
scale ambiguity
pose drift
motion blur
lighting variation
object detector false positives
open-vocabulary label ambiguity
transparent / reflective objects
dynamic people / moved furniture
limited field of view
safety and collision avoidance
```

因此真机阶段必须保留模块化边界：

```text
RSC-Nav:
  semantic memory and planning

robot stack:
  localization, collision avoidance, low-level control
```

### 真机扩展验收建议

最低真机展示：

```text
1. 手机 RGB 扫描一个真实房间。
2. 构建 semantic BEV / object memory。
3. 用户输入 "我要喝水"。
4. 系统输出候选位置和 waypoint。
5. 手持相机到旧候选位置验证目标缺失。
6. 系统更新 memory 并重新推荐 sink / fridge / cup 等位置。
```

机器人版本展示：

```text
1. 机器人在真实房间中执行 waypoint。
2. RSC-Nav 只负责语义 stopover decision。
3. Nav2 / local planner 负责避障和路径执行。
4. 可视化展示 memory update 和 replanning。
```

论文表达上，真机可以作为 extension：

```text
simulation main experiments
+ real-world qualitative demo
```

不建议把真机作为最小毕业版本的主验收，因为它会引入大量与论文主变量无关的工程风险。

## MVP 验收标准

第一版 demo 不要求真机、不要求端到端训练。

最低验收：

```text
1. 使用 Habitat RGB-D / oracle pose 或已有 coverage-loop 数据建立 semantic memory。
2. 能展示 semantic BEV、object memory、landmark graph。
3. 用户 query = "我要喝水"。
4. 系统能扩展为 bottle / cup / sink / fridge 等候选。
5. planner 能选择第一候选 waypoint。
6. 到达旧位置后模拟或观测到 bottle missing。
7. negative evidence 使 bottle confidence 下降并变 stale / missing。
8. 系统自动重新检索并选择第二候选 waypoint。
9. 页面展示 plan -> verify -> update -> replan 的全过程。
```

增强验收：

```text
1. 使用真实 qwen3-max API planner。
2. 使用 Habitat navmesh shortest path 替代 BEV proxy。
3. A/B context remapping：home_A -> home_B -> home_A。
4. 对比 baseline：
   - memory reset
   - memory carried without adaptive update
   - memory carried with adaptive update
   - forced-single-context
   - context-remapping
```

## 论文表达价值

该 demo 对应论文贡献点：

```text
1. First-person observation 到可复用 semantic-spatial memory。
2. object / landmark memory 的长期复用。
3. evidence-aware adaptive update，避免“没看到”误判为“不存在”。
4. context remapping，避免新环境污染旧记忆。
5. semantic task planner 将记忆转为 stopover waypoints。
6. stale memory 触发二次规划，体现动态语义导航能力。
```

该 demo 比单次 `find bed` 更适合作为主展示，因为它能体现：

```text
不是一次性找物体，
而是长期可复用、可更新、可规划的动态语义空间记忆导航策略。
```

## 后续实施建议

推荐拆成三个增量版本：

```text
Demo v0:
  使用已有 best96 semantic memory。
  手工构造 "我要喝水" affordance candidates。
  qwen3-max 规划到第一候选。

Demo v1:
  加入 expected-visible miss simulation。
  object confidence 下降。
  stale 后自动二次规划。

Demo v2:
  加入 context remapping。
  home_A -> home_B -> home_A。
  旧 context 保留并可回访复用。
```

## 当前执行记录

### 2026-07-04: API + Habitat simulation MVP

已完成第一版仿真 demo：

```text
natural language:
  去找到有水的地方，然后回到主人（在床上）身边

planner:
  qwen3-max API

execution:
  Habitat navmesh shortest path
  first-person RGB video
  face target landmark/location and wait at each stopover
```

当前语义 map 可用标签：

```text
bed
chair
door
sofa
table
```

API planner 决策：

```text
water-place:
  selected label = table
  reason = 当前 map 缺少 sink / faucet / fridge / bottle / cup / kitchen / bathroom 等显式水相关标签；
           table 是现有语义元素中最合理的可搜索位置。

owner:
  selected label = bed
```

执行结果：

```text
segment 1:
  table waypoint reachable
  geodesic_distance_m = 2.5609

segment 2:
  bed waypoint reachable
  geodesic_distance_m = 10.4434

first-person frames:
  115

observation wait:
  2.5 s at water-place stopover
  2.5 s at owner/bed stopover
```

产物：

```text
outputs/phase5a_navmesh_validation/qwen3_max_find5_20260704/navmesh_validation_report.html
outputs/phase5a_sim_demo/water_then_owner_bed_20260704/demo_report.html
outputs/phase5a_sim_demo/water_then_owner_bed_20260704/water_then_owner_bed_first_person.mp4
outputs/phase5a_sim_demo/water_then_owner_bed_20260704/semantic_map_then_task_linked.gif
```

边界说明：

```text
1. 本次已验证 API-selected waypoint 能回到 Habitat navmesh 并真实可达。
2. “有水的地方”不是脚本硬编码 table，而是 API 根据当前语义 map 自行推理；但由于当前 map 标签有限，table 仍只是最佳可用 proxy。
3. 本次页面已补上“先进入环境、用 coverage-loop 构建语义地图，再输入自然语言指令执行”的联图 GIF；coverage-loop 是 MVP 替代，后续应由兴趣/好奇心探索策略接管。
4. 尚未加入 bottle / cup / sink / fridge / kitchen / bathroom 等更真实 water-affordance 类别。
5. 尚未实现旧位置没水后的 negative evidence update 和自动二次规划。
```

> 最后更新：2026-07-04
