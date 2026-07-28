# Phase 5: Navigation Policy

## 阶段定位

Phase 5 的目标是把 memory retrieval 结果注入导航决策，验证 RSC memory 是否能改善 waypoint / stop decision。

## 当前状态

状态：Phase5A planner/navmesh 最小闭环已通过；单场景内已跑通同步在线的“自主熟悉环境 -> 一次纠偏 -> API 规划 -> 任务执行”完整 MVP。覆盖收敛、负证据准确性和实例级 cup recall 仍待论文级验收。

说明：当前已完成的是 `planner_context.json -> semantic task plan -> waypoint ranking -> BEV waypoint executor -> trace/report` 的工程闭环。真实 API / VLM planner 和 Habitat navmesh shortest-path executor 尚未替换进来，仍属于 Phase5A 后续 refinement。

## 计划输入

- traditional BEV。
- semantic BEV。
- landmark retrieval results。
- context-selected memory。
- goal specification。

## 计划输出

- waypoint candidates。
- landmark-conditioned waypoint ranking。
- stop decision。
- navigation trace and metrics。

## 设计边界

主实验优先使用固定 planner / waypoint teacher，避免把结论混入策略学习能力。论文主变量仍然是长期记忆机制，而不是训练一个更强的低层控制器。

## 参考旧协议的阶段设计

旧协议中的“检索与策略头”在新阶段中拆为 Phase 3 retrieval 和 Phase 5 navigation。Phase 5 只处理如何使用检索结果行动：

- node-to-BEV attention / region projection。
- waypoint candidate from BEV。
- landmark-conditioned waypoint ranking。
- stop decision。
- shortest-path or waypoint teacher as supervision。

可选扩展：

- imitation waypoint head。
- learned stop head。
- direct action output ablation。

策略输入应保留：

```text
confidence
freshness
status flag
landmark relation
context match
```

使策略能够区分可靠新观测与过期旧记忆。

## 关键评估

- Success Rate。
- SPL。
- Path Length。
- Distance-to-Goal。
- wrong old-location stop rate。
- memory reuse gain。

## 待补

- real API / VLM planner 调用与稳定 JSON parser。
- Habitat navmesh shortest-path executor 替换当前 BEV waypoint proxy。
- geometry-only / API-without-memory / API-with-RSC-memory 三组最小 baseline。
- wrong-stop attribution logging。

## 阶段日志

### 2026-07-03: Phase5A MVP closed loop

目标：

```text
M3.5 planner_context.json
-> semantic task plan
-> stopover waypoint ranking
-> stop probability
-> traditional planner execution proxy
-> trace / metrics / HTML report
```

本次新增：

```text
scripts/phase5a_api_semantic_planner_eval.py
```

输入：

```text
outputs/m35_semantic_representation_alignment/representation_bundle_best96_20260703/planner_context.json
```

输出：

```text
outputs/phase5a_api_semantic_planner/best96_find_bed_20260703/phase5a_planner_report.html
outputs/phase5a_api_semantic_planner/best96_find_bed_20260703/planner_request.json
outputs/phase5a_api_semantic_planner/best96_find_bed_20260703/planner_output.json
outputs/phase5a_api_semantic_planner/best96_find_bed_20260703/execution_trace.json
outputs/phase5a_api_semantic_planner/best96_find_bed_20260703/metrics.json
```

当前运行模式：

```text
mode_used: deterministic
```

解释：

```text
脚本已支持 OpenAI-compatible API 调用：
  --mode api
  OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL

本次本地没有配置 API key，因此 auto 模式自动降级到 deterministic teacher。
这不代表最终 planner 是规则式；它只用于先固定 Phase5A 的输入输出、trace 和验收口径。
```

运行摘要：

```text
goal_query: find bed
selected landmark: lm_m25_gdino_full96_views2_conf035_bed_groundingdino_bed_015
selected waypoint: wp_lm_m25_gdino_full96_views2_conf035_bed_groundingdino_bed_015_0
stop_probability: 0.7515
executor: traditional_bev_waypoint_proxy_v1
segments: 1
path_length_m: 7.1999
final_distance_to_target_m: 0.9
checks: 6/6 passed
```

验收检查：

```text
json_parse_success: passed
stopover_waypoints_valid: passed
selected_waypoint_valid: passed
task_plan_references_known_memory: passed
traditional_planner_proxy_executable: passed
stop_decision_available: passed
```

边界说明：

```text
1. 当前 traditional planner execution 是 BEV candidate-waypoint proxy：
   它验证 waypoint 来自 candidate list，并生成 segment trace。
2. 真实 Habitat shortest-path / navmesh execution 尚未接入。
3. 当前只完成 API planner with RSC memory 的单例闭环；
   geometry-only、without-memory、with-memory 的 baseline 对照下一步补。
```

结论：

```text
Phase5A MVP closed-loop runnable。
下一步建议：
  A. 接入真实 OpenAI-compatible API / VLM planner，保留 deterministic teacher 作为 fallback。
  B. 接入 Habitat navmesh shortest-path executor，替换 BEV waypoint proxy。
  C. 补最小 baseline：geometry-only / API without memory / API with RSC memory。
```

### 2026-07-03: Real API model benchmark and final model pick

目标：

```text
测试用户提供的两个 OpenAI-compatible API：
  Ark Seed2.x
  DashScope Qwen3.x

并为 Phase5A semantic task planner 选择默认真实 API 模型。
```

测试模型：

```text
Seed2.0:
  doubao-seed-2-0-pro-260215

Seed2.1 candidate:
  doubao-seed-2-1-pro-260628

Qwen3:
  qwen3-max

Qwen3 newer candidate:
  qwen3.7-max
```

测试任务：

```text
find bed
find door
find sofa
```

统一输入：

```text
M3.5 planner_context.json
G/S/O/L representation bundle
candidate_waypoints
topk_landmarks
object_memory_summary
```

统一验收：

```text
JSON parse success
valid waypoint ids
valid selected_waypoint_id
task_plan references known memory
BEV waypoint executor trace generated
stop_probability exists
stop_success_proxy
final_distance_to_target_m
segments count
```

评测产物：

```text
outputs/phase5a_api_semantic_planner/api_model_benchmark_20260703/api_model_benchmark.html
outputs/phase5a_api_semantic_planner/api_model_benchmark_20260703/api_model_benchmark.json
outputs/phase5a_api_semantic_planner/api_model_benchmark_20260703/selected_phase5a_model.json
```

核心结果：

```text
qwen3-max:
  initial runnable: 3/3
  expanded runnable: 8/8
  expanded closed-loop success: 8/8
  mean_final_distance_m: 0.9
  mean_segments: 1.0

qwen3.7-max:
  initial runnable: 3/3
  expanded runnable: 8/8
  expanded closed-loop success: 5/8
  mean_final_distance_m: 3.3774
  mean_segments: 1.875
  issue: bed / chair / table 等任务中仍会选择多个或错误 stopover，最终偏离 selected target

doubao-seed-2-0-pro-260215:
  initial runnable: 3/3
  expanded runnable: 8/8
  expanded closed-loop success: 2/8
  mean_final_distance_m: 3.73
  mean_segments: 2.875
  success cases: find table / approach table
  issue: 多数任务倾向输出多个候选 stopover，而不是单个 next waypoint，导致最终位置偏离 selected target

doubao-seed-2-1-pro-260628:
  bed 任务 90s timeout
```

最终选择：

```text
Phase5A 默认真实 API planner:
  provider: DashScope OpenAI-compatible
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model: qwen3-max
```

选择理由：

```text
qwen3-max 在 bed / door / sofa 三个目标上均能：
  1. 输出可解析 JSON。
  2. 引用合法 landmark 和 waypoint id。
  3. 选择单个 next waypoint。
  4. 让 BEV waypoint executor 停到目标 0.9m 内。

扩展到 8 个样本后仍保持 8/8：
  find bed
  find door
  find sofa
  find chair
  find table
  go to chair
  approach table
  navigate to door

它最符合当前 Phase5A 的 next-waypoint semantic planner 需求。
```

严格复核：

```text
qwen3-max targeted stress test:
  cases: 20
  labels: bed / chair / door / sofa / table
  query styles: find / go to / approach / navigate to
  strict passed: 20/20
  goal_label_aligned: 20/20
  stop_success_proxy: 20/20
  mean_final_distance_m: 0.9
  mean_path_length_m: 5.6867

artifact:
  outputs/phase5a_api_semantic_planner/qwen3_max_20case_strict_20260703/qwen3_max_20case_strict_report.html
```

泄漏复核：

```text
qwen3-max no-leak targeted stress test:
  condition:
    prompt hides goal_matching_waypoint_ids
    evaluator still checks selected anchor label against the hidden target label

  cases: 20
  strict passed: 20/20
  goal_label_aligned: 20/20
  stop_success_proxy: 20/20
  exposed_goal_matching_waypoint_ids_total: 0
  auto_goal_waypoint_cases: 8
  mean_final_distance_m: 0.9

artifact:
  outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_20260703/qwen3_max_20case_noleak_report.html
```

解释：

```text
上一轮 20/20 不是由 goal_matching_waypoint_ids 直接暴露正确答案造成。
但该测试仍然属于结构化语义记忆接口下的 next-waypoint selection，
不是完整自然语言导航 benchmark。

仍然暴露给 planner 的字段：
  goal_query
  goal_target_label
  candidate_waypoints.anchor_label
  landmark label / confidence / position 等结构化语义记忆字段

这些字段是 Phase5A 设计中的真实 planner 输入，
不是评测泄漏；但它们确实让任务比端到端自由语言导航简单。
后续需要增加更难样本：
  paraphrase / multi-object instruction
  distractor landmarks
  API without RSC memory
  geometry-only baseline
  Habitat navmesh executor
```

本次复核修正了一个重要评估口径：

```text
第一版 expanded benchmark 主要检查：
  JSON 可解析
  selected_waypoint_id 合法
  BEV waypoint proxy 能停到 selected target 附近

但它没有显式检查：
  selected waypoint 的 anchor_label 是否等于 query 目标类别

因此 chair / table 等目标在上游缺少 goal-matching candidate waypoint 时，
模型可能选择 door waypoint，同时在 task_plan 中写 chair/table landmark，
旧指标会把这种情况误算为成功。
```

修正后：

```text
planner_request 新增：
  goal_target_label
  goal_matching_waypoint_ids

验收新增：
  selected_anchor_matches_goal_label

Phase5A runner 临时补齐：
  当 top-k landmark 中存在目标类别，但 candidate_waypoints 缺失对应 anchor_label 时，
  自动生成 wp_auto_* goal-perimeter waypoints。

后续需要前移到 M3.5 representation exporter：
  对每个 top-k landmark 都生成可验证的 candidate waypoint，
  不应只依赖 Phase5A runner 热补。
```

安全说明：

```text
API key 只在运行时通过环境变量或命令注入。
阶段文档、报告和 selected model config 不保存密钥正文。
```

### 2026-07-04: Habitat navmesh validation and natural-language sim demo

目标：

```text
qwen3-max selected waypoint
-> Habitat navmesh snap
-> shortest-path reachability
-> first-person simulation execution trace
```

新增脚本：

```text
scripts/phase5a_navmesh_validate.py
scripts/phase5a_sim_language_demo.py
scripts/phase5a_make_storyboard_demo.py
```

5-case navmesh validation：

```text
cases:
  find bed
  find chair
  find door
  find sofa
  find table

result:
  reachable: 5/5
  snap_ok: 5/5
  all_passed: true

artifact:
  outputs/phase5a_navmesh_validation/qwen3_max_find5_20260704/navmesh_validation_report.html
```

自然语言仿真 demo：

```text
input:
  去找到有水的地方，然后回到主人（在床上）身边

planner:
  qwen3-max API

semantic map labels available in current MVP:
  bed / chair / door / sofa / table

planner decision:
  water-place step -> table waypoint
  return-to-owner step -> bed waypoint

planner reasoning:
  当前 semantic map 缺少 sink / faucet / fridge / bottle / cup / kitchen / bathroom 等显式水相关标签；
  在现有 map 中 table 是最合理的可搜索位置，因为桌面常可放杯子或水瓶。

execution:
  segment 1 table: reachable, geodesic 2.5609 m
  segment 2 bed: reachable, geodesic 10.4434 m
  first-person frames: 115
  observation wait: 2.5 s at each stopover
  wait behavior: face the target landmark / location before continuing
  video: water_then_owner_bed_first_person.mp4

map-before-task storyboard:
  stage 1: coverage-loop traversal as MVP substitute for future curiosity/interest exploration
  stage 2: freeze constructed semantic BEV / object memory, then run API planner and navmesh execution
  linked GIF: semantic_map_then_task_linked.gif
  frames: 56 mapping + 72 task execution at 6 fps

artifact:
  outputs/phase5a_sim_demo/water_then_owner_bed_20260704/demo_report.html
```

本次边界：

```text
1. 本 demo 已经不是硬编码 table；API planner 读取 semantic map 后自行选择当前最合理的 water-place candidate。
2. 由于当前语义 map 没有真实 water affordance 类别，planner 的 table 选择属于“不确定条件下的最佳可用搜索点”。
3. 当前 demo 的“初入环境建图”使用固定 coverage-loop 遍历；后续应替换为 exploration-interest / curiosity policy。
4. 当前 demo 展示 map -> plan -> navmesh execution，不包含旧位置没水后的 negative-evidence update / replan。
5. 下一步应加入 bottle/cup/sink/fridge/kitchen/bathroom 等语义候选，或构造同环境对象缺失实验，形成完整 plan -> verify -> update -> replan。
```

> **最后更新时间:** 2026-07-04

### 2026-07-04: Case2 natural-language sim demo

输入：

```text
帮我去客厅餐桌上找个水杯拿回给我
```

执行链路：

```text
coverage-loop mapping storyboard
-> qwen3-max semantic planner
-> table waypoint as cup/dining-table proxy
-> bed waypoint as owner return location
-> Habitat navmesh shortest path
-> first-person execution video with 2.5 s observation wait at each stopover
```

Planner 决策：

```text
target landmark:
  lm_m25_gdino_full96_views2_conf035_table_groundingdino_table_083

water/cup proxy waypoint:
  wp_auto_lm_m25_gdino_full96_views2_conf035_table_groundingdino_table_083_2

return-to-owner waypoint:
  wp_lm_m25_gdino_full96_views2_conf035_bed_groundingdino_bed_015_2
```

结果：

```text
segment 1 table: reachable, geodesic 2.5609 m
segment 2 bed: reachable, geodesic 10.4434 m
first-person frames: 115
linked storyboard: 56 mapping frames + 72 task frames at 6 fps
```

产物：

```text
outputs/phase5a_sim_demo/case2_cup_from_dining_table_20260704/demo_report.html
outputs/phase5a_sim_demo/case2_cup_from_dining_table_20260704/semantic_map_then_task_linked.gif
outputs/phase5a_sim_demo/case2_cup_from_dining_table_20260704/water_then_owner_bed_first_person.mp4
```

边界说明：

```text
当前 semantic map 没有 cup / dining table / living room 的显式细粒度标签。
因此 case2 验证的是自然语言目标到现有 table landmark 的可执行映射，
不是验证水杯实体本身的 open-vocabulary detection。
后续应加入 cup/bottle/counter/dining-table/room-function labels，
再做物体级取回任务和 negative-evidence replan。
```

> **最后更新时间:** 2026-07-04

### 2026-07-04: Storyboard semantic panel updated to object-centric evidence

变更：

```text
linked storyboard 左下角语义地图面板
from:
  Phase2-style dense colored semantic BEV
to:
  object_inventory_projection_evidence.png
  object-centric semantic evidence / observation centroids
```

理由：

```text
Phase2 染色 semantic BEV 更适合说明语义证据累积和稳定性；
Phase5A demo 更需要表达 planner 输入来自 object memory / landmark candidates，
因此 object observation centroid + label/confidence 的表示更贴近论文级语义记忆和后续任务规划叙事。
```

当前使用资产：

```text
outputs/m35_semantic_representation_alignment/representation_bundle_best96_20260703/assets/object_inventory_projection_evidence.png
```

已更新：

```text
outputs/phase5a_sim_demo/water_then_owner_bed_20260704/semantic_map_then_task_linked.gif
outputs/phase5a_sim_demo/case2_cup_from_dining_table_20260704/semantic_map_then_task_linked.gif
```

> **最后更新时间:** 2026-07-04

### 2026-07-23: From-zero room familiarization and all-cup search demo

自然语言任务统一为：

```text
自行熟悉房间并找到所有水杯
```

本次不再复用已有 semantic BEV，也不再把 table 当作 cup proxy。完整链路为：

```text
empty BEV / empty object memory
-> Habitat coverage-loop RGB-D + pose exploration
-> online-style traditional BEV reconstruction
-> GroundingDINO open-vocabulary cup/table/counter/sink evidence
-> depth + pose projection to 3D
-> table/counter/sink candidate selection
-> navmesh active search with <=15 degree adjacent-frame yaw
-> multi-view spatial evidence merge
-> stable cup tracks in semantic BEV / object memory
```

测试场景改为 HM3D `00861-GLAQ4DNUx5U`。该场景的 post-hoc semantic
oracle 含 16 个在输入帧中可见的 `glass` 实例；oracle 不参与探索、候选选择、
GroundingDINO 推理或 memory 更新。

固定判定参数与结果：

```text
cup minimum independent views: 5
cup minimum mean confidence: 0.28
3D merge radius: 0.30 m
active-search yaw increment: 15 degrees

combined RGB-D observations: 523
GroundingDINO projected detections: 1719
multi-view confirmed cup tracks: 4
post-hoc oracle-visible glass instance IDs: 16
active-search maximum adjacent-frame yaw: 15.0 degrees
active-search maximum adjacent-frame translation: 0.30 m
```

边界：

```text
1. “所有水杯”表示当前覆盖和主动扫描下所有通过固定门槛的杯具轨迹。
2. HM3D glass 和 natural-language water cup 类别粒度不同，4 / 16 不是实例级 recall。
3. 原始 evidence tracks 保留在 metrics 中；主图仅显示稳定轨迹。
4. GroundingDINO-tiny 仍存在误检，多视角 3D 一致性不能替代更强 grounding、
   segmentation 和实例级关联。
5. 当前“熟悉房间”仍由 coverage-loop 代替兴趣/好奇心探索。
```

产物：

```text
outputs/phase5a_sim_demo/zero_map_find_all_cups_hm3d_20260723/final_report/zero_map_find_all_cups.html
outputs/phase5a_sim_demo/zero_map_find_all_cups_hm3d_20260723/final_report/zero_map_find_all_cups.gif
outputs/phase5a_sim_demo/zero_map_find_all_cups_hm3d_20260723/final_report/zero_map_find_all_cups.mp4
outputs/phase5a_sim_demo/zero_map_find_all_cups_hm3d_20260723/final_report/zero_map_find_all_cups_metrics.json
```

旧 Case1/Case2 保留为历史 planner/navmesh smoke test，不再作为 GitHub 首页主 demo。

> **最后更新时间:** 2026-07-23

## 2026-07-24：完整在线 Demo 实现与验收

### 实现目标

本轮不是生成离线故事板，而是在同一个 Habitat episode 内实际执行以下因果链路：

```text
从零启动
-> 在线 RGB-D GroundingDINO
-> DenseBEV + 三态对象记忆
-> 兴趣/frontier 自主熟悉环境
-> 一次预先声明的局部纠偏
-> MemoryReady
-> 注入“请找到房间里的所有水杯，并按位置汇报”
-> Qwen3-Max 根据当时已有语义记忆排序候选
-> 传统 BEV/navmesh 执行
-> 新发现 cup 候选在线插入剩余 support 搜索之前
-> 候选自然耗尽后结束
```

约束：

- task 在熟悉环境完成前不可见；
- API 只接收 task-start 时刻已经存在的对象/支撑面候选；
- semantic oracle 不参与在线决策；
- 全场 navmesh 仅作为特权几何低层执行器；
- coverage oracle 只在 episode 结束后评估。

### 新增实现

```text
src/online_semantic_task_planner.py
scripts/phase5a_online_interest_explorer.py
scripts/phase5a_online_interest_report.py
tests/test_online_semantic_task_planner.py
```

关键行为：

- Qwen3-Max 真实 API 输出只允许引用输入中的 candidate ID；
- API 失败时保留可审计的 deterministic fallback；
- support surface 做空间去重并限制为 6 个代表区域，避免碎片轨迹耗尽预算；
- task execution 期间新出现的 cup memory 会插到剩余 support 候选之前；
- 保存 `autonomous_before_guidance`、`after_guidance`、`task_start`、`final` 四组 BEV、memory、tracks 检查点；
- 联图逐帧显示当前 phase、动作、active candidate、三态 evidence 和正常速度倍数。

### 正式运行

```text
run:
  outputs/phase5a_sim_demo/full_demo_autonomous_guided_api_task_final_v3_20260724/

steps:
  total: 771
  familiarization complete: 360
  task injected: 360
  API plan ready: 370
  task execution: 401

planner:
  model: qwen3-max
  mode: real API
  latency: 9.52 s
  API error: none

exploration:
  one correction: step 120 -> 248
  correction explored-cell gain: +13,286
  post-hoc navmesh observation coverage: 94.60%
  detected collisions: 0
  stuck recoveries: 2

task:
  task-start candidates: 3 cup tracks + 6 deduplicated support regions
  inspected cup tracks: 5
  search-stage multi-view confirmed cup tracks: 4
  scanned support regions: 4
  stop reason: task_execution_exhausted
```

在线插队证据：

```text
step 691:
  track_206 first becomes an eligible cup candidate
  track_206 is inserted before the remaining support candidate
  focused re-observation starts in the same step
```

这说明任务执行不是固定回放 task-start 的静态列表，而会消费持续更新的对象记忆。

### 迭代记录

```text
v1:
  29 个碎片化 support tracks 进入队列，任务预算耗尽。

v2:
  限制 support 数量后仍把动态 cup 加到队尾；
  新 cup 会被剩余 support 延迟，任务预算仍耗尽。

v3:
  support 空间去重为 6 个代表区域；
  动态 cup 插入剩余 support 之前；
  任务队列自然耗尽并正常结束。
```

### 验收边界

本轮可以声称：

- 完成单场景、同步在线、因果可审计的完整 Demo；
- 自主探索、一次纠偏、真实 API 规划和在线任务执行在同一 episode 内闭合；
- 任务执行会响应运行时新增的 cup memory；
- 可视化从零展示 BEV/对象记忆增长，而非加载已有地图。

本轮不能声称：

- 4 个确认 track 等于 4 个真实杯具实例；
- 已证明找到场景内全部真实水杯；
- 已完成 RGB-only 几何；当前仍使用 Habitat depth、exact pose 和 navmesh；
- GroundingDINO 的碎片轨迹和误检已经解决；
- 单场景 94.60% post-hoc coverage 可以代表跨场景泛化。

正式验收产物：

```text
report:
  outputs/phase5a_sim_demo/full_demo_autonomous_guided_api_task_final_v3_20260724/report/online_interest_exploration.html
  outputs/phase5a_sim_demo/full_demo_autonomous_guided_api_task_final_v3_20260724/report/online_interest_exploration.gif
  outputs/phase5a_sim_demo/full_demo_autonomous_guided_api_task_final_v3_20260724/report_step_by_step_3x/online_interest_exploration.html
  outputs/phase5a_sim_demo/full_demo_autonomous_guided_api_task_final_v3_20260724/report_step_by_step_3x/online_interest_exploration.gif
  outputs/phase5a_sim_demo/full_demo_autonomous_guided_api_task_final_v3_20260724/report_step_by_step_30s/online_interest_exploration.html
  outputs/phase5a_sim_demo/full_demo_autonomous_guided_api_task_final_v3_20260724/report_step_by_step_30s/online_interest_exploration.gif

audit:
  online_trace.jsonl
  online_summary.json
  posthoc_coverage_metrics.json
  task_planner/planner_request.json
  task_planner/planner_output.json
  checkpoints/
```

验证：

```text
remote full unit tests: 16 / 16 passed
HTML local media references: 5 / 5 resolved
GIF: 960 x 540, about 51 MB
step-by-step GIF: 771 / 771 steps, 330 ms per frame, 3.03x normal,
  960 x 540, about 153 MB, total playback about 4 min 14 s
30-second GIF: 771 / 771 steps, 40 ms per frame, 25.0x normal,
  960 x 540, about 153 MB, total playback 30.84 s
```

> **本节更新时间:** 2026-07-24

## 2026-07-23：层次兴趣探索与 LingBot-Map 在线接入

### 目标与参考

本轮针对“撞墙后持续前进、重复观察已看区域、未覆盖完整场景”的问题，引入并测试：

```text
frontier connected components
-> cluster-level candidate viewpoints
-> ray-cast visible unknown gain
-> geodesic path cost + revisit cost + obstacle risk
-> target viewpoint yaw
-> arrival alignment + compact fan scan
```

方法设计参考了 FUEL/TARE 的分层前沿与路径代价思想、next-best-view
的信息增益思想，以及 Nav2 的 blocked-target recovery/blacklist 机制。当前实现是
轻量二维 BEV 版本，不等同于这些工作的完整复现。

### LingBot-Map 接入边界

新增常驻 LingBot worker，保留模型 KV cache，并按以下顺序参与真实在线循环：

```text
8 current/past RGB frames causal bootstrap
-> fixed one-time depth-scale calibration
-> current RGB -> LingBot depth/confidence
-> confidence-gated DenseBEVMapper
-> GroundingDINO + predicted depth 3D projection
-> RSC object memory
-> same hierarchical frontier policy
-> Habitat discrete action
```

本轮 LingBot 组仍使用 Habitat exact pose，且 bootstrap 阶段用 Habitat depth
做一次固定尺度标定。因此结论级别是 `LingBot predicted-depth + exact-pose diagnostic`，
不是完整 RGB-only SLAM。在线策略不读取 semantic sensor；完整 navmesh 仅用于统一
低层路径执行，覆盖率由 episode 结束后的独立 evaluator 计算。

### 同场景对照

统一条件：

```text
scene: HM3D 00861-GLAQ4DNUx5U
start: [-9.366882, -1.592887, 6.400316], yaw=0
main action budget: 240
resolution: 384 x 384
frontier arrival fan scan: 45 degrees
low-level executor: hybrid_navmesh
semantic sensor: disabled
```

最终主表：

| 组别 | 策略 | 几何来源 | 步数 | 后验覆盖率 | 低增益帧 | 扫描动作 | 均值闭环延迟 | 碰撞/卡住 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A3 | greedy frontier | Habitat RGB-D + exact pose | 240 | 86.02% | 49.6% | 72 | 406 ms | 0 / 1 |
| B2 | hierarchical multi-view | Habitat RGB-D + exact pose | 241 | 84.13% | 46.5% | 99 | 647 ms | 0 / 1 |
| C2 | hierarchical multi-view | LingBot depth + exact pose | 241 | 80.54% | 36.9% | 91 | 829 ms | 0 / 0 |

解释：

```text
B2 vs A3:
  coverage = -1.89 percentage points
  low-gain frame ratio = -3.1 percentage points
  current hierarchical implementation did not beat greedy under fixed action budget.
  It must remain an experimental branch instead of replacing the default policy.

C2 vs B2:
  coverage = -3.59 percentage points
  mean loop latency = +182 ms
  completed the full online loop with zero collision/stuck.
  This passes the preliminary <=8 percentage-point non-inferiority diagnostic,
  but does not establish RGB-only superiority.
```

LingBot 在线 BEV 能保持主走廊和左右房间的整体结构，但墙体更厚、存在放射状噪声，
局部门洞和房间边界更容易断裂。其在线“探索格/低增益帧”可能受预测深度噪声虚增，
所以跨几何前端比较以 post-hoc navmesh observation coverage 为主。

### 实现与验证

新增或更新：

```text
src/interest_exploration.py
scripts/phase5a_online_interest_explorer.py
scripts/lingbot_map_online_worker.py
scripts/phase5a_exploration_ablation_report.py
tests/test_interest_exploration.py
```

验证：

```text
10 interest-exploration unit tests: passed
40-step hierarchical smoke: passed, 0 collision / 0 stuck
240-step A3/B2/C2 paired runs: completed
post-hoc navmesh coverage audit: completed
HTML + GIF reports: generated
```

产物：

```text
outputs/phase5a_sim_demo/exploration_ablation_A3_greedy_hardbudget240_20260723/
outputs/phase5a_sim_demo/exploration_ablation_B2_hierarchical_multiview_habitat240_20260723/
outputs/phase5a_sim_demo/exploration_ablation_C2_hierarchical_multiview_lingbot240_20260723/
outputs/phase5a_sim_demo/exploration_ablation_lingbot_multiview_20260723/exploration_ablation.html
```

### 后续优化

```text
1. hierarchy:
   加入跨帧稳定 cluster ID、目标切换迟滞和 room/region-level coverage；
   当前每次重聚类缺少真正的全局拓扑层。

2. scoring:
   将“到点额外扫描动作”计入 candidate utility；
   使用 coverage-AUC 和 actions-to-85%-coverage 调参，避免只优化局部 unknown gain。

3. execution:
   主探索实验改为 observed-BEV local planner；
   hybrid_navmesh 只保留为低层执行上界。

4. LingBot geometry:
   用 persistent pose 输出替换 exact pose；
   以相机高度/动作里程计做因果尺度锚定，并加入 pose-jump/submap 处理。

5. evaluation:
   扩展到 3+ HM3D scenes、每场景 5 seeds；
   当前单场景结果只能作为工程诊断，不能作为论文统计结论。
```

> **本节更新时间:** 2026-07-23

## 2026-07-24：一次人工引导纠偏覆盖右上漏扫区域

### 干预协议

上一版最终 BEV 的右上区域没有实际轨迹进入。经用户明确授权，本轮允许一次主动引导，
并将它作为公开、可审计的干预，而不是自主策略结果：

```text
selection:
  在 BEV 右上象限的 Habitat navmesh 可导航点中，
  选择距离既有轨迹最远的点。

target world XYZ:
  [0.059, -1.5929, 8.886]

trigger:
  deep_familiarization step 120

execution:
  navmesh path + normal discrete actions
  no teleport
  360-degree horizontal scan + downward scan

after completion:
  guided correction permanently disabled
  resume autonomous familiarization
  then enter autonomous cup_search
```

GIF 中该阶段显示为 `deep_fam | GUIDE`，逐步 trace 使用
`guided_correction_navigation` / `guided_correction_scan`，summary 保存目标、
起止 step 和新增探索格。

### 最终运行结果

```text
run:
  deep_familiarization_guided_right_then_cup_search_final_20260724

total steps: 821
guided correction start / complete: 120 / 212
guided explored-cell gain: +8,474
familiarization complete: step 364
cup-search actions: 456

post-hoc navmesh coverage:
  before correction version: 86.03%
  guided final: 94.08%
  delta: +8.05 percentage points

stable cup candidate tracks: 11
inspected/failed-once cup track IDs: 12
search-stage re-confirmed tracks: 10
support-surface regions inspected: 6
collisions / stuck events: 0 / 1
```

熟悉阶段在扩展右上区域后：

```text
recent 60-step explored-cell gain: 17
known-free cells reobserved >=2 times: 49.13%
transition reason: max_familiarization_steps
```

重复观测比例低于无干预版的 63.17%，原因是新增区域扩大了已知自由空间分母，
且本轮只允许一次人工引导，没有再人工引导第二遍。几何覆盖更完整，但新区域的复访深度
仍是后续自主 revisit policy 要解决的问题。

### 同步修复

纠偏后的新区域产生了靠墙 cup 候选。第一次正式试跑暴露出：

```text
unreachable/stuck cup candidate
-> recovery
-> same candidate selected again
-> repeated budget consumption
```

现已改为：

```text
cup candidate unreachable or hard navigation failure
-> record failure in trace
-> mark inspected for this episode
-> do not select it again
-> continue next cup/surface target
```

修复后 stuck 记录由 12 降到 1，并恢复了支持面巡视和后续候选搜索。

### 结论边界

```text
can claim:
  一次人工航点纠偏确实补齐了右上漏扫区；
  干预前后覆盖增益有独立 post-hoc evaluator；
  纠偏后系统能恢复自主熟悉和 cup_search。

must not claim:
  94.08% 是纯自主兴趣探索覆盖率；
  10 个确认轨迹等于 10 个真实杯子；
  右上新增区域已达到与旧区域相同的重复熟悉程度。
```

产物：

```text
outputs/phase5a_sim_demo/deep_familiarization_guided_right_then_cup_search_final_20260724/
outputs/phase5a_sim_demo/deep_familiarization_guided_right_then_cup_search_final_20260724/report/
```

> **本节更新时间:** 2026-07-24

## 2026-07-24：完整 Demo 协议固化

在一次右上区域纠偏试验的基础上，进一步固化正式演示协议：

```text
autonomous exploration from empty memory
-> one human coverage correction
-> resume autonomous familiarization
-> MemoryReady checkpoint
-> inject natural-language task
-> API semantic task planning
-> executable waypoint navigation
-> observation / memory update / replanning
-> final report
```

关键边界：

```text
纠偏只能指定未充分覆盖的空间区域，不能指定任务对象；
任务在熟悉阶段结束后才输入；
Habitat oracle 只做 post-hoc 评测；
高层 API planner 不直接输出底层动作；
traditional BEV/navmesh 继续负责可执行路径。
```

完整的状态机、镜头脚本、检查点、指标和验收标准见：

```text
docs/demo_autonomous_explore_guided_correction_task.md
```

当前引导版本可作为 Act 1-3 的原型，但还需接入运行时任务注入、Qwen3-Max
结构化计划、四个地图检查点和三态证据可视化，才能称为完整在线闭环 Demo。

> **本节更新时间:** 2026-07-24

## 2026-07-23：深度熟悉后再执行杯具搜索

### 修正的问题

此前在线演示同时运行 Grounding 和 object memory，但主要动作仍停留在环境覆盖阶段；
“覆盖过程中检测到 cup”不等于“熟悉完成后主动寻找 cup”。本轮将任务改成不可逆的两阶段状态机：

```text
Stage 1: deep_familiarization
  frontier exploration
  + repeated observation
  + causal map-saturation test

transition:
  minimum familiarization steps reached
  AND recent explored-cell gain saturated
  AND known free-space reobservation ratio reached
  OR causal maximum familiarization budget reached

Stage 2: cup_search
  stable cup memory candidates first
  -> navigate to candidate
  -> face candidate and perform compact fan scan
  -> require search-stage reobservation for confirmation
  -> then inspect table/counter/sink regions for missed cups
```

进入 `cup_search` 后不再因零碎 frontier 退回熟悉阶段。状态切换、熟悉快照、
候选检查和搜索阶段确认均写入逐步 JSONL trace。

### 正式长程运行

```text
scene: HM3D 00861-GLAQ4DNUx5U
task: 先深度熟悉房间，再根据语义记忆找到所有水杯
frontier policy: greedy
geometry: Habitat RGB-D + exact pose
total run: 761 steps
```

熟悉阶段：

```text
familiarization complete step: 332
transition reason: saturated_and_reobserved
recent 60-step explored-cell gain: 0
known-free cells reobserved >=2 times: 63.17%
explored BEV cells at transition: 39,239
post-hoc navmesh observation coverage at run end: 86.03%
```

搜索阶段：

```text
cup-search actions: 428
stable cup candidates at transition: 2
cup candidate tracks inspected during search: 8
support-surface regions inspected: 6
final stable candidate tracks: 9
search-stage re-confirmed candidate tracks: 7
collisions / stuck events: 0 / 1
stop reason: step budget completed current scan
```

解释边界：

```text
1. 7 是 GroundingDINO 多视角轨迹数，不是 Habitat 真实杯子实例数。
2. 邻近杯具轨迹仍可能由 box-depth centroid 抖动产生碎片，不能宣称找到 7 个真实杯子。
3. 相比 240-step A3，跑后几何覆盖率没有明显增加；“深度熟悉”的直接收益是
   重复观测比例和语义记忆稳定性，而不是更大的场景覆盖面积。
4. 完整实例 recall 仍需独立 Habitat semantic oracle 或人工标注在 episode 后评估。
```

### 可视化时间语义

报告新增明确回放倍率：

```text
normal reference: 1 discrete action = 1.0 s
GIF playback: 8.3x normal
gif frame stride: 3
per-displayed-frame duration: 360 ms
```

该倍率只描述回放节奏，不代表模型实时推理速度；真实闭环耗时继续单独显示在每帧状态栏。
画面同时显示 `deep_familiarization` / `cup_search`，避免把两个阶段混为一谈。

实现与产物：

```text
src/interest_exploration.py
scripts/phase5a_online_interest_explorer.py
scripts/phase5a_online_interest_report.py
tests/test_interest_exploration.py

outputs/phase5a_sim_demo/deep_familiarization_then_cup_search_20260723/
outputs/phase5a_sim_demo/deep_familiarization_then_cup_search_20260723/report/
```

验证：

```text
12 unit tests: passed
forced transition smoke: passed
cup candidate approach/scan smoke: passed
761-step formal online run: completed
post-hoc coverage audit: completed
HTML/GIF report with playback-speed annotation: generated
```

> **本节更新时间:** 2026-07-23

### 2026-07-23: Current final state - online alignment with the offline coverage reference

> 本节记录当前最终实现与结论；其后的 `Real-time interest-driven exploration MVP`
> 保留为改造前的历史基线，不再代表当前能力。

目标：

```text
以 zero_map_find_all_cups_hm3d 非实时 Demo 的覆盖、台面巡视和稳定语义图为行为参考，
在不使用未来帧、semantic sensor、semantic scene 或预制 coverage route 的前提下，
修复实时探索中的撞墙、转向振荡、重复巡视和瞬时 track 过度显示。
```

本轮实现：

```text
1. observed BEV frontier:
   仍由截至当前帧的 depth BEV 产生探索目标，并按信息增益、距离和重访成本排序。

2. reachable low-level execution:
   对当前在线目标调用完整场景 Habitat navmesh global shortest path，
   再取短 lookahead 逐步转向/前进；
   navmesh 不负责选目标、不读取语义、不提供预制整段探索路线，
   但它仍是当前仿真中的 privileged geometric oracle。

3. navigation robustness:
   修复 30-degree 离散转向在航点方向两侧振荡；
   引入障碍膨胀、blocked-frontier blacklist、无进展检测和恢复动作；
   单次轻微碰撞不再立即永久拉黑整个前沿区域。

4. coverage-first state machine:
   coverage -> repeated completion scans -> semantic inspection；
   连续两次低新增完整扫描后才进入语义巡视。

5. spatial semantic revisit suppression:
   table / counter / sink 按世界坐标区域去重；
   到达新区域后执行俯视与环视，避免重复检查同一张桌子。

6. stable semantic display:
   原始 online tracks 继续进入可审计 memory；
   BEV 只绘制达到多视角/置信度门槛的稳定 tracks。

7. candidate / confirmation split:
   candidate cup = 覆盖过程中形成的稳定多视角候选；
   focused-confirmed cup = candidate 且至少在一个 semantic surface scan 帧中再次观测到；
   当前不要求多次 focused 命中，也未验证它与当前台面的实例归属。

8. graceful stop:
   动作预算到点后完成当前扫描，不再领取新兴趣目标。
```

最终运行：

```text
scene: HM3D 00861-GLAQ4DNUx5U
task: 自行熟悉房间并找到所有水杯
model: GroundingDINO-tiny persistent worker
resolution: 384 x 384
normal action budget: 650
actual recorded steps: 666

coverage phase:
  initial panorama: 12
  frontier exploration: 231
  frontier/completion scans: 148
  coverage confirmations: 2

semantic phase:
  semantic-interest navigation: 111
  semantic surface scan: 162
  graceful-stop marker: 1 after the final scan completed
  independent scanned surface regions: 9

motion:
  move_forward actions: 261
  effective forward actions: 261
  detected collisions: 0
  stuck events: 0
  blacklisted frontiers: 1

mapping:
  final explored cells: 41040
  post-hoc navmesh observation coverage: 86.53%

semantics:
  raw memory items: 214
  stable cup candidates: 8
  focused-scan confirmed cup candidates: 7

runtime:
  mean closed-loop latency: 393.6 ms / step
  p95 closed-loop latency: 497.7 ms / step
  approximate sustained rate: 2.54 Hz
```

验收判断：

```text
passed:
  实时因果闭环保持成立。
  在线 BEV 确实参与 frontier target selection。
  本次 run 的 261 次 move_forward 均产生有效位移，未出现旧版持续撞墙现象。
  不再出现离散转向 left/right 振荡。
  覆盖优先于语义兴趣，且不同台面区域按空间去重。
  最后一个扫描动作序列完整结束。
  稳定语义图不再显示全部瞬时投影轨迹。

partially passed:
  轨迹、覆盖优先和台面巡视在定性行为上接近非实时 coverage-loop 参考；
  旧离线版尚未按相同 navmesh 分母重算，不能宣称定量等价或优于；
  单楼层后验 navmesh 观察覆盖为 86.5%，不能声称完整覆盖。
  实时稳定 cup 候选由 15 降到 8，仍高于离线版 4；
  7 条 focused-confirmed 仅表示曾在巡视帧再次命中，不能替代实例级真值验证。
  本次未发生 collision/stuck，恢复代码路径未在该 final run 中被触发验收。

remaining:
  GroundingDINO box-depth centroid 仍有假阳性和关联碎片；
  下一轮应加入 SAM mask 投影、类别相关 association gate 和 track consolidation。
  observed-depth BEV 对门洞/边缘的连通表达限制了剩余空间的 frontier discovery。
  完整 Habitat navmesh global shortest path 是当前仿真低层执行 privilege；
  真机阶段应替换为传统 occupancy planner。
  运行因 step budget 收尾停止，而不是 interest_exhausted；最后仍有语义兴趣目标。
```

新增/更新实现：

```text
src/interest_exploration.py
scripts/phase5a_online_interest_explorer.py
scripts/phase5a_online_interest_report.py
scripts/phase5a_posthoc_coverage_eval.py
tests/test_interest_exploration.py
```

最终产物：

```text
remote:
  outputs/phase5a_sim_demo/online_interest_bev_aligned_final_20260723_v4/

local:
  outputs/phase5a_sim_demo/online_interest_bev_aligned_final_20260723_v4/report/online_interest_exploration.html
  outputs/phase5a_sim_demo/online_interest_bev_aligned_final_20260723_v4/report/online_interest_exploration.gif
  outputs/phase5a_sim_demo/online_interest_bev_aligned_final_20260723_v4/report/realtime_vs_offline.html

audit:
  online_trace.jsonl
  online_summary.json
  posthoc_coverage_metrics.json
```

独立只读审计：

```text
confirmed:
  未发现未来帧、semantic sensor 或 semantic_scene 泄漏。
  BEV frontier 确实来自当前 mapper.explored/free/observation_count。
  motion 全部通过 Habitat discrete sim.step，而非 teleport。
  trace 未出现导航阶段 left/right 连续振荡。

claim boundary:
  当前系统应称为：
  单场景、RGB-D/真值位姿、BEV 高层选点与 navmesh 特权低层执行的在线兴趣探索 MVP。

  不应称为：
  完整自主覆盖；
  纯 observed-BEV navigation；
  找到所有真实水杯；
  实例准确率达到或超过旧离线 Demo；
  collision recovery 已在 final run 中完成动态验收。
```

> **最后更新时间:** 2026-07-23

### 2026-07-23: Real-time interest-driven exploration MVP

目标：

```text
不再使用“先录制完整序列 -> 离线 Grounding -> 离线重放建图”的演示方式，
改为严格在线因果闭环：

current RGB-D + pose
-> persistent GroundingDINO inference
-> depth + pose 3D projection
-> traditional BEV update
-> object track / RSC memory update
-> interest-policy decision
-> Habitat discrete action
-> next observation
```

在线执行约束：

```text
semantic sensor: disabled
semantic_scene read: disabled
precomputed coverage route: disabled
navmesh target/path sampling: disabled
motion: Habitat sim.step discrete actions only
oracle: allowed only in a separate post-hoc evaluator
```

兴趣策略 MVP：

```text
1. initial panorama:
   24 * 15 degrees，先建立当前出生点的视角覆盖。

2. frontier information gain:
   从当前已观测 BEV 中提取 free/unknown frontier；
   score = information gain - distance cost - revisit cost。

3. semantic interest:
   对多视角稳定的 table / counter / sink 轨迹评分并主动接近、俯视和横向扫描。

4. spatial revisit suppression:
   已扫描表面按世界坐标半径去重，避免碎片 track 反复触发同一区域。

5. exploration/semantic alternation:
   每次语义区域扫描后设置 cooldown，强制回到 frontier exploration，
   防止语义兴趣长期压制空间覆盖。
```

记忆更新约束：

```text
positive observation:
  当前 GroundingDINO detection 经当前 depth + pose 投影后更新。

not observable:
  未形成 >=3 个独立视角的稳定轨迹，或历史 3D 高度/FOV/深度遮挡条件不满足。

expected-visible miss:
  只有稳定历史轨迹确实处于当前水平/垂直视野，且深度未显示前景遮挡时才成立。

evidence weight:
  由相邻真实 pose 的平移和旋转量经 smooth curve 计算；
  静止帧低权重，正常位移/转向较高，极端运动受抑制。
```

最终 240-step 运行：

```text
scene: HM3D 00861-GLAQ4DNUx5U
task: 自行熟悉房间并找到所有水杯
model: GroundingDINO-tiny, persistent worker
resolution: 384 x 384

initial panorama scan: 24 steps
frontier exploration: 71 steps
semantic-interest approach: 73 steps
semantic surface scan: 72 steps
forward motion: 50 actions / 10.71 m
collisions: 0

projected detections: 1092
raw online tracks: 139
independent multi-view confirmed cup tracks: 5
explored BEV cells: 31851

model cold start: 21.64 s
mean online loop: 373 ms / step
online loop frequency: about 2.7 Hz
p95 online loop: 448 ms
240-step accumulated online loop time: about 89.6 s
stop reason: max_steps
```

验收结论：

```text
passed:
  observe-ground-update-decide-step 顺序可由逐步 JSONL trace 审计。
  当前动作只依赖 current/past frames。
  GroundingDINO 只冷启动一次，随后逐帧同步推理。
  BEV 和 object memory 在动作执行过程中从零在线增长。
  frontier interest 与 semantic interest 均实际触发。
  semantic oracle / semantic scene / precomputed route 不参与在线决策。

not yet claimed:
  “找到全部真实杯具”的实例级 recall。
  全场景覆盖收敛。
  RGB-only geometry；当前仍使用 Habitat depth + exact pose。
  论文主实验级跨场景统计显著性。
```

独立只读审计结论：

```text
online causal loop:
  passed

semantic oracle / semantic_scene / precomputed route leakage:
  not found

allowed simulator privilege:
  navmesh random navigable initial spawn
  Habitat depth
  Habitat exact pose

claim level:
  single-scene synchronous online interest-driven exploration MVP at about 2.7 Hz

must not claim:
  complete autonomous room coverage
  all real cups found
  negative-evidence accuracy validated
  pure-RGB online mapping
```

新增实现：

```text
src/interest_exploration.py
scripts/groundingdino_online_worker.py
scripts/phase5a_online_interest_explorer.py
scripts/phase5a_online_interest_report.py
tests/test_interest_exploration.py
```

产物：

```text
remote:
  outputs/phase5a_sim_demo/online_interest_realtime_final_20260723/

local report:
  outputs/phase5a_sim_demo/online_interest_realtime_final_20260723/report/online_interest_exploration.html
  outputs/phase5a_sim_demo/online_interest_realtime_final_20260723/report/online_interest_exploration.gif

audit:
  online_trace.jsonl
  online_summary.json
  online_tracks.json
  online_object_memory.json
  online_bev_state.npz
```

当前不足与下一步：

```text
1. GroundingDINO-tiny 的 box-depth centroid 抖动仍产生较多碎片轨迹；
   下一步应加入 SAM mask、类别相关 association gate 和 track consolidation。
2. 当前 frontier 运动是 observed-BEV reactive steering；
   下一步应在已知 free BEV 上加入 A*/fast-marching 局部路径，并记录 blocked-frontier blacklist。
3. 当前以 240-step budget 停止，尚未形成稳定的 interest-exhaustion coverage criterion。
4. 当前精确 depth/pose 来自 Habitat；RGB-only 主线仍需由 VGGT/DUST3R 几何替换实验闭合。
5. cup track 的 oracle 对照必须保持 post-hoc、独立进程，不能反馈给在线 policy。
6. 当前 negative evidence 仅用轨迹质心、局部深度窗口与稳定视角门槛判断；
   已证明代码路径会运行，尚未通过对象尺度遮挡模型和专门诊断集验证准确性。
7. smooth evidence weight 当前使用平移与 yaw；pitch 由可见性判定吸收，
   但尚未作为独立运动量进入权重函数。
```

> **最后更新时间:** 2026-07-23
