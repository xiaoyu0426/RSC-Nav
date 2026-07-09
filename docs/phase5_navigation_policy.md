# Phase 5: Navigation Policy

## 阶段定位

Phase 5 的目标是把 memory retrieval 结果注入导航决策，验证 RSC memory 是否能改善 waypoint / stop decision。

## 当前状态

状态：Phase5A 最小闭环 MVP 已开始并通过 deterministic teacher 验收。

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
  first-person frames: 75
  video: water_then_owner_bed_first_person.mp4

artifact:
  outputs/phase5a_sim_demo/water_then_owner_bed_20260704/demo_report.html
```

本次边界：

```text
1. 本 demo 已经不是硬编码 table；API planner 读取 semantic map 后自行选择当前最合理的 water-place candidate。
2. 由于当前语义 map 没有真实 water affordance 类别，planner 的 table 选择属于“不确定条件下的最佳可用搜索点”。
3. 当前 demo 展示 plan -> navmesh execution，不包含旧位置没水后的 negative-evidence update / replan。
4. 下一步应加入 bottle/cup/sink/fridge/kitchen/bathroom 等语义候选，或构造同环境对象缺失实验，形成完整 plan -> verify -> update -> replan。
```

> **最后更新时间:** 2026-07-04
