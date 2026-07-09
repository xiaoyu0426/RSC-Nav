# M2.5: Open-Vocabulary Grounding Adapter
*(RGB / RGB-D -> Semantic Candidates -> BEV Bridge -> RSC Memory)*

## Goal

补齐 M2 bridge 前端缺失的语义标定层，并持续维护本阶段执行文档：

```text
RGB / RGB-D frames
-> open-vocabulary grounding candidates
-> depth + pose 3D association
-> NLMap-style object inventory
-> M2 BEV bridge
-> RSC object memory
-> Phase 3 landmark retrieval
-> Habitat oracle validation report
```

一句话目标：不再只依赖手写或半真实 object inventory，而是让真实或仿真的第一视角观测可以产生同一套 semantic candidates，并用 Habitat semantic oracle 检验通路是否正确。

## 阶段边界

M2.5 做：

- 定义 grounding candidate schema。
- 实现 Habitat semantic oracle backend，作为可复跑 gold / upper-bound。
- 支持 external JSON backend，后续接 OWLv2、GroundingDINO + SAM、ViLD / CLIP。
- 将 candidates 接入现有 M2 bridge。
- 生成 `m25_grounding_report.html`。
- 与 Habitat oracle object memory 做 precision / recall / F1 / centroid error 对照。

M2.5 暂不做：

- 训练 open-vocabulary detector。
- 强制安装大型模型权重。
- VGGT / DUST3R RGB-only geometry replacement。
- Phase 5 API planner。

## 当前执行计划

```text
M2.5A: candidate schema
M2.5B: Habitat oracle backend
M2.5C: external-json backend contract
M2.5D: run M2 bridge + Phase 3 retrieval
M2.5E: oracle validation report
M2.5F: later plug in OWLv2 or GroundingDINO+SAM predictions
```

## Grounding Backend 对照策略

根据 semantic BEV / semantic map 调研，论文里更重要的不是固定配色，而是清楚地区分内部表示：

```text
occupancy / explored geometry channels
+ per-class semantic evidence / confidence channels
+ object inventory / landmark graph
+ visualization PNG
```

因此 M2.5 后续不应只比较“哪张彩色图更好看”，而应比较不同 grounding backend 注入同一 BEV bridge 后的稳定性。

当前采用三层对照：

```text
1. Habitat semantic oracle
   role: gold / upper-bound
   use: 判断 BEV projection 和 object memory bridge 是否正确

2. OWLv2
   role: lightweight baseline
   use: 低部署成本、快速复跑、作为论文中的轻量开放词表 grounding baseline
   limitation: box evidence 粗糙，容易把背景带入对象投影；小物体和门/墙类容易不稳定

3. GroundingDINO + SAM/SAM2
   role: high-quality semantic grounding branch
   use: 先检测/ground 文本目标，再用 mask 约束投影像素，生成更密集、更干净的 semantic evidence
   expected benefit: 降低 box crop 背景污染，提升 semantic BEV 的边界、稳定性和 oracle 对照 F1
```

统一评估入口保持不变：

```text
grounding_candidates.json
-> M2 bridge
-> semantic BEV / RSC object memory
-> Phase 3 retrieval
-> Habitat oracle validation
```

高质量分支新增输出应尽量沿用同一 schema，但把 `mask_backend` 从 `box` 扩展为 `sam` / `sam2`，并在 `raw` 中保存：

```text
detector_backend: grounding_dino
segmenter_backend: sam or sam2
text_prompt
bbox
mask_path or compressed mask reference
projected_mask_points
depth_valid_ratio
view_id
```

验收时重点看：

- object-level precision / recall / F1。
- centroid error。
- semantic BEV cell-level precision / recall / F1。
- repeated coverage-loop 下的 confidence 稳定性。
- false positive 是否被 memory update / confidence saturation 抑制。
- 与 OWLv2 baseline 相比，是否更少把墙/空气/背景投成目标对象。

## 进度日志

### 2026-07-02: Goal accepted

Active goal:

```text
实现 M2.5 Open-Vocabulary Grounding Adapter + Habitat Oracle Validation：
补齐 RGB/RGB-D 到 semantic candidates 的 grounding 前端接口，接入现有 M2 bridge，
并用 Habitat oracle 做可复跑的 gold 对照验证；同时维护阶段性执行文档。
```

设计判断：

- M2 已完成的是 `semantic candidates / pointcloud -> semantic BEV -> RSC memory -> retrieval`。
- v5 主线仍需要 `RGB / RGB-D -> semantic candidates`。
- 先用 Habitat semantic oracle 做 upper-bound / gold validation，避免大型 detector 依赖阻塞主链路。
- 后续真实模型只要导出同一 schema，即可用 `--backend external-json` 进入同一评估。

### 2026-07-02: Adapter implemented and Habitat oracle validation passed

新增实现：

- `src/semantic_grounding_adapter.py`
  - `GroundingCandidate` schema。
  - Habitat object memory -> grounding candidates。
  - external JSON -> grounding candidates。
  - predicted vs gold candidates 的 label + centroid greedy matching。
- `scripts/m25_grounding_adapter_eval.py`
  - `--backend habitat-oracle`：使用 Habitat semantic oracle memory 生成 candidates。
  - `--backend external-json`：接入未来 OWLv2 / GroundingDINO + SAM / ViLD / CLIP 的导出结果。
  - 自动调用 M2 bridge，生成 BEV / semantic BEV / RSC memory / Phase 3 retrieval。
  - 生成 `m25_grounding_report.html`。
- `scripts/phase23_habitat_control_server.py`
  - `save_bev_state()` 额外保存 `oracle_free_mask`、`semantic_state`、`semantic_confidence`，供后续严格 same-grid gold validation 使用。

本地验证命令：

```bash
/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/m25_grounding_adapter_eval.py \
  --backend habitat-oracle \
  --habitat-memory-json outputs/phase213_episode_runs/20260625-170940_phase213_hd384_pitch_sweep_depth_coverage/live_object_memory.json \
  --gold-memory-json outputs/phase213_episode_runs/20260625-170940_phase213_hd384_pitch_sweep_depth_coverage/live_object_memory.json \
  --metrics-json outputs/phase213_episode_runs/20260625-170940_phase213_hd384_pitch_sweep_depth_coverage/metrics.json \
  --image-dir outputs/phase213_episode_runs/20260625-170940_phase213_hd384_pitch_sweep_depth_coverage/images \
  --out-dir outputs/m25_open_vocab_grounding_adapter/habitat_oracle_validation_20260702 \
  --context-id habitat_mp3d_A_oracle \
  --scene-id habitat_mp3d_A_oracle \
  --horizontal-axes xz \
  --resolution-m 0.10 \
  --padding-m 1.0 \
  --semantic-radius-m 0.25
```

验证结果：

```text
status: passed
backend: habitat-oracle
num_candidates: 34
validation precision / recall / F1: 1.0 / 1.0 / 1.0
mean_centroid_error_m: 0.0
bridge status: passed
bridge num_objects: 34
bridge num_labels: 4
Phase 3 retrieval: passed, 4/4 checks
```

输出：

```text
outputs/m25_open_vocab_grounding_adapter/habitat_oracle_validation_20260702/
  grounding_candidates.json
  grounding_metrics.json
  m25_grounding_report.html
  bridge/
    bridge_report.html
    occupancy_bev.png
    semantic_bev.png
    rsc_memory_init.json
    phase3_retrieval/retrieval_report.html
```

环境检查：

```text
Local Codex runtime:
  numpy / PIL available
  torch / transformers not available

Remote /workspace/yujiexiao:
  rscnav-habitat22:
    habitat_sim / habitat available
    torch / transformers unavailable
  system python:
    torch / transformers / opencv / detectron2 / clip / open_clip available
    habitat unavailable
```

因此后续真实 grounding 推荐拆成两步：

```text
system python:
  RGB frames -> OWLv2 / GroundingDINO+SAM predictions -> grounding_candidates.json

rscnav-habitat22 or local bridge runtime:
  grounding_candidates.json + Habitat oracle gold -> M2.5 validation -> M2 bridge
```

当前结论：

```text
M2.5 adapter contract is implemented.
Habitat oracle upper-bound validation is passed.
Real open-vocabulary detector integration is not yet claimed complete; it should target the external-json schema next.
```

### 2026-07-02: True OWLv2 RGB-D grounding full-chain validation passed

本次按完整目标补跑真实 open-vocabulary detector，而不是继续使用 Habitat oracle adapter 冒充 grounding。

链路：

```text
Habitat RGB-D frames
-> OWLv2 object detection
-> box mask crop (SAM/SAM2 optional, not used in this run)
-> depth + sensor pose projection to 3D
-> object inventory / grounding_candidates.json
-> M2 bridge
-> semantic BEV / RSC object memory
-> Phase 3 landmark retrieval
-> Habitat semantic oracle comparison
```

新增实现：

- `scripts/m25_habitat_rgbd_export.py`
  - 从 Habitat 导出 RGB、depth、semantic oracle、sensor pose / rotation。
  - 同步保存 `frames_metadata.json`、`habitat_oracle_object_memory.json`、`bev_state.npz`。
- `scripts/m25_owlv2_grounding_export.py`
  - 使用 HuggingFace OWLv2 权重对 RGB 帧做真实开放词表检测。
  - 使用检测框内 depth crop + sensor pose 将候选投影到 3D。
  - 合并多帧检测为 object inventory。
  - 输出 detection overlay、`detections.json`、`projection_debug.json`、`grounding_candidates.json`。
- `scripts/m25_grounding_adapter_eval.py`
  - `--overlay-dir` 展示 detector overlay。
  - `--grounding-export-report` 链接 OWLv2 导出报告。
  - external-json 报告文案改为真实模型导出后的链路验证说明。

远端/本地执行说明：

```text
Habitat RGB-D export:
  remote rscnav-habitat22

OWLv2 inference:
  remote system python + local-copied google/owlv2-base-patch16-ensemble

M2 bridge / Phase 3 / oracle validation:
  local Codex Python runtime
```

验证结果：

```text
frames: 10
OWLv2 projected detections: 62
projection debug items: 63 projected/rejected = 62/1
object candidates: 13
mask backend: box
SAM/SAM2: not used in this run

oracle comparison:
  predicted_count: 13
  gold_count: 6
  true_positive: 3
  false_positive: 10
  false_negative: 3
  precision: 0.231
  recall: 0.500
  F1: 0.316
  mean_centroid_error_m: 0.172

M2 bridge:
  status: passed
  num_objects: 13
  num_labels: 4
  semantic_cells: 297

Phase 3 retrieval:
  status: passed
  checks: 4/4
```

输出：

```text
outputs/m25_open_vocab_grounding/full_chain_rgbd_20260702/
  frames_metadata.json
  habitat_oracle_object_memory.json
  frames/

outputs/m25_open_vocab_grounding/full_chain_owlv2_20260702/
  grounding_candidates.json
  detections.json
  projection_debug.json
  owlv2_grounding_report.html
  overlays/

outputs/m25_open_vocab_grounding/full_chain_validation_20260702/
  m25_grounding_report.html
  grounding_metrics.json
  bridge/bridge_report.html
  bridge/semantic_bev.png
  bridge/occupancy_bev.png
  bridge/phase3_retrieval/retrieval_report.html
```

当前结论：

```text
完整 M2.5 pipeline 已经打通：
真实 OWLv2 detection 可以从 Habitat RGB-D frames 生成 object inventory，
并进入 M2 bridge / RSC memory / Phase 3 retrieval / oracle validation。

但检测质量还不高：
当前 F1=0.316，false positive 较多，说明这次验证的是“链路可行”和“可对照评估”，
还不能宣称 OWLv2 grounding 效果已经达标。

后续优化方向：
1. 接入 GroundingDINO + SAM/SAM2 或 OWLv2 + SAM/SAM2，减少 box crop 噪声。
2. 增加更多视角/轨迹帧，提升 object track 合并稳定性。
3. 在 oracle label set 与 open-vocabulary prompt set 之间做同义词和类别映射。
4. 将检测置信度、几何一致性、多视角复现次数共同用于 object inventory 置信度。
```

### 2026-07-03: RGB-to-semantic-BEV MVP evidence visualization

补充可视化目标：

```text
RGB frame
-> OWLv2 detection overlay
-> traditional BEV from RGB-D/depth+pose
-> object inventory projection evidence
-> Phase2-style semantic BEV MVP
```

这次图的重点不是只展示 object inventory，也不是把 OWLv2 candidate 贴到 Habitat oracle semantic BEV 上，而是展示当前 MVP 的证据链：

```text
geometry BEV:
  由 depth + pose 生成，作为导航底座

object projection evidence:
  由 OWLv2 box + depth + pose 生成 object-centric BEV evidence

semantic BEV MVP:
  将 geometry BEV 与 object projection evidence 合并，得到 Phase2 风格的语义记忆图
```

新增脚本：

```text
scripts/m25_make_semantic_bev_mvp_evidence.py
```

输出：

```text
outputs/m25_open_vocab_grounding/full_chain_validation_20260702/rgb_to_semantic_bev_mvp/
  rgb_to_semantic_bev_mvp.html
  rgb_to_semantic_bev_mvp_pipeline.png
  traditional_bev_phase2_style.png
  object_inventory_projection_evidence.png
  semantic_bev_from_owlv2_mvp.png
  rgb_to_semantic_bev_mvp_metadata.json
```

可视化结论：

```text
当前系统已经不仅能生成 object inventory，
也能把 OWLv2 object evidence 投影到传统 BEV 上，
并合并成 Phase2 风格的 semantic BEV MVP。
```

限制：

```text
该 semantic BEV MVP 仍是 object-centric evidence map。
当前 mask backend=box，不是 SAM/SAM2 dense mask。
因此它能证明 RGB/OWLv2 -> semantic BEV 的链路形式成立，
但不能替代后续 dense open-vocabulary segmentation / SAM mask 版本。
```

### 2026-07-03: Whole-environment coverage-loop evidence visualization

为避免 10 帧局部走廊图不足以表达完整环境，M2.5 exporter 复用了 Phase2 的 coverage-loop 遍历策略：

```text
navmesh sampling
-> perimeter waypoint selection
-> shortest-path loop
-> route resampling
-> yaw scan every N route steps
-> full RGB-D / pose / BEV state export
```

本次运行：

```text
trajectory mode: coverage-loop
frames: 96
route steps: 72
yaw scan: every 12 route steps, 4 turns each
BEV grid: 574 x 574 at 0.05 m/cell
Habitat oracle object memory: 38 items
OWLv2 projected detections: 697
OWLv2 object candidates: 119
```

验证结果：

```text
M2 bridge: passed
Phase 3 retrieval: passed
oracle comparison:
  predicted_count: 119
  gold_count: 38
  true_positive: 21
  false_positive: 98
  false_negative: 17
  precision: 0.176
  recall: 0.553
  F1: 0.268
  mean_centroid_error_m: 0.320
```

输出：

```text
outputs/m25_open_vocab_grounding/full_env_rgbd_20260703/
outputs/m25_open_vocab_grounding/full_env_owlv2_20260703/
outputs/m25_open_vocab_grounding/full_env_validation_20260703/
  m25_grounding_report.html
  rgb_to_semantic_bev_mvp/
    rgb_to_semantic_bev_mvp.html
    rgb_to_semantic_bev_mvp_pipeline.png
    traditional_bev_phase2_style.png
    object_inventory_projection_evidence.png
    semantic_bev_from_owlv2_mvp.png
```

结论：

```text
整环境版图已经能表达完整 MVP 链路：
RGB/OWLv2 observation -> traditional BEV + object projection evidence -> Phase2-style semantic BEV MVP。

该图比局部 10 帧版本更符合 Phase2 汇总报告的地图形态；
同时也暴露了 OWLv2 box-only 的 false positive 问题。
后续若接入 GroundingDINO + SAM/SAM2 或多视角一致性过滤，应优先在这张整环境图上对比改进。
```

### 2026-07-03: Current mainline for M2.5-GDINO

当前主线已从“证明 OWLv2 能输出 object inventory”推进到：

```text
引入 GroundingDINO 对照分支，
并优化 RGB 输入链路到论文级对齐的 semantic BEV 表示。
```

核心判断：

```text
当前链路已经通了，但效果很差。
因此后续主目标不是继续证明 pipeline connected，
而是建立可度量的 quality improvement loop。
```

执行原则：

```text
OWLv2:
  保留为 lightweight baseline。
  继续使用 box evidence，提供低成本、快速复跑的对照。

GroundingDINO + SAM/SAM2:
  新增为 high-quality semantic grounding branch。
  目标是从 box-level evidence 过渡到 mask-level evidence，减少背景污染和误投影。

Habitat semantic oracle:
  保留为 gold / upper-bound。
  用于区分 grounding 错误、projection 错误和 memory update 错误。
```

本阶段不改变 M2 bridge 和 Phase 3 retrieval 的数据契约。GroundingDINO 分支必须导出与 OWLv2 相同的 `grounding_candidates.json`，新增字段只作为可选信息：

```text
source: grounding_dino_sam
mask_backend: sam or sam2
bbox
mask_ref
projected_mask_points
depth_valid_ratio
source_view_ids
raw.detector_score
raw.mask_area
raw.prompt
```

验收目标：

```text
Habitat RGB-D frames
-> GroundingDINO text grounding
-> optional SAM/SAM2 mask
-> depth + pose mask projection
-> grounding_candidates.json
-> M2 bridge
-> semantic BEV / RSC object memory
-> Phase 3 retrieval
-> Habitat oracle comparison
-> side-by-side OWLv2 vs GroundingDINO(+SAM) report
```

质量优化指标：

```text
object-level:
  precision / recall / F1
  false positive count
  centroid error
  repeated-view merge rate

semantic BEV:
  semantic evidence cells
  cell-level precision / recall / F1 when oracle semantic BEV is available
  evidence confidence distribution
  false evidence near wall / background

memory:
  active / stale / missing object count
  confidence stability under repeated traversal
  negative evidence 是否只在 expected-visible miss 时触发
```

优化优先级：

```text
P0: detector quality
  OWLv2 box baseline -> GroundingDINO box branch

P1: projection quality
  box crop depth -> mask / foreground depth projection

P2: temporal consistency
  单帧 candidate -> 多帧一致性合并
  低置信单帧预测不直接污染 semantic BEV

P3: semantic BEV rendering
  彩色 PNG 只做可视化；
  内部以 semantic evidence/confidence channel 为准
```

论文级可视化目标：

```text
RGB frame
detector / mask overlay
traditional BEV
object / mask projection evidence
semantic BEV evidence channels
semantic evidence confidence
oracle comparison metrics
```

环境状态：

```text
remote system python:
  torch / CUDA / transformers / cv2 available
  groundingdino / segment_anything / sam2 not installed yet

implementation note:
  先实现 adapter 与 report contract。
  模型安装和权重下载作为 M2.5-GDINO 的显式执行步骤，不混入既有 OWLv2 baseline。
```

### 2026-07-03: GroundingDINO box branch smoke and first quality improvement

实现：

```text
scripts/m25_groundingdino_export.py
```

说明：

- 使用 transformers 原生 `GroundingDinoProcessor` / `GroundingDinoForObjectDetection`。
- 第一版为 `mask_backend=box`，SAM/SAM2 mask 字段已在 schema 中预留。
- 输出与 OWLv2 相同的 `grounding_candidates.json`，可直接进入 M2 bridge / Phase 3 retrieval。
- 模型权重下载到：`downloads/hf_models/grounding-dino-tiny/model.safetensors`。
- 代理仅在 HuggingFace 权重下载时临时开启；推理、validation、bridge 默认 `unset http_proxy https_proxy all_proxy`。

2-frame smoke：

```text
GroundingDINO tiny
frames: 2
projected detections: 5
object candidates: 4
M2 bridge / Phase 3 retrieval contract: runnable
oracle F1: 0.0
```

判断：2 帧 smoke 只用于验证接口，不用于质量结论。

24-frame stride-4 baseline：

```text
GroundingDINO tiny
frames: 24, stride: 4
box_threshold: 0.20
text_threshold: 0.20
projected detections: 190
object candidates: 85

oracle comparison:
  TP / FP / FN: 18 / 67 / 20
  precision: 0.212
  recall: 0.474
  F1: 0.293
  mean centroid error: 0.281 m

M2 bridge: passed
Phase 3 retrieval: passed
```

与 OWLv2 full-env baseline 对比：

```text
OWLv2 96 frames:
  candidates: 119
  TP / FP / FN: 21 / 98 / 17
  precision / recall / F1: 0.176 / 0.553 / 0.268

GroundingDINO 24 frames:
  candidates: 85
  TP / FP / FN: 18 / 67 / 20
  precision / recall / F1: 0.212 / 0.474 / 0.293
```

初步结论：

```text
GroundingDINO 即使只跑 24 帧，也已经略优于 OWLv2 96 帧的 F1，
并且 precision 更高、centroid error 更低。
但 FP 仍多，说明 detector 本身还不够，必须加入多视角一致性和 mask-level projection。
```

多视角一致性过滤实验：

```text
raw:
  n=85, TP/FP/FN=18/67/20, P/R/F1=0.212/0.474/0.293, err=0.281

min_views >= 2:
  n=48, TP/FP/FN=16/32/22, P/R/F1=0.333/0.421/0.372, err=0.270

min_views >= 3:
  n=24, TP/FP/FN=10/14/28, P/R/F1=0.417/0.263/0.323, err=0.232

confidence >= 0.30:
  n=46, TP/FP/FN=15/31/23, P/R/F1=0.326/0.395/0.357, err=0.218

min_views >= 2 and confidence >= 0.30:
  n=33, TP/FP/FN=14/19/24, P/R/F1=0.424/0.368/0.394, err=0.210
```

当前最佳：

```text
GroundingDINO tiny + multi-view consistency filter
condition: source views >= 2 and confidence >= 0.30
F1: 0.394
FP: 98 (OWLv2 baseline) -> 19
```

输出：

```text
outputs/m25_open_vocab_grounding/full_env_groundingdino_t020_stride4_20260703/
outputs/m25_open_vocab_grounding/full_env_groundingdino_t020_stride4_validation_20260703/
outputs/m25_open_vocab_grounding/full_env_groundingdino_t020_stride4_views2_conf030_validation_20260703/
outputs/m25_open_vocab_grounding/grounding_backend_quality_comparison_20260703.html
```

下一步优化：

```text
1. 跑完整 96-frame GroundingDINO coverage-loop。
2. 做 prompt / threshold sweep，找到 precision-recall tradeoff。
3. 将多视角一致性过滤正式纳入 exporter 或 post-filter script。
4. 接 GroundingDINO + SAM/SAM2 mask-level projection，减少 box 背景污染。
5. 在统一 report 中比较 OWLv2 / GroundingDINO raw / GroundingDINO filtered / GroundingDINO+SAM。
```

### 2026-07-03: Full 96-frame GroundingDINO and formal candidate filtering

新增：

```text
scripts/m25_filter_grounding_candidates.py
```

作用：

```text
grounding_candidates.json
-> min_views / min_confidence / label allowlist / max-per-label filtering
-> filtered grounding_candidates.json
-> M2 bridge / semantic BEV / Phase 3 retrieval
```

96-frame raw run：

```text
GroundingDINO tiny
frames: 96
box_threshold: 0.20
text_threshold: 0.20
projected detections: 767
object candidates: 176

oracle comparison:
  TP / FP / FN: 23 / 153 / 15
  precision: 0.131
  recall: 0.605
  F1: 0.215
  mean centroid error: 0.282 m
```

判断：

```text
完整 coverage 提升了 recall，但 raw evidence 直接累积会显著放大 false positive。
所以“走更多帧”本身不是解法；必须配合 temporal / multi-view consistency。
```

96-frame filtering sweep：

```text
min_views >= 2:
  n=121, TP/FP/FN=22/99/16, P/R/F1=0.182/0.579/0.277, err=0.279

min_views >= 3:
  n=91, TP/FP/FN=19/72/19, P/R/F1=0.209/0.500/0.295, err=0.288

confidence >= 0.30:
  n=89, TP/FP/FN=22/67/16, P/R/F1=0.247/0.579/0.346, err=0.280

min_views >= 2 and confidence >= 0.30:
  n=73, TP/FP/FN=21/52/17, P/R/F1=0.288/0.553/0.378, err=0.277

min_views >= 3 and confidence >= 0.30:
  n=62, TP/FP/FN=19/43/19, P/R/F1=0.306/0.500/0.380, err=0.288

min_views >= 2 and confidence >= 0.35:
  n=42, TP/FP/FN=18/24/20, P/R/F1=0.429/0.474/0.450, err=0.277
```

当前最佳：

```text
GroundingDINO full96 + multi-view/confidence filter
condition: source views >= 2 and confidence >= 0.35

Compared with OWLv2 full-env baseline:
  F1: 0.268 -> 0.450
  FP: 98 -> 24
  precision: 0.176 -> 0.429
```

输出：

```text
outputs/m25_open_vocab_grounding/full_env_groundingdino_t020_full96_20260703/
outputs/m25_open_vocab_grounding/full_env_groundingdino_t020_full96_validation_20260703/
outputs/m25_open_vocab_grounding/full_env_groundingdino_t020_full96_views2_conf035_validation_20260703/
outputs/m25_open_vocab_grounding/grounding_backend_quality_comparison_20260703.html
```

阶段结论：

```text
当前的主要质量瓶颈已定位为 false semantic evidence。
GroundingDINO + full coverage + multi-view/confidence filtering 已经证明能显著提升 semantic BEV 输入质量。
下一步应转向：
  1. prompt / threshold systematic sweep
  2. mask-level GroundingDINO + SAM/SAM2 projection
  3. 将 filtering/evidence weighting 接入 memory update，而不只是离线 JSON 后处理
```

SAM/SAM2 环境检查：

```text
remote system python:
  transformers SamModel: available
  transformers SamProcessor: available
  transformers Sam2Model / Sam2Processor: unavailable
  segment_anything package: unavailable
  sam2 package: unavailable

local SAM weights:
  not found under downloads / HuggingFace cache
```

因此下一步 mask-level MVP 建议：

```text
Use transformers SAM:
  facebook/sam-vit-base

GroundingDINO boxes
-> SAM box-prompt mask
-> mask pixels + depth + pose projection
-> same grounding_candidates schema
-> compare against GroundingDINO box branch
```

代理原则：

```text
只在下载 facebook/sam-vit-base 权重时临时开启海外代理；
下载结束立即 unset；
推理、bridge、validation 默认不使用代理。
```

### 2026-07-03: GroundingDINO + SAM mask projection MVP

新增实现：

```text
scripts/m25_groundingdino_export.py
  --mask-backend sam
  --sam-model-id downloads/hf_models/sam-vit-base
```

链路：

```text
GroundingDINO box
-> SAM box-prompt mask
-> mask pixels + depth + pose
-> 3D centroid / object candidate
-> M2 bridge / semantic BEV / Phase 3 retrieval
```

权重：

```text
facebook/sam-vit-base
downloaded to downloads/hf_models/sam-vit-base
```

2-frame smoke：

```text
mask_backend: sam
frames: 2
projected detections: 13
object candidates: 8
result: exporter / SAM projection path runnable
```

24-frame stride-4 SAM raw：

```text
object candidates: 81
TP / FP / FN: 17 / 64 / 21
precision / recall / F1: 0.210 / 0.447 / 0.286
mean centroid error: 0.278 m
M2 bridge: passed
Phase 3 retrieval: passed
```

24-frame SAM filtering sweep：

```text
min_views >= 2:
  n=49, TP/FP/FN=16/33/22, P/R/F1=0.327/0.421/0.368, err=0.266

confidence >= 0.30:
  n=48, TP/FP/FN=15/33/23, P/R/F1=0.312/0.395/0.349, err=0.222

min_views >= 2 and confidence >= 0.30:
  n=35, TP/FP/FN=14/21/24, P/R/F1=0.400/0.368/0.384, err=0.204

min_views >= 2 and confidence >= 0.35:
  n=19, TP/FP/FN=10/9/28, P/R/F1=0.526/0.263/0.351, err=0.185
```

判断：

```text
SAM mask projection 已接入并验证可运行。
当前 SAM 分支没有超过 box full96 filtered 的 F1=0.450；
但在相同 24-frame 条件下，SAM filtered 的 centroid error 更低：
  box views2_conf030 err=0.210
  SAM views2_conf030 err=0.204
并且更严格过滤下 precision 可到 0.526。
```

结论：

```text
SAM 不是“接上就自动变好”。
它更像提升几何定位/减少背景污染的潜力分支；
要成为最终论文级 semantic BEV，需要继续做：
  prompt tuning
  mask quality filtering
  mask depth validity filtering
  semantic evidence weighting
  full96 SAM run or selective SAM-on-filtered-detections
```

输出：

```text
outputs/m25_open_vocab_grounding/full_env_groundingdino_sam_t020_stride4_20260703/
outputs/m25_open_vocab_grounding/full_env_groundingdino_sam_t020_stride4_validation_20260703/
outputs/m25_open_vocab_grounding/full_env_groundingdino_sam_t020_stride4_views2_conf030_validation_20260703/
outputs/m25_open_vocab_grounding/grounding_backend_quality_comparison_20260703.html
```

### 2026-07-03: SAM 调参复盘

修复：

```text
scripts/m25_groundingdino_export.py
  --sam-min-iou
  --sam-max-mask-area-ratio
  --sam-min-mask-area-px

scripts/m25_filter_grounding_candidates.py
  --min-sam-iou
  --min-depth-valid-ratio
```

发现一个实现问题：

```text
之前 exporter 已经计算了 SAM reject_reason，
但没有在投影前真正跳过被 reject 的 mask。
因此 area / iou gate 一开始看似跑了，实际没有影响指标。
已修复为：若 mask_info.reject_reason 存在，直接记录 projection_debug 并跳过该 detection。
```

SAM mask gate 快速实验：

```text
24-frame stride-4, SAM raw:
  n=81, TP/FP/FN=17/64/21, P/R/F1=0.210/0.447/0.286, err=0.278

area<=0.50, iou>=0.85:
  raw: n=82, TP/FP/FN=17/65/21, P/R/F1=0.207/0.447/0.283, err=0.275
  views>=2, conf>=0.30: n=34, TP/FP/FN=13/21/25, P/R/F1=0.382/0.342/0.361, err=0.205

area<=0.35, iou>=0.85:
  raw: n=81, TP/FP/FN=16/65/22, P/R/F1=0.198/0.421/0.269, err=0.249
  views>=2, conf>=0.30: n=33, TP/FP/FN=13/20/25, P/R/F1=0.394/0.342/0.366, err=0.205
```

结论：

```text
简单收紧 SAM mask area / iou gate 没有提升 F1；
过严 gate 会删除有效对象。
SAM 的主要收益更偏向 centroid error / mask-level 定位，而不是自动提升对象库存质量。
```

96-frame SAM full run：

```text
SAM raw:
  n=176, TP/FP/FN=24/152/14, P/R/F1=0.136/0.632/0.224, err=0.276

SAM views>=2, conf>=0.30:
  n=73, TP/FP/FN=20/53/18, P/R/F1=0.274/0.526/0.360, err=0.260

SAM views>=2, conf>=0.35:
  n=38, TP/FP/FN=15/23/23, P/R/F1=0.395/0.395/0.395, err=0.277

SAM views>=3, conf>=0.30:
  n=63, TP/FP/FN=19/44/19, P/R/F1=0.302/0.500/0.376, err=0.257
```

与当前最优主分支对比：

```text
GroundingDINO box 96 + views>=2 + conf>=0.35:
  n=42, TP/FP/FN=18/24/20, P/R/F1=0.429/0.474/0.450, err=0.277

GroundingDINO + SAM 96 + views>=2 + conf>=0.35:
  n=38, TP/FP/FN=15/23/23, P/R/F1=0.395/0.395/0.395, err=0.277
```

当前阶段判断：

```text
1. SAM 已经接入 Habitat RGB-D -> mask projection -> object inventory -> M2 bridge -> semantic BEV -> Phase3 retrieval。
2. 当前默认推荐仍是 GroundingDINO box + full coverage + multi-view/confidence filter。
3. SAM 保留为 optional high-quality semantic grounding branch / localization refinement。
4. 下一步优化重点不是单纯启用 SAM，而是：
   - prompt / label vocabulary tuning
   - selective SAM-on-filtered-detections
   - semantic evidence weighting
   - detector confidence + source view count + depth validity 的联合打分
   - 再回到论文级 semantic BEV 表示进行可视化和 oracle 对照
```

新增输出：

```text
outputs/m25_open_vocab_grounding/full_env_groundingdino_sam_t020_full96_20260703/
outputs/m25_open_vocab_grounding/full_env_groundingdino_sam_t020_full96_20260703_validation_20260703/
outputs/m25_open_vocab_grounding/full_env_groundingdino_sam_t020_full96_20260703_views2_conf035_validation_20260703/
outputs/m25_open_vocab_grounding/full_env_groundingdino_sam_t020_full96_20260703_views2_conf035_validation_20260703/rgb_to_semantic_bev_mvp/rgb_to_semantic_bev_mvp.html
outputs/m25_open_vocab_grounding/grounding_backend_quality_comparison_20260703.html
```

### 2026-07-03: 阶段出口判断

本阶段当前按 MVP 闭环通过，而不是按最终论文级效果通过。

已满足的通过条件：

```text
1. RGB / RGB-D observation 能进入 open-vocabulary grounding backend。
2. OWLv2 / GroundingDINO / GroundingDINO+SAM 三类分支均已落到统一 grounding_candidates schema。
3. grounding_candidates 能通过 depth + pose 投影形成 object inventory。
4. object inventory 能进入 M2 bridge，生成 occupancy BEV / semantic BEV / RSC object memory。
5. RSC object memory 能进入 Phase 3 landmark retrieval。
6. 全链路能与 Habitat semantic oracle 做 TP / FP / FN / precision / recall / F1 / centroid error 对照。
7. 已形成可浏览的 paper-aligned semantic BEV MVP 可视化页面。
```

当前最佳默认分支：

```text
GroundingDINO box
-> 96-frame coverage-loop
-> source views >= 2
-> confidence >= 0.35
-> M2 bridge / RSC memory / Phase 3 retrieval

metric:
  n=42
  TP/FP/FN=18/24/20
  P/R/F1=0.429/0.474/0.450
  centroid error=0.277 m
```

为什么可以暂时通过：

```text
本阶段已经证明 RGB semantic evidence 可以稳定进入 RSC-Nav 的 BEV memory / object memory / retrieval 后端。
瓶颈已从“链路是否可行”转移到“语义表示和证据融合是否足够论文级”。
继续在 SAM mask gate 上局部调参，短期收益低于先固定表示规范、对齐论文常见 semantic map 形式。
```

保留问题：

```text
1. 当前 semantic BEV 仍偏 object-centric evidence，不是 dense per-cell semantic segmentation map。
2. open-vocabulary grounding FP 仍偏多，需要后续 prompt / label vocabulary / evidence weighting / multi-view fusion 优化。
3. SAM 当前是 optional refinement，不作为默认主分支。
4. RGB-only geometry 的 VGGT / DUST3R 分支已跑通 MVP，但还没有成为默认几何来源。
5. 语义地图内部表示与论文常见表示需要进一步系统对齐。
```

下一步承接：

```text
进入 M3.5 Semantic Representation Alignment。

目标不是继续换检测器，而是定义并审计 RSC-Nav 的语义空间记忆表示：
  G: geometry BEV
  S: semantic evidence BEV
  O: object memory / object inventory
  L: landmark / topology graph

随后再进入 Phase 5 API planner / waypoint decision 时，模型输入就不再摇摆于彩色图、对象列表、语义栅格或拓扑图之间。
```
