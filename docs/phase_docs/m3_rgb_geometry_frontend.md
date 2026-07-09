# M3: RGB Geometry Frontend
*(RGB Sequence -> Depth / Pose / Point Cloud -> BEV Bridge)*

## 阶段定位

M3 对应 VGGT-NLMap Semantic BEV Bridge 的 Stage C。

它的目标不是重写 RSC-Nav 主线，也不是立即替代所有 Habitat oracle，而是在阶段性实验中验证：

```text
first-person RGB sequence
-> VGGT / DUST3R / MASt3R geometry frontend
-> estimated depth / camera pose / point cloud
-> same BEV / semantic BEV bridge
-> RSC memory and retrieval
```

当前 RSC-Nav 主线仍可继续使用 Habitat RGB-D / oracle pose 来隔离语义记忆和规划变量。M3 是将来脱离 oracle geometry、贴近真实拍摄 RGB 序列的扩展桥。

## 当前背景

M2.5 已经跑通：

```text
Habitat RGB-D frames
-> OWLv2 object detection
-> depth + pose projection
-> object inventory
-> traditional BEV + object projection evidence
-> Phase2-style semantic BEV MVP
-> RSC object memory / Phase 3 retrieval
```

但当前 semantic BEV 仍是 object-centric MVP：

```text
OWLv2 box
-> depth crop
-> 3D centroid / object candidate
-> BEV local evidence radius
```

它能证明 RGB grounding evidence 可以进入 BEV memory，但还不是 Phase2 理想状态下那种整张地图稳定染色的 dense semantic BEV。

## Dense Semantic BEV 的真实要求

理想的 Phase2-style 染色 semantic BEV 应该更接近：

```text
RGB frame
-> open-vocabulary grounding
-> SAM/SAM2 mask or dense semantic proposal
-> depth + pose project mask pixels to 3D
-> per-cell semantic evidence update
-> confidence / freshness / prior-live semantic BEV
```

也就是说，语义证据要从 object centroid 扩展到 mask / pixel / local surface level。

## 主要误差来源

### Semantic Recognition Error

OWLv2 / GroundingDINO / VLM 如果类别判断错误，会把区域染成错误语义。

### Mask / Segmentation Error

当前 box crop 会把背景带入对象证据。SAM/SAM2 mask 能改善边界，但仍可能受遮挡、镜面、透明物体和小物体影响。

### Geometry Error

RGB-only geometry frontend 的 depth / pose / scale 如果不准，会让语义 evidence 投影到错误 BEV cell。多帧累积时，误差会表现为重影、漂移和地图变糊。

## 闭环收敛机制

后续 dense semantic BEV 不能一帧染色即定稿，需要收敛机制：

```text
multi-view evidence accumulation:
  多视角重复观测才增强 confidence

geometry-aware weighting:
  depth uncertainty / pose uncertainty / 视角角度 / 投影距离影响 evidence weight

confidence saturation:
  原地重复看同一视角不能无限刷高置信度

positive / not observable / expected-visible miss:
  只有理论可见却未见时产生 negative evidence

temporal consistency and clustering:
  同一对象跨帧一致才合并；离群检测降权或过滤

context remapping:
  大范围几何/语义不一致时不强行覆盖旧地图，而是创建或切换 context
```

## M3 实施拆解

### M3.0 Data Contract

定义 RGB geometry frontend 的统一输出格式：

```text
frames.json
images/
intrinsics.json or estimated_camera.json
poses_est.json
depth_est/
pointcloud.ply
alignment.json
```

这些文件必须能被现有 BEV bridge 读取，或通过 adapter 转成现有 `frames_metadata.json` 风格。

### M3.1 Oracle Alignment Baseline

先用 Habitat RGB-D / oracle pose 导出同格式数据，作为 VGGT / DUST3R 输出的 gold 对照。

验收：

```text
oracle RGB-D / pose
-> pointcloud
-> traditional BEV
-> semantic projection
```

必须能复现当前 M2.5 的 BEV 结果。

### M3.2 RGB Geometry Model Adapter

优先尝试：

```text
VGGT demo
```

备选：

```text
DUST3R / MASt3R
```

输出至少包括：

```text
estimated camera poses
estimated depth or point maps
point cloud
```

### M3.3 Metric Alignment

解决 RGB-only reconstruction 常见的尺度和坐标系问题。

仿真阶段：

```text
estimated poses / pointcloud
-> Sim(3) alignment to Habitat oracle poses
```

真实拍摄阶段可用：

```text
camera height
known object scale
gravity direction
ICP / visual alignment
```

### M3.4 Pointcloud-to-BEV Bridge

将 estimated geometry 接入当前 DenseBEVMapper：

```text
estimated depth / pose / point cloud
-> occupancy / explored BEV
-> semantic evidence projection
-> object memory
```

### M3.5 Validation

与 Habitat oracle RGB-D pipeline 对齐：

```text
pose:
  ATE / RPE

depth:
  RMSE / scale error

point cloud:
  coverage / Chamfer

BEV:
  free / occupied IoU

object projection:
  centroid error

semantic BEV:
  cell-level precision / recall / F1
```

### M3.6 Semantic Grounding Branch Comparison

M3 不只要问“VGGT 能不能生成 BEV”，还要把 RGB 到 semantic BEV 的误差拆成两个来源：

```text
geometry branch:
  Habitat oracle depth/pose
  vs VGGT estimated depth/pose/pointcloud

semantic grounding branch:
  Habitat semantic oracle
  vs OWLv2 box evidence
  vs GroundingDINO + SAM/SAM2 mask evidence
```

推荐矩阵：

```text
A. oracle geometry + oracle semantic
   role: upper-bound / gold

B. oracle geometry + OWLv2 box
   role: lightweight semantic baseline

C. oracle geometry + GroundingDINO+SAM
   role: high-quality semantic branch; isolate semantic improvement

D. VGGT geometry + OWLv2 box
   role: current M3 smoke; tests RGB-only geometry plus lightweight semantics

E. VGGT geometry + GroundingDINO+SAM
   role: target RGB-only semantic BEV MVP
```

这样可以回答三个问题：

```text
1. semantic BEV 不稳，是 geometry drift 造成的，还是 grounding false positive 造成的？
2. GroundingDINO+SAM 相比 OWLv2 是否减少 box 背景污染和错误投影？
3. VGGT 几何误差进入 semantic BEV 后，RSC memory / retrieval 是否仍能靠多帧 evidence update 保持可用？
```

## 初步验收标准

M3 MVP 算完成的最低标准：

```text
1. 选定一个 RGB sequence。
2. 用 VGGT 或 DUST3R 跑出 depth / pose / point cloud。
3. 导出统一 geometry artifact。
4. 对齐到 Habitat oracle 坐标系。
5. 生成 traditional BEV。
6. 将 M2.5 semantic candidates 投影到该 BEV。
7. 输出对比报告：
   oracle geometry BEV vs RGB-only geometry BEV
   oracle semantic projection vs RGB-only semantic projection
```

## 当前状态

```text
status: 端到端 smoke 已跑通，几何/语义质量仍需优化
current backend:
  VGGT-1B + OWLv2 boxes
current limitation:
  单窗口 VGGT 对 coverage-loop 全环境重建仍不稳定；需要窗口化/重叠片段/更严格的候选过滤。
```

## 执行日志

### 2026-07-03: VGGT-1B smoke run

目标链路：

```text
Habitat coverage-loop RGB sequence
-> VGGT estimated camera pose / depth / point cloud
-> Sim(3) align to Habitat oracle world frame
-> DenseBEVMapper
-> traditional BEV
-> OWLv2 semantic evidence projection
-> semantic BEV MVP
-> M2 bridge / RSC object memory / Phase 3 retrieval
-> oracle RGB-D/pose BEV comparison
```

已完成内容：

- 在开发机下载并使用本地 VGGT-1B 权重：`downloads/hf_models/vggt/model.pt`。
- 新增脚本：`scripts/m3_vggt_geometry_eval.py`。
- 输出报告：
  - `outputs/m3_rgb_geometry_frontend/vggt_habitat_smoke_first16_20260703/m3_vggt_geometry_report.html`
  - `outputs/m3_rgb_geometry_frontend/vggt_habitat_smoke_first16_20260703/rgb_to_semantic_bev_mvp/rgb_to_semantic_bev_mvp.html`
  - `outputs/m3_rgb_geometry_frontend/vggt_habitat_smoke_first16_20260703/bridge/bridge_report.html`

关键结果：

```text
run: first 16 contiguous frames
status: needs_review
num_frames: 16
num_candidates: 42

alignment:
  ATE RMSE: 1.609 m
  ATE mean: 1.098 m
  ATE max: 5.287 m

BEV comparison:
  explored IoU: 0.503
  free IoU: 0.172
  occupied IoU: 0.200

semantic oracle validation:
  precision: 0.048
  recall: 0.053
  F1: 0.050

bridge:
  M2 bridge: passed
  Phase 3 retrieval: passed
```

判断：

- 从“工程链路”角度，M3 MVP 已经打通：VGGT 输出被对齐、进入 BEV、接入 OWLv2 evidence、生成 RSC object memory，并能复用 Phase 3 retrieval。
- 从“论文展示质量”角度，当前还不能作为最终结果：16 帧只覆盖局部区域；VGGT 单窗口对整圈 coverage-loop 的几何一致性仍不足；semantic validation 被 OWLv2 低置信误检和重复候选明显拉低。
- 与第一版 8 帧稀疏抽样相比，连续帧明显改善了几何：ATE RMSE 从约 4.46 m 降到约 1.61 m，explored IoU 从约 0.37 提升到约 0.50。

下一步建议：

- 不再用全局稀疏抽样直接喂 VGGT；改为重叠连续窗口。
- 每个窗口内做局部 VGGT reconstruction，再通过 oracle/similarity alignment 或相邻窗口 overlap 做拼接。
- 对 OWLv2 candidates 增加置信度、空间聚类、类别白名单和重复合并，以降低 semantic BEV 噪声。
- 保留 Habitat RGB-D/oracle pose 作为 M3 gold，用于定位是几何误差还是 semantic grounding 误差。

### 2026-07-03: Visualization aligned with semantic BEV survey

基于 semantic BEV 表示方式调研，已更新 `scripts/m25_make_semantic_bev_mvp_evidence.py` 的可视化输出：

```text
RGB frame
OWLv2 detection overlay
traditional BEV: unknown/free/occupied/explored + trajectory
object inventory projection evidence: TP/FP rings
semantic BEV: geometry substrate + semantic evidence channels
semantic evidence confidence map
```

输出页面：

```text
outputs/m25_open_vocab_grounding/full_env_validation_20260703/rgb_to_semantic_bev_mvp/rgb_to_semantic_bev_mvp.html
outputs/m3_rgb_geometry_frontend/vggt_habitat_smoke_first16_20260703/rgb_to_semantic_bev_mvp/rgb_to_semantic_bev_mvp.html
```

图示说明已从“彩色 semantic BEV”改成更准确的：

```text
PNG is visualization only.
Internal representation should be:
  occupancy / explored geometry
  + per-class semantic evidence / confidence channels
  + object inventory
  + landmark graph
```

当前结果仍保持 OWLv2 作为 lightweight baseline。下一步应新增 GroundingDINO+SAM/SAM2 分支，与 Habitat oracle 对照，验证 mask-level grounding 是否能显著提升 semantic BEV 稳定性。

### 2026-07-03: Current mainline for RGB-to-paper-grade semantic BEV

当前主线：

```text
GroundingDINO semantic branch
+ RGB/VGGT geometry branch
+ paper-aligned semantic BEV visualization / metrics
```

阶段边界要清楚拆分：

```text
M2.5-GDINO:
  解决 RGB frame -> semantic grounding evidence。
  先使用 Habitat oracle depth/pose 投影，隔离 semantic grounding 质量。

M3-VGGT:
  解决 RGB sequence -> estimated depth/pose/pointcloud。
  在 semantic branch 可控后，再替换 oracle geometry。
```

推荐执行顺序：

```text
1. GroundingDINO(+SAM) on Habitat RGB-D + oracle pose
   -> 与 OWLv2 baseline 比较 semantic BEV 稳定性。

2. 同一批 semantic candidates 接入 M2 bridge / Phase 3 retrieval
   -> 验证数据契约不变。

3. 将 GroundingDINO(+SAM) candidates 投影到 VGGT geometry BEV
   -> 观察 geometry drift 对 semantic BEV 的影响。

4. 输出统一对照页
   oracle semantic / OWLv2 / GroundingDINO(+SAM)
   oracle geometry / VGGT geometry
```

论文级对齐要求：

```text
不只输出一张染色图。
必须同时输出：
  traditional BEV geometry
  semantic evidence BEV
  confidence/evidence strength
  object or mask inventory
  oracle comparison metrics
  detector/mask overlay examples
```

当前可接受的 MVP：

```text
GroundingDINO-only boxes 可作为第一步；
若 SAM/SAM2 安装或权重下载受阻，先完成 detector-box branch。
但最终论文级 semantic BEV 应优先使用 mask-level projection。
```

## 与 M2.5 的关系

M2.5 已经证明：

```text
RGB-D / oracle pose + OWLv2
-> object evidence
-> semantic BEV MVP
```

M3 要替换的是几何来源：

```text
Habitat depth / oracle pose
-> VGGT / DUST3R estimated depth / pose / point cloud
```

语义 grounding backend 和 RSC memory bridge 可继续复用 M2.5 的 schema。

### 2026-07-03: RGB 输入链路阶段出口

当前判断：

```text
RGB 输入链路可以按 MVP 暂时通过，但不按最终论文效果通过。
```

已完成：

```text
1. Habitat RGB-D / coverage-loop frames 已可导出。
2. OWLv2 / GroundingDINO / GroundingDINO+SAM 都已通过 M2.5 schema 进入 BEV bridge。
3. GroundingDINO box + full coverage + multi-view/confidence filter 成为当前默认 semantic branch。
4. SAM 已验证可作为 mask-level projection / localization refinement，但当前不优于 box-only best F1。
5. VGGT geometry frontend 已完成 smoke / first16 MVP，并能对齐到 Habitat world frame 后进入 DenseBEVMapper。
6. 当前结果能生成 traditional BEV、object projection evidence、semantic BEV MVP、confidence map 和 oracle validation metrics。
```

保留限制：

```text
1. VGGT 目前不是默认几何来源；主线仍可使用 Habitat RGB-D / oracle pose 隔离语义记忆变量。
2. 当前 semantic BEV 仍偏 object-centric evidence，不是稳定 dense per-cell semantic segmentation。
3. GroundingDINO / SAM 的局部调参收益已经低于先固定表示契约。
4. 如果直接进入 Phase 5 planner，输入格式会摇摆于 PNG、semantic grid、object inventory 和 landmark graph。
```

阶段出口理由：

```text
本阶段已经证明：
  RGB semantic evidence 和 RGB geometry estimate 都可以进入 RSC-Nav 的 BEV / memory / retrieval 后端。

下一阶段不应继续只优化某个 detector 或 segmenter，
而应先定义 RSC-Nav 的语义空间记忆表示：
  G: geometry BEV
  S: semantic evidence BEV
  O: object memory
  L: landmark / topology graph
```

下一步：

```text
进入 M3.5 Semantic Representation Alignment。
先审计常见 paper 中 semantic BEV / semantic map / object-centric map / topological graph 的表示方式，
再把 RSC-Nav 当前 G/S/O/L 四层表示固定为 Phase 5 planner 的输入契约。
```
