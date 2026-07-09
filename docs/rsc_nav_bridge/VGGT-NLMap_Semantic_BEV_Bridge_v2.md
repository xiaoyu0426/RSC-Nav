# VGGT-NLMap Semantic BEV Bridge v2
*(RGB-only 3D Reconstruction + Open-Vocabulary Semantic Mapping + Long-Term BEV Memory)*

---

## 1. 项目一句话

**VGGT-NLMap Semantic BEV Bridge** 是一个面向 3D 视觉与具身导航的桥接项目：

```text
手机 / 离线 RGB 图像序列
-> VGGT / DUST3R 重建几何
-> NLMap-style open-vocabulary semantic grounding
-> semantic BEV / occupancy BEV
-> RSC-Nav long-term object memory
```

短期目标：作为符合 3D 视觉岗位 JD 的项目展示，体现 RGB-only 3D reconstruction、open-vocabulary semantic mapping、BEV / occupancy、embodied navigation memory 的综合实践能力。

长期目标：作为 RSC-Nav 论文方向的前期扩展补充，为 RSC-Nav 提供更真实的语义感知与几何重建前端。

---

## 2. 为什么这样命名

```text
VGGT:
  体现 3D vision foundation model / RGB-only geometry reconstruction

NLMap:
  体现 language-conditioned open-vocabulary 3D semantic mapping

Semantic BEV:
  体现 Occupancy / BEV / semantic map 技术栈

Bridge:
  体现它连接感知前端与 RSC-Nav 长期记忆后端
```

这个名字对外比 `RSC-Nav Bridge` 更直接，因为岗位 JD 里更容易识别：

- VGGT / DUST3R：近年 3D foundation model。
- NLMap：语言条件 3D 语义地图。
- BEV / occupancy：机器人和自动驾驶常用空间表示。

---

## 3. 三个系统的角色

### 3.1 VGGT / DUST3R: Geometry Frontend

作用：

```text
RGB sequence
-> camera pose
-> depth / pointmap
-> point cloud
```

它解决的是几何来源问题：如果没有 RGB-D 相机或 oracle pose，也能从普通 RGB 图像序列恢复一个可用的 3D 场景结构。

第一优先级建议：

```text
VGGT first
DUST3R as fallback / comparison
```

原因：VGGT 更适合包装成近年 3D vision foundation model 实践；DUST3R 更成熟，可作为备选。

### 3.2 NLMap-style Semantic Grounding Frontend

作用：

```text
language query / object prompt
-> open-vocabulary object grounding
-> object label / bbox / mask / score
-> 3D object location
```

这里的 NLMap 不只是离线数据集，也不必死守原版 CLIP / ViLD。我们采用更宽泛的表述：

```text
NLMap-style open-vocabulary 3D semantic mapping
```

也就是保留 NLMap 的系统范式：

```text
language query
-> semantic grounding
-> 3D scene association
-> queryable semantic scene representation
```

原版 CLIP / ViLD 作为 baseline；实际 MVP 使用更轻量现代的 grounding backend。

### 3.3 RSC-Nav: Long-Term BEV Memory Backend

作用：

```text
occupancy BEV / semantic BEV / object memory
-> landmark retrieval
-> adaptive update
-> context remapping
```

RSC-Nav 不是这个 bridge 的第一步依赖，但它是后续落点。Bridge 生成的 semantic BEV 和 object memory 会作为 RSC-Nav 的外部语义地图输入。

---

## 4. Grounding Backend 决策

最终采用 two-tier backend：

```text
MVP backend:
  OWLv2 / OWL-ViT

Quality backend:
  GroundingDINO + SAM

Baseline backend:
  CLIP / ViLD
```

### 4.1 MVP: OWLv2 / OWL-ViT

优点：

- 工程更轻。
- 依赖更少。
- 推理和调试更快。
- 更适合先跑通端到端闭环。
- 对少量室内类别足够，例如 chair、table、door、bed、sofa、cup。

缺点：

- 多数情况下主要输出 box。
- mask 质量不如 SAM。
- semantic BEV 边界可能偏粗。
- 小物体、遮挡物体、复杂场景 recall 可能不足。

适用阶段：

```text
Bridge Stage A/B MVP
end-to-end bridge validation
quick demo
```

### 4.2 Quality: GroundingDINO + SAM

优点：

- open-vocabulary grounding 质量更强。
- SAM mask 能提供更干净的物体区域。
- 3D semantic projection 更准。
- semantic BEV 展示效果更好。
- 更适合论文图、展示视频和复杂场景。

缺点：

- 模型和依赖更重。
- 显存、速度和部署成本更高。
- 工程链路更长：

```text
GroundingDINO box
-> SAM mask
-> depth / pose projection
-> semantic BEV
```

适用阶段：

```text
quality upgrade
paper visualization
complex scenes
```

### 4.3 Baseline: CLIP / ViLD

作用：

- 与 NLMap 原论文路线保持一致。
- 作为复现 baseline。
- 不作为当前主实现路线。

原因：

- ViLD 较老。
- TensorFlow 依赖偏重。
- 工程维护成本高。
- 对手机 RGB / VGGT-DUST3R 扩展不如现代 backend 顺。

---

## 5. 推荐实施路线

统一执行顺序如下。`M1` 只指 RSC-Nav Phase 3 minimal；Bridge 内部仍使用 Stage A/B/C 表述，避免和 RSC-Nav 原始 Phase 编号混淆。

```text
1. RSC-Nav Phase 3 M1
  object memory -> landmark nodes -> goal query top-k retrieval

2. Bridge Stage A/B
  NLMap-style semantic map
  -> BEV / semantic BEV
  -> RSC object memory
  -> Phase 3 retrieval

3. Bridge Stage C
  RGB sequence
  -> VGGT / DUST3R depth / pose / point cloud
  -> same BEV bridge

4. Return to later RSC-Nav phases
  context remapping / navigation policy / main experiments
```

### Step 1: RSC-Nav Phase 3 M1

先完成 RSC-Nav 的最小 landmark retrieval，让后续外部导入的 semantic objects 有明确落点。

输入：

```text
RSC object memory
goal query
```

输出：

```text
landmark_nodes.json
topk_retrieval.json
retrieval_visualization.html
```

目标：

- object memory 转 landmark nodes。
- 支持 goal-to-node top-k retrieval。
- 检索分数包含 semantic match、confidence、freshness、status、context match。
- 检索结果能投影回 BEV 区域。

### Step 2: Bridge Stage A/B

先不用 VGGT / DUST3R，直接用已有 NLMap 离线 RGB-D / pose / point cloud 数据跑通 NLMap-style semantic map 到 BEV / semantic BEV / RSC object memory 的桥接，并接入 Step 1 的 Phase 3 retrieval。

输入：

```text
RGB-D frames
pose_data.pkl
pointcloud.pcd
object detections / labels / scores
```

输出：

```text
occupancy_bev.npz
semantic_bev.npz
objects.json
rsc_memory_init.json
bridge_report.html
```

目标：

- 读取 point cloud。
- 生成 occupancy BEV。
- 将 object grounding 结果投影到 semantic BEV。
- 导出 RSC-Nav 可读 object memory。
- 接入 Phase 3 M1 retrieval，验证 imported objects 可被 goal query 检索。
- 生成可视化报告。

### Step 3: Bridge Stage C

用手机或离线 RGB 图像序列替换 RGB-D / pose 几何来源。

输入：

```text
RGB image sequence
optional camera intrinsics
```

输出：

```text
estimated_camera_pose
estimated_depth / pointmap
estimated_pointcloud.ply / .pcd
```

目标：

- 跑通 VGGT 或 DUST3R demo。
- 从 RGB 序列生成 point cloud。
- 与原 RGB-D / pose point cloud 做可视化对比。
- 走同一个 Stage A/B BEV bridge。

### Step 4: Return to RSC-Nav 后续阶段

完成 Bridge Stage A/B/C 后，再回到 RSC-Nav 后续阶段：

```text
RSC-Nav Phase 4:
  context remapping

RSC-Nav Phase 5:
  retrieval-conditioned waypoint / stop decision

RSC-Nav Phase 6+:
  main experiments and thesis packaging
```

---

## 6. 与 RSC-Nav 的开发顺序

推荐顺序：

```text
1. RSC-Nav Phase 3 M1
2. Bridge Stage A/B
3. Bridge Stage C
4. RSC-Nav later phases
```

原因：

1. **Phase 3 给 Bridge 提供落点**  
   NLMap objects 不能只画在 BEV 上，最好能进入 RSC object memory 和 landmark retrieval。

2. **Bridge 先于 RSC 后续 Phase 4/5 更划算**  
   Phase 4 context remapping 和 Phase 5 navigation policy 依赖更重。Bridge 能更快形成 JD 可展示成果。

3. **Bridge 反过来服务 RSC-Nav**  
   Bridge 生成的 real / semi-real semantic BEV 可作为后续 RSC-Nav Phase 4 context remapping 和 Phase 5 navigation 的输入。

---

## 7. 技术难点

### 7.1 Point Cloud -> Occupancy BEV

简单投影不难：

```text
3D point (x, y, z)
-> BEV cell (i, j)
```

难点：

- 坐标系对齐。
- BEV 原点、分辨率、边界设定。
- floor / wall / furniture / obstacle 高度区分。
- free / occupied / unknown 推断。
- 点云噪声、空洞和多帧融合。

### 7.2 Semantic Grounding -> Semantic BEV

难点：

- box / mask 到 3D 点的投影。
- 多视角同一物体合并。
- label 同义词归一。
- object centroid / extent 估计。
- confidence 聚合。
- RSC memory schema 对齐。

### 7.3 RGB-only Geometry

难点：

- pose scale ambiguity。
- point cloud drift。
- 重建空洞。
- 手机视频运动模糊。
- 坐标系与 BEV / NLMap / RSC 对齐。

---

## 8. 最小 MVP

第一版只追求端到端闭环：

```text
NLMap offline RGB-D / pose / pointcloud
-> OWLv2 / OWL-ViT object grounding
-> semantic BEV / occupancy BEV
-> RSC object memory json
-> visualization html
```

MVP 不要求：

- 完整真实机器人控制。
- 完整 RSC navigation policy。
- 高质量 mask。
- VGGT / DUST3R 替代几何。

MVP 成功标准：

- 能显示 occupancy BEV。
- 能显示 semantic BEV。
- 能列出 object memory。
- 能被 RSC-Nav Phase 3 M1 retrieval 读取。

---

## 9. 可贴技术栈

MVP 完成后可贴：

```text
Open-vocabulary 3D semantic mapping
OWL-ViT / OWLv2 visual grounding
RGB-D / pose point cloud processing
Point cloud to occupancy BEV
Semantic BEV construction
Embodied navigation memory
RSC-Nav object memory import
```

Quality backend 完成后可贴：

```text
GroundingDINO + SAM
Open-vocabulary segmentation
mask-to-3D semantic projection
high-quality semantic BEV
```

VGGT / DUST3R 完成后可贴：

```text
VGGT / DUST3R reproduction
RGB-only 3D reconstruction
camera pose / depth / pointmap estimation
3D foundation model frontend for navigation memory
```

---

## 10. 项目表述

简历长版：

> 构建 VGGT-NLMap Semantic BEV Bridge，将 NLMap-style open-vocabulary 3D semantic mapping 转换为 occupancy BEV、semantic BEV 和 RSC-Nav object memory；采用 OWLv2 / OWL-ViT 作为轻量级 grounding backend 跑通端到端链路，并预留 GroundingDINO + SAM 质量增强。后续接入 VGGT / DUST3R，用 RGB 序列恢复 point cloud 与 camera pose，减少对 RGB-D / oracle pose 的依赖。

简历短版：

> VGGT-NLMap Semantic BEV Bridge：将开放词汇 3D 语义地图投影为 BEV / occupancy memory，并探索 VGGT / DUST3R 作为 RGB-only 几何前端，后续接入 RSC-Nav 长期语义空间记忆。

论文扩展表述：

> 该桥接项目为 RSC-Nav 提供一个更真实的感知前端：使用 NLMap-style open-vocabulary semantic grounding 和 RGB-only geometry reconstruction，将真实或半真实场景转为可长期复用和更新的 BEV semantic memory。

---

## 11. 当前状态

当前文档为 v2 初步规划。

已有基础：

- NLMap-Qwen3 项目已有 Qwen3 object proposal、NLMap 复现、RGB-D / pose / point cloud / open-vocabulary semantic map 基础。
- RSC-Nav Phase 2 已有 BEV / semantic BEV / object memory、negative evidence update、A/B stress visualization。
- RSC-Nav Phase 3 M1 是接入 Bridge 前的推荐前置步骤。

尚未实现：

- NLMap-style semantic map -> BEV bridge。
- OWLv2 / OWL-ViT backend adapter。
- RSC external memory import API。
- NLMap objects -> RSC landmark nodes。
- GroundingDINO + SAM quality backend。
- VGGT / DUST3R demo 接入。

---

> **版本:** bridge-v2  
> **最后更新时间:** 2026-06-29
