# RSC-Nav 实验协议
*(Experimental Protocol for Adaptive Semantic-Spatial Memory Augmentation)*

![Status: Protocol](https://img.shields.io/badge/Status-Protocol-green)
![Scope: Thesis_Experiments-blue](https://img.shields.io/badge/Scope-Thesis_Experiments-blue)
![Focus: Long_Term_Memory_Augmentation-success](https://img.shields.io/badge/Focus-Long_Term_Memory_Augmentation-success)

---

## 0. 开源继承与实现边界

本项目不从零构建具身导航仿真、数据集和通用导航训练框架。实验实现应大方继承成熟开源生态，将工程工作量集中到 RSC-Nav 的长期语义-空间记忆增强机制和连续目标协议上。

总体原则如下：

- Habitat-Lab / Habitat-Sim、HM3D / HM3D-OVON、Habitat-Baselines / ObjectNav baseline 等成熟生态可作为环境、数据、训练与评估底座。
- GOSE / VLFM / OVRL-V2 / HM3D-OVON baseline 等已有方法可作为模块设计、训练范式和强对照实验的参考。
- CLIP / SigLIP / DINOv2 / open-vocabulary detector 等基础模型可作为冻结感知或语义证据来源，不作为本文主要创新点。
- 每个阶段执行时再具体决定复用、改写或重实现哪些模块，避免在全局规划阶段过早绑定工程细节。

本文主要贡献集中在 RSC-inspired 长期语义-空间记忆增强导航。memory reuse 和 memory adaptive update 是该长期记忆能力的两个必要表现，分别通过 memory reset / carried 和 carried-stale / carried-adaptive 对照验证。

---

## 1. 协议目标

本协议用于支撑 RSC-Nav 的硕士论文主实验。核心目标是验证长期语义-空间记忆增强导航是否成立。该主线下包含两个必要能力：

**Capability 1: Memory Reuse**

> 在同一室内场景连续语义目标导航中，保留语义-空间记忆是否比清空记忆更能复用历史探索信息？

**Capability 2: Memory Adaptive Update**

> 当环境或语义证据发生轻量变化时，新观测是否能修正旧语义-地标记忆，而不是让 agent 盲目依赖过期地图？

主实验不追求完整官方 GOAT-Bench / LangMap，也不以真实机器人部署或完整动态 3D SLAM 为目标。B 档动态变化限定为轻量、可控、可复现的语义-空间扰动。

---

## 2. 实验任务

### 2.1 A 档任务

- Habitat ObjectNav 或小规模 HM3D-OVON。
- 用于验证系统能稳定运行。
- 不要求完整动态修正实验，但 memory 数据结构应从一开始预留 confidence、freshness、last_seen_time 和 status 字段。

### 2.2 B 档主任务

B 档围绕长期语义-空间记忆增强导航这一主线展开，包含两个必要能力的实验验证。

**B1: Memory reuse in static sequential goals**

- HM3D-OVON 子集。
- same-scene sequential semantic goal protocol。
- 目标类型限定为：
  - object goal。
  - landmark-related object goal。
- 核心对照：
  - memory reset。
  - memory carried。

**B2: Memory adaptive update under lightweight semantic-spatial changes**

- lightweight dynamic semantic-spatial update protocol。
- 动态变化类型限定为：
  - object relocation。
  - object disappearance。
  - semantic confidence correction。
- 核心对照：
  - memory reset。
  - carried-stale。
  - carried-adaptive。

暂不将 room / region goal 作为 B 档硬目标，除非数据集中已有可靠 room / region 标注。房间级和区域级目标可放入 C 档扩展。

### 2.3 C 档扩展任务

- 官方 GOAT-Bench 或接近官方协议。
- LangMap / HieraNav 子集。
- pose/depth 噪声鲁棒性。
- 真实 open-vocabulary detector / segmentation 噪声。
- ConvGRU learned fusion。
- learned memory update gate。

---

## 3. 语义证据来源

语义证据分两级。

**MVP 语义证据：**

- 仿真语义标注。
- Habitat semantic sensor。
- 预提取 object / semantic evidence。

MVP 阶段优先验证记忆机制本身，避免感知误差掩盖 memory reuse 和 memory adaptive update 的结论。

**扩展语义证据：**

- CLIP / SigLIP / DINOv2 特征。
- open-vocabulary detector。
- segmentation model。

扩展阶段用于验证感知噪声下的鲁棒性。

---

## 4. 记忆状态定义

RSC-Nav 的 memory item 不只是语义标签或地图坐标。为支持 H4，每个语义或地标记忆单元至少包含：

```text
id
semantic_label / semantic_embedding
bev_position
confidence
freshness
last_seen_time
visit_count
negative_evidence_count
status: active / stale / missing / relocated
context_id or context_embedding
```

MVP 阶段可以使用规则更新：

```text
confirm:
  新观测支持旧证据，提高 confidence，刷新 last_seen_time

weaken:
  重访旧位置但未观测到目标，降低 confidence，增加 negative evidence

relocate:
  同类或同实例证据在新位置连续出现，旧节点标记 stale / relocated，新节点升权

overwrite:
  新证据稳定强于旧证据时，更新 semantic map 或 landmark node
```

---

## 5. Sequential Goal 生成规则

### 5.1 基本格式

```text
scene_i:
  goal_1: find object_a
  goal_2: find object_b
  goal_3: find object_c near object_a
```

示例：

```text
scene_i:
  goal_1: find sofa
  goal_2: find table
  goal_3: find lamp near sofa
```

### 5.2 静态记忆复用控制变量

memory reset 与 memory carried 必须使用：

- 相同 scene。
- 相同初始位置。
- 相同 goal sequence。
- 相同最大步数。
- 相同目标成功半径或 stop 判定规则。

`goal_1` 主要作为探索与记忆写入阶段。从 `goal_2` 开始比较 memory reset 与 memory carried 的性能差异。

### 5.3 轻量动态变化规则

B 档主实验包含轻量动态变化，用于验证语义-空间记忆是否能被新观测修正。动态变化不追求完整真实世界物理模拟，优先采用可控语义扰动：

```text
scene_i:
  goal_1: find object_a
  memory update: object_a observed at location_x
  perturbation: object_a relocated / removed / confidence degraded
  goal_2: find object_a or object_b near object_a
```

动态变化对照必须使用：

- 相同 scene。
- 相同初始位置。
- 相同 goal sequence。
- 相同扰动类型和扰动时间。
- 相同最大步数、成功半径和 stop 判定规则。

核心对照为：

- `memory reset`：清空旧记忆。
- `carried-stale`：保留旧记忆，但不启用负证据、置信度衰减和 stale 修正。
- `carried-adaptive`：保留旧记忆，并允许新观测修正 semantic confidence、landmark status 和检索权重。

---

## 6. 模型与 Baseline

### 6.1 A 档模型

- Reactive。
- LSTM Memory。
- BEV-only。
- RSC-Nav A-minimal。

### 6.2 B 档模型

- Reactive。
- LSTM Memory。
- BEV-only。
- Graph-only。
- RSC-Nav full, memory reset between goals。
- RSC-Nav full, memory carried across goals。
- RSC-Nav full, memory carried with stale update。
- RSC-Nav full, memory carried with adaptive update。

### 6.3 Baseline 作用说明

- Reactive 用于衡量无记忆策略的下限。
- LSTM Memory 用于对比隐式记忆是否足以替代显式语义-空间记忆。
- BEV-only 用于验证稠密空间/语义地图的贡献。
- Graph-only 用于验证地标拓扑记忆的贡献。
- carried-stale 与 carried-adaptive 用于验证 H4，即可更新记忆是否优于静态保留旧记忆。

---

## 7. 消融实验

| 消融 | 对应问题 |
|---|---|
| full allocentric BEV vs local egocentric map | 全局参考系转换是否必要 |
| noisy / degraded pose projection | 参考系转换对位姿质量是否敏感 |
| 无 pose/head-direction code | policy 额外位姿编码是否必要 |
| 无 semantic confidence | 置信度语义融合是否必要 |
| 无 freshness / last_seen_time | 记忆新旧程度是否必要 |
| 无 negative evidence | 旧语义证据能否被新观测削弱 |
| 无 landmark status | active / stale / missing / relocated 状态是否必要 |
| 无 landmark graph | 地标锚定是否必要 |
| Graph-only | 只有拓扑记忆是否足够 |
| BEV-only | 只有稠密地图是否足够 |
| 无 graph-to-BEV attention | 分层检索是否必要 |
| Memory reset between goals | 长期记忆复用是否真的带来收益 |
| carried-stale vs carried-adaptive | adaptive update 是否减少过期记忆误导 |
| 直接动作输出替代 waypoint | waypoint 解耦是否降低训练难度 |

可选扩展：

```text
deterministic / confidence fusion
vs
ConvGRU learned fusion

rule-based memory update
vs
learned memory update gate
```

---

## 8. 指标定义

### 8.1 常规导航指标

- Success Rate。
- SPL。
- Distance-to-Goal。
- Path Length。
- Stop Accuracy。

### 8.2 记忆复用指标

**Memory Reset Gap**

同一 scene、同一 goal、同一初始条件下，memory carried 与 memory reset 的性能差。

**Memory Reuse Gain**

对 goal index `g >= 2`：

```text
Memory Reuse Gain(g) = SPL_carried(g) - SPL_reset(g)
```

也可用 Success、Path Length 或 Distance-to-Goal 定义同类差值。核心原则是比较同一目标在 carried/reset 条件下的差异，而不是把后续目标与首目标直接比较。

**Revisit Success**

目标位于已探索区域时的成功率。

**Retrieval Hit@K**

目标检索是否命中正确地标、对象或区域节点。

**Exploration Efficiency**

单位步数探索面积或目标相关区域覆盖率。

**Repeated-Visit Consistency**

重复访问同一区域时语义/几何记忆的一致性。

### 8.3 记忆自适应更新指标

**Adaptation Success**

环境或语义证据发生轻量变化后，agent 是否仍能完成后续目标。

**Recovery Steps**

旧记忆失效后，agent 需要多少步才能通过新观测修正检索目标或行动方向。

**Stale Memory Error Rate**

由于过期语义或地标记忆导致错误检索、错误返回旧位置或错误 stop 的比例。

**Relocated Object Hit@K**

目标对象位置变化后，检索 top-k 是否命中新位置或更新后的 landmark node。

**Map Correction Latency**

从新观测出现到 semantic map / landmark graph 中旧证据被降低置信度或标记 stale 所需的步数。

---

## 9. 可视化要求

至少输出：

- BEV occupancy / explored / semantic map 随时间更新。
- landmark graph 构建过程。
- semantic confidence、freshness、last_seen_time 和 landmark status 的更新过程。
- goal-to-node top-k retrieval。
- graph-to-BEV attention。
- memory reset vs memory carried 的轨迹对比。
- carried-stale vs carried-adaptive 的动态修正轨迹对比。
- 成功与失败案例分析。

---

## 10. 实现路线

### Phase 0: 协议冻结（W0-W1）

- 确定 Habitat 版本、数据子集、动作空间和 BEV 分辨率。
- 冻结 same-scene sequential goal protocol。
- 冻结 lightweight dynamic semantic-spatial update protocol。
- 冻结语义证据来源。
- 冻结 memory item 字段和 update rule。
- 产出 `dataset_spec.md`、episode 生成脚本和 10 条可视化轨迹。

### Phase 1: BEV 记忆（W2-W5）

- 实现 RGB-D 到 BEV 投影。
- 实现 log-odds occupancy map。
- 实现 deterministic explored map。
- 实现 confidence-weighted semantic map。
- 实现 semantic confidence decay、negative evidence、last_seen_time 和 stale flag。

### Phase 2: 轻量拓扑记忆（W6-W8）

- 实现 keyframe node 和 landmark node。
- 实现 temporal / spatial adjacency。
- 实现节点合并、visit count、timestamp 和 active / stale / missing / relocated 状态。
- 实现 confirm / weaken / relocate / overwrite 四类规则更新。

### Phase 3: 检索与策略头（W9-W12）

- 实现 goal-to-node top-k retrieval。
- 实现 node-to-BEV attention。
- 实现 waypoint / stop head。
- 使用 shortest-path 或 waypoint teacher 做 imitation learning。
- 训练输入包含 confidence、freshness、stale flag 和 landmark status，使策略能区分可靠新观测与过期旧记忆。

### Phase 4: A 档实验（W13-W16）

- 训练 Reactive、LSTM、BEV-only、RSC-Nav A-minimal。
- 完成 ObjectNav / 小规模 OVON 表格。
- 验证 memory 数据结构和可视化不影响系统稳定运行。

### Phase 5: B 档主实验（W17-W22）

- 构建 HM3D-OVON 子集。
- 构建 same-scene sequential goal protocol。
- 构建 lightweight dynamic semantic-spatial update protocol。
- 加入 Graph-only baseline。
- 加入 Memory Reset / Memory Carried baseline。
- 加入 carried-stale / carried-adaptive 对照。
- 完成记忆复用指标、记忆自适应更新指标、消融表和轨迹对比。

### Phase 6: 论文整理（W23-W25）

- 整理方法图、实验表、指标曲线和可视化。
- 撰写论文初稿。
- 若 B 档结果稳定，再选择一个 C 档扩展。

---

> **维护者:** WangLab / RSC_VLN  
> **版本:** v2 Experimental Protocol  
> **最后更新时间:** 2026-06-14
