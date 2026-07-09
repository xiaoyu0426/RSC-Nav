# RSC-Nav：压后皮层启发的语义-空间记忆增强导航方法
*(RSC-Nav: Retrosplenial-Inspired Semantic-Spatial Memory Augmentation for Navigation)*

![Status: Project Overview](https://img.shields.io/badge/Status-Project_Overview-blue)
![Environment: Habitat](https://img.shields.io/badge/Environment-Habitat--Lab-blue)
![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Focus: Adaptive_Memory_Augmentation-success](https://img.shields.io/badge/Focus-Adaptive_Memory_Augmentation-success)

---

## 1. 项目定位

RSC-Nav 研究**语义-空间记忆增强导航**。它不是一般意义上的视觉语言导航（VLN），也不是一个试图覆盖所有具身导航任务的大系统。项目主线收敛为：

> 在室内连续目标导航中，显式语义-空间记忆是否能帮助 agent 复用历史探索信息，并在新观测出现时修正过期或冲突的记忆，从而比非结构化记忆、静态地图或单一空间记忆更可靠地完成后续目标？

项目关注的核心场景是：agent 在同一室内环境中连续接收多个语义目标。与每个目标都从空记忆开始不同，RSC-Nav 希望通过持续维护语义-空间记忆，在后续目标中复用先前探索过的空间、地标和语义证据。

从计算角度看，RSC-Nav 也解决连续导航中的**历史上下文膨胀与遗忘问题**。agent 不可能把所有历史 RGB-D 帧、动作、语义证据和轨迹都直接放入策略网络的短期上下文中。RSC-Nav 将长历史压缩为可检索、可更新的外部语义-空间记忆，使策略只读取与当前目标相关的地图、地标和记忆状态。

同时，RSC-Nav 不将记忆视为一次性构建的静态地图。受海马体 remapping 和 RSC context-dependent spatial representations 启发，长期语义-空间记忆应具备 adaptive update 能力：旧证据可以被确认、削弱、迁移或覆盖，进而影响后续的目标检索和 waypoint 选择。

最终论文题目建议：

> **RSC-Nav: Retrosplenial-Inspired Semantic-Spatial Memory Augmentation for Navigation**

中文题目：

> **RSC-Nav：压后皮层启发的语义-空间记忆增强导航方法**

---

## 2. 核心研究假设

本项目将压后皮层（Retrosplenial Cortex, RSC）作为**计算启发**，而不是生物机制复现。RSC-inspired 体现在四个可检验假设：

### H1: 参考系转换

第一视角 RGB-D 观测需要被转换到世界中心或局部全局一致的空间记忆中。该假设对应：

```text
egocentric RGB-D observation + pose
-> allocentric BEV memory
```

### H2: 地标锚定

长期导航不能只依赖连续帧隐状态。稳定对象、关键视角和区域线索应被锚定为可复用的记忆单元。该假设对应：

```text
semantic evidence + keyframes
-> landmark-topological memory
```

### H3: 目标条件检索

当前目标应先检索长期语义-空间记忆，再映射到可行动的局部区域。该假设对应：

```text
semantic goal
-> relevant landmarks / nodes
-> BEV action context
-> waypoint / stop
```

### H4: Context-Dependent Memory Reconfiguration

Inspired by hippocampal remapping and RSC context-dependent spatial representations, RSC-Nav treats semantic-spatial memory as an updateable representation rather than a static map. New observations can weaken, confirm, relocate, or overwrite previous semantic-landmark evidence, and this adaptive memory state directly affects goal-conditioned retrieval and waypoint selection.

该假设对应：

```text
new observation + previous semantic-spatial memory
-> confidence / freshness / context update
-> partial reconfiguration of semantic map and landmark graph
-> stale, missing or relocated landmark correction
-> adaptive retrieval and navigation
```

---

## 3. 方法总览

RSC-Nav 是一个结构化、可更新的语义-空间记忆增强导航框架。它由四类记忆与两个决策模块组成：

```text
RGB-D + pose + semantic goal
        |
        v
Allocentric BEV projection
        |
        +--> Occupancy / explored memory
        +--> Semantic feature memory
        +--> Landmark-topological memory
        +--> Adaptive memory state
                confidence / freshness / status / context
                     |
semantic goal --------+
        |
        v
Goal-conditioned hierarchical retrieval
        |
        v
Waypoint / stop policy
```

设计原则：

- **几何记忆显式化**：occupancy 与 explored map 使用 deterministic / Bayesian fusion。
- **语义记忆置信化**：semantic map 使用 confidence-weighted fusion，并保留 freshness、last_seen_time 和 negative evidence。
- **长期结构图式化**：landmark-topological memory 使用轻量 keyframe / landmark node 与 temporal / spatial adjacency。
- **记忆状态可重配置**：旧语义和地标证据可被新观测确认、削弱、迁移或覆盖。
- **历史上下文外部化**：长历史观测不直接堆入策略上下文，而是压缩到可检索的 semantic-spatial memory 中。
- **学习模块聚焦检索和决策**：retrieval / attention 负责目标条件读取，policy 负责 waypoint / stop。
- **ConvGRU 不作为核心模块**：仅作为可选 learned fusion 扩展。

---

## 4. 核心模块

### 4.1 输入与参考系转换

输入包括 RGB-D、相机内参、pose / heading 和 semantic goal。第一阶段使用 oracle pose/depth，以隔离 SLAM 噪声，优先验证语义-空间记忆机制本身。

### 4.2 Occupancy / Explored Memory

该模块维护 obstacle、free-space、unknown 和 explored 区域，为局部 waypoint 和 stop 判断提供稳定几何基础。MVP 阶段优先实现 2D BEV 记忆，不追求完整 3D 重建。

### 4.3 Semantic Feature Memory

该模块维护对象、房间、区域和地标相关语义证据。每条语义证据至少应包含 confidence、last_seen_time、visit_count、freshness 和 negative evidence，以支持静态复用和动态修正。

### 4.4 Landmark-Topological Memory

该模块维护 keyframe node 和 landmark node，用于表示长期可复用的空间线索。第一版只保留轻量节点、temporal / spatial adjacency、节点合并和状态字段，避免图构建过重。

节点状态包括：

```text
active / stale / missing / relocated
```

### 4.5 Adaptive Memory Reconfiguration

该模块负责根据新观测更新语义-空间记忆状态。MVP 阶段优先使用规则更新，保证机制清晰、可解释、可消融：

```text
confirm:   新观测再次支持旧证据
weaken:    重访旧位置但未观察到目标
relocate:  同一语义或实例在新位置被连续确认
overwrite: 新证据稳定强于旧证据
```

后续可扩展为 learned memory update gate，但不作为 B 档硬目标。

### 4.6 Goal-Conditioned Retrieval

检索模块先根据目标检索相关地标或节点，再将节点上下文映射到 BEV 区域，形成策略输入。检索分数不只依赖 semantic match，也应考虑 confidence、freshness、landmark status 和 context match。

### 4.7 Waypoint / Stop Policy

策略头输出短期 waypoint 与 stop probability。低层控制器将 waypoint 转换为 Habitat 离散动作。第一阶段不从零 PPO 训练，主线使用 shortest-path 或 waypoint teacher 进行 imitation learning。

---

## 5. 目标收敛

为了控制硕士论文工作量，项目采用 A/B/C 三档目标。

### A 档：保底系统验证

目标：证明系统能在标准导航任务中稳定运行。

- Habitat ObjectNav 或小规模 HM3D-OVON。
- 完成 BEV memory、semantic memory、lightweight landmark memory。
- 对比 Reactive、LSTM、BEV-only。
- 输出基础 Success / SPL / Distance-to-Goal 和可视化。

### B 档：硕士论文主目标

目标：证明长期语义-空间记忆增强导航在同一场景连续目标任务中的有效性。该主线下包含两个必要能力：

1. **Memory reuse**：保留记忆能帮助后续目标复用历史探索信息。
2. **Memory adaptive update**：新观测能修正过期、冲突或迁移的语义-地标记忆，减少 stale memory 对导航的误导。

B 档包含：

- HM3D-OVON 子集。
- 自建 same-scene sequential semantic goal protocol。
- 自建 lightweight dynamic semantic-spatial update protocol。
- 对比 Reactive、LSTM、BEV-only、Graph-only、RSC-Nav。
- 加入关键对照：memory reset vs memory carried。
- 加入动态修正对照：carried-stale vs carried-adaptive。
- 重点分析 Memory Reset Gap、Memory Reuse Gain、Retrieval Hit@K、Revisit Success、Adaptation Success、Recovery Steps、Stale Memory Error Rate 和轨迹可视化。

### C 档：投稿/冲刺扩展

仅在 B 档稳定后选择性加入：

- 官方 GOAT-Bench 或接近官方协议。
- LangMap / HieraNav 子集。
- pose/depth 噪声鲁棒性。
- 真实 open-vocabulary detector / segmentation 噪声。
- ConvGRU learned fusion 或 learned memory update gate。
- Docker/EvalAI 风格复现实验封装。

---

## 6. 预期贡献

### 模型贡献

提出一种结构化、可更新的语义-空间记忆增强导航框架，将 BEV 几何记忆、语义特征记忆、地标拓扑记忆和 adaptive memory state 统一到目标条件检索策略中。

### 认知启发贡献

将 RSC 的参考系转换、地标锚定、目标定向检索和 context-dependent memory reconfiguration 转化为可实现、可消融的计算模块。

### 实验贡献

通过 same-scene sequential goal protocol、memory reset / memory carried 对照，以及 carried-stale / carried-adaptive 对照，验证长期语义-空间记忆增强导航中的 memory reuse 和 memory adaptive update。

### 工程贡献

形成一套可复现的 Habitat 实验管线，包括数据生成、BEV 投影、语义记忆、拓扑图、记忆状态更新、检索策略、导航策略和可视化。

---

## 7. 与已有研究的关系

已有研究已经覆盖了许多单独部件：

- occupancy / semantic map for navigation。
- ObjectNav / OVON。
- semantic SLAM / lifelong semantic mapping。
- topological memory / landmark graph。
- cross-modal attention。
- waypoint navigation。
- lifelong navigation benchmarks。

本项目不将单个部件作为创新，而是强调：

1. 用 RSC 计算假设组织结构化语义-空间记忆。
2. 用显式记忆增强导航策略，而不是单纯依赖循环隐状态。
3. 用 memory reset vs memory carried 的连续目标协议，直接验证记忆复用收益。
4. 用 carried-stale vs carried-adaptive 的轻量动态协议，验证记忆能否被新观测修正。

---

## 8. 风险与收敛策略

| 风险 | 应对 |
|---|---|
| 官方 GOAT-Bench / LangMap 工程量过大 | 降为 C 档，B 档使用自建 sequential goal |
| 动态环境工程量过大 | B 档只做轻量语义-空间变化，不做完整动态 3D SLAM |
| 语义特征不稳定 | MVP 使用仿真语义标注或预提取 semantic evidence |
| 拓扑图生成复杂 | MVP 只保留 keyframe + landmark 和轻量状态字段 |
| 自适应更新过于复杂 | B 档使用规则更新，learned update gate 放入 C 档 |
| PPO 训练不稳定 | 主线使用 imitation learning |
| RSC 叙事被质疑 | 明确写成计算假设，并用四类假设对应消融验证 |
| 记忆收益不明显 | 使用 Memory Reset Gap、Retrieval Hit@K、Revisit Success 等针对性指标 |
| 静态记忆被质疑实用价值有限 | B 档加入 carried-stale vs carried-adaptive，验证新观测能修正语义-空间记忆 |

---

## 9. 最终产出

硕士论文最终应产出：

1. 一个可复现的 RSC-Nav 原型系统。
2. 一套 same-scene sequential semantic goal 评估协议。
3. 一套 lightweight dynamic semantic-spatial update 评估协议。
4. 一组 ObjectNav / OVON 子集上的基础导航结果。
5. 一组 memory reset vs memory carried 的记忆复用实验。
6. 一组 carried-stale vs carried-adaptive 的记忆修正实验。
7. 消融实验、记忆检索指标、动态修正指标和可视化分析。
8. 论文初稿、方法图、实验 README 和关键脚本。

---

## 10. 配套文档

具体实验协议、指标公式、控制变量、语义证据来源和 episode 生成规则详见：

- [RSC-Nav_实验协议.md](E:/WangLab/RSC_VLN/RSC-Nav_实验协议.md)

---

> **维护者:** WangLab / RSC_VLN  
> **版本:** v6 Project Overview  
> **最后更新时间:** 2026-06-14  
> **设计原则:** 项目总览负责阐明研究方向、架构和论文主张；实验细节放入独立协议文档，避免主文档被细节淹没。
