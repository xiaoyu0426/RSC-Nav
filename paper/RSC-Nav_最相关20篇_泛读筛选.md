# RSC-Nav 最相关 20 篇文献泛读筛选

> 整理日期：2026-06-15  
> 来源：从 `RSC-Nav_相关文献50篇_摘要级整理.md` 中筛选。  
> 阅读状态：基于题目、摘要、项目页与公开元信息的泛读筛选，尚非全文精读。  
> 筛选目标：优先支持 RSC-Nav 的唯一主线——**长期语义-空间记忆增强导航**。  

---

## 0. 筛选标准

RSC-Nav 的长期记忆能力包含两个必要表现：

1. **Memory reuse**：过去探索到的空间、语义和地标信息能被后续目标复用。
2. **Memory adaptive update**：旧语义-地标记忆能被新观测确认、削弱、迁移或覆盖，避免 stale memory 误导导航。

因此，本轮从 50 篇中优先筛选符合以下条件的论文：

- 与 ObjectNav / OVON / multi-goal navigation 直接相关。
- 明确涉及显式地图、语义地图、拓扑图、frontier 或 instance memory。
- 能为 `memory reset vs memory carried` 或 `carried-stale vs carried-adaptive` 提供实验对照参考。
- 能支撑 RSC-inspired 的四个假设：参考系转换、地标锚定、目标条件检索、context-dependent memory reconfiguration。
- 工程上可直接借鉴 Habitat / HM3D / baseline / map / retrieval 设计。

---

## 1. 一句话结论

最值得优先读的 20 篇可以分成四组：

| 组别 | 文献作用 | 代表论文 |
|---|---|---|
| A. 任务与 benchmark | 决定 RSC-Nav 应该在哪类任务上讲故事 | HM3D-OVON、GOAT、GOAT-Bench |
| B. 强 baseline 与模块化导航 | 决定我们该借鉴/对比谁 | GOSE、VLFM、OVRL-V2、RIM |
| C. 地图、拓扑与长期记忆模块 | 决定 BEV / graph / memory update 怎么做 | Active Neural SLAM、Neural Topological SLAM、Lifelong Semantic Mapping |
| D. RSC / 海马认知启发 | 决定 H1-H4 的神经科学叙事边界 | RSC reviews、hippocampal remapping |

---

## 2. 最相关 20 篇总表

| 优先级 | 论文 | 核心关系 | 建议阅读深度 |
|---:|---|---|---|
| 1 | GOAT: GO to Any Thing | lifelong navigation + instance-aware semantic memory | 精读 |
| 2 | GOAT-Bench | 多目标 lifelong benchmark，直接对应连续目标协议 | 精读 |
| 3 | HM3D-OVON | RSC-Nav B 档数据和 OVON 任务基础 | 精读 |
| 4 | VLFM | 最重要模块化强 baseline 之一 | 精读 |
| 5 | Object Goal Navigation using Goal-Oriented Semantic Exploration | 语义地图 ObjectNav 经典 baseline | 精读 |
| 6 | OVRL-V2 | 强 learned baseline，对照显式记忆路线 | 精读 |
| 7 | Object Goal Navigation with Recursive Implicit Maps | 隐式地图强相关，帮助界定显式/隐式差异 | 泛读到方法 |
| 8 | Active Neural SLAM | BEV map + planner 工程路线参考 | 泛读到方法 |
| 9 | Neural Topological SLAM | landmark / topological memory 参考 | 泛读到方法 |
| 10 | Learning to Map for Active Semantic Goal Navigation | semantic map prediction 与目标导向探索 | 泛读到方法 |
| 11 | Lifelong Semantic Mapping in Indoor Environments | adaptive update、stale memory、长期语义地图 | 泛读到方法 |
| 12 | Kimera | 动态 scene graph / 语义空间结构参考 | 泛读到结构 |
| 13 | Neural Map | 空间结构化外部记忆基础 | 泛读 |
| 14 | EgoMap | 投影式结构化记忆，与 H1 对齐 | 泛读 |
| 15 | Memory-Augmented RL for Image-Goal Navigation | episodic memory + attention retrieval | 泛读 |
| 16 | What Does the Retrosplenial Cortex Do? | RSC 总体认知定位 | 精读 |
| 17 | Cues, Context, and Long-Term Memory | RSC、context、long-term memory | 精读 |
| 18 | Retrosplenial Cortex and Its Role in Spatial Cognition | RSC 空间认知综述 | 精读 |
| 19 | Retrosplenial Representations of Space and Future Goal Locations Develop with Learning | RSC 目标位置和学习表征 | 泛读到实验结论 |
| 20 | Understanding Memory through Hippocampal Remapping | H4 的 remapping 理论来源 | 精读 |

---

## 3. 必读核心 10 篇

### 01. GOAT: GO to Any Thing

- 链接：https://arxiv.org/abs/2311.06430
- 关键词：lifelong navigation, multimodal goal, instance-aware semantic memory
- 泛读摘要：GOAT 是最接近 RSC-Nav 主线的系统之一。它强调机器人在同一环境中长期执行多个目标，并通过 continually augmented instance-aware semantic memory 从过去经验中获益。它不是只做单目标 ObjectNav，而是关注经验积累如何提升后续导航。
- 为什么入选：它直接证明“长期记忆增强导航”是当前 embodied navigation 的重要方向。RSC-Nav 可以借鉴它的 instance memory 思想，但应把差异放在 RSC-inspired semantic-spatial memory、BEV + landmark graph、adaptive update 机制上。
- 对 RSC-Nav 的具体作用：
  - 定义长期记忆增强导航的近邻工作。
  - 帮助设计 memory carried across goals 的实验。
  - 可作为论文 related work 中最重要的对照之一。
- 下一步精读重点：
  - instance-aware semantic memory 数据结构。
  - lifelong setting 如何定义。
  - success 随经验增长的实验设计。
  - 是否处理 stale / changed memory。

### 02. GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation

- 链接：https://arxiv.org/abs/2404.06609
- 关键词：lifelong benchmark, sequential goals, explicit memory, implicit memory
- 泛读摘要：GOAT-Bench 把导航目标扩展到类别、语言描述和图像，并要求 agent 在序列目标中持续导航。论文专门分析显式和隐式 scene memory 在 lifelong scenarios 中的作用。
- 为什么入选：RSC-Nav 的 same-scene sequential semantic goal protocol 和 `memory reset vs memory carried` 很接近 GOAT-Bench 的问题空间。
- 对 RSC-Nav 的具体作用：
  - 参考多目标 episode 组织方式。
  - 参考 explicit vs implicit memory 对照。
  - 为 B 档协议和 C 档扩展提供 benchmark 背景。
- 下一步精读重点：
  - lifelong episode 的控制变量。
  - memory reset / memory carry 是否有类似设置。
  - 使用哪些指标评估长期记忆收益。

### 03. HM3D-OVON: A Dataset and Benchmark for Open-Vocabulary Object Goal Navigation

- 链接：https://arxiv.org/abs/2409.14296
- 关键词：HM3D, OVON, open-vocabulary ObjectNav
- 泛读摘要：HM3D-OVON 基于 HM3DSEM，提供开放词汇对象目标和大量对象实例，是 RSC-Nav B 档最合适的数据来源之一。它还比较了 IL、RL、modular 等路线。
- 为什么入选：RSC-Nav 的实验任务、语义目标来源和 baseline 工程都可直接基于 HM3D-OVON。
- 对 RSC-Nav 的具体作用：
  - 定义 HM3D-OVON 子集。
  - 支撑 object goal / landmark-related object goal。
  - 借鉴 baseline 训练与评估管线。
- 下一步精读重点：
  - episode 生成规则。
  - open-vocabulary goal 表示。
  - baseline 模型和训练细节。

### 04. VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation

- 链接：https://arxiv.org/abs/2312.03275
- 关键词：frontier, value map, VLM, zero-shot ObjectNav
- 泛读摘要：VLFM 用 depth 构建 occupancy map 和 frontier，再用视觉语言模型为 frontier / region 赋予目标相关价值，选择最有希望找到目标的探索方向。
- 为什么入选：它是当前模块化 semantic navigation 的强代表，也是 RSC-Nav 很可能被要求比较或至少讨论的 baseline。
- 对 RSC-Nav 的具体作用：
  - 借鉴 frontier / value map 设计。
  - 作为 strong modular baseline。
  - 帮助说明 RSC-Nav 与“当前目标驱动探索”不同，重点是长期 memory reuse 和 adaptive update。
- 下一步精读重点：
  - value map 如何构建。
  - frontier selection 如何与 navigation policy 连接。
  - 是否保留跨目标长期记忆。

### 05. Object Goal Navigation using Goal-Oriented Semantic Exploration

- 链接：https://arxiv.org/abs/2007.00643
- 关键词：ObjectNav, semantic map, modular exploration
- 泛读摘要：GOSE / SemExp 是语义地图 ObjectNav 的经典工作。它构建 episodic semantic map，并利用目标类别相关语义先验指导探索。
- 为什么入选：RSC-Nav 的 BEV semantic memory、ObjectNav A 档 baseline、模块化设计都绕不开这篇。
- 对 RSC-Nav 的具体作用：
  - BEV occupancy / explored / semantic map 设计参考。
  - 目标导向探索 baseline。
  - 消融中 BEV-only 的理论背景。
- 下一步精读重点：
  - semantic map 表示。
  - long-term goal selection。
  - 模块化系统如何评估和消融。

### 06. OVRL-V2: A Simple State-of-Art Baseline for ImageNav and ObjectNav

- 链接：https://arxiv.org/abs/2303.07798
- 关键词：learned baseline, ViT, LSTM, ImageNav, ObjectNav
- 泛读摘要：OVRL-V2 使用预训练视觉表征和通用策略架构，在 ImageNav 和 ObjectNav 上取得强性能，不依赖显式检测、分割、地图或规划模块。
- 为什么入选：它代表“强 learned baseline”路线，可用于反衬 RSC-Nav 的显式、可解释、可更新记忆。
- 对 RSC-Nav 的具体作用：
  - 定义外部强 baseline 的参考。
  - 帮助回答“为什么不用纯 learned policy”。
  - 支撑 LSTM / implicit memory 对照设计。
- 下一步精读重点：
  - policy architecture。
  - 预训练视觉特征使用方式。
  - 与 ObjectNav modular 方法的比较。

### 07. Object Goal Navigation with Recursive Implicit Maps

- 链接：https://arxiv.org/abs/2308.05602
- 关键词：implicit map, transformer update, ObjectNav
- 泛读摘要：该方法用 transformer 递归更新隐式空间地图，并通过辅助任务鼓励空间和语义推理。它介于纯 learned policy 和显式地图之间。
- 为什么入选：RSC-Nav 需要说明为什么选择显式 BEV + landmark graph，而不是只用 implicit map。
- 对 RSC-Nav 的具体作用：
  - 作为 implicit memory / implicit map 近邻工作。
  - 参考递归更新和辅助任务。
  - 帮助设计 “显式 vs 隐式” 讨论。
- 下一步精读重点：
  - implicit map 更新机制。
  - 辅助任务如何提升导航。
  - 是否支持长期跨目标记忆。

### 08. Active Neural SLAM

- 链接：https://arxiv.org/abs/2004.05155
- 关键词：neural SLAM, map prediction, planning
- 泛读摘要：Active Neural SLAM 把神经地图预测和经典规划结合，用于探索和导航。它构建局部/全局地图，并选择长期目标。
- 为什么入选：RSC-Nav 的 BEV projection、map memory、waypoint / planner 可借鉴这类 modular pipeline。
- 对 RSC-Nav 的具体作用：
  - 参考 occupancy / explored map 工程接口。
  - 参考 mapping-policy-planner 解耦。
  - 支撑 A 档可运行系统。
- 下一步精读重点：
  - map builder 与 policy 的接口。
  - global / local planner 设计。
  - 训练与评估设置。

### 09. Neural Topological SLAM for Visual Navigation

- 链接：https://arxiv.org/abs/2005.12256
- 关键词：topological map, semantic node, visual navigation
- 泛读摘要：该方法构建带语义特征和粗几何关系的拓扑图，用于长程导航。它说明拓扑记忆可以比单纯局部历史更适合长期导航。
- 为什么入选：RSC-Nav 的 landmark-topological memory 与该文高度相关。
- 对 RSC-Nav 的具体作用：
  - 借鉴 node / edge / topological connectivity。
  - 支撑 Graph-only baseline。
  - 对应 H2 地标锚定。
- 下一步精读重点：
  - 节点创建与合并规则。
  - noisy actuation 下如何维护图。
  - 图如何服务 policy。

### 10. Lifelong Semantic Mapping in Indoor Environments

- 链接：https://arxiv.org/abs/2010.08846
- 关键词：lifelong semantic mapping, dynamic environments, map update
- 泛读摘要：该方向关注机器人长期运行中语义地图如何随环境变化更新，涉及动态对象、新语义发现、过期信息和地图维护。
- 为什么入选：它最直接支持 RSC-Nav 的 memory adaptive update，而不只是 memory reuse。
- 对 RSC-Nav 的具体作用：
  - 借鉴 confidence、timestamp、negative evidence。
  - 支撑 carried-stale vs carried-adaptive。
  - 帮助定义 stale memory error。
- 下一步精读重点：
  - 动态语义变化如何建模。
  - 旧地图证据如何降权或删除。
  - 有哪些可用指标。

---

## 4. 优先泛读 10 篇

### 11. Learning to Map for Active Semantic Goal Navigation

- 链接：https://arxiv.org/abs/2106.15648
- 关键词：semantic map prediction, uncertainty, ObjectNav
- 泛读摘要：该方法预测未观测区域的语义地图，并利用不确定性在探索和利用之间取舍。
- 入选理由：对 RSC-Nav 的 semantic map 和目标相关区域预测有启发，但它不是长期跨目标记忆主线。
- 重点看：semantic map 表示、不确定性如何进入决策。

### 12. Kimera: From SLAM to Spatial Perception with 3D Dynamic Scene Graphs

- 链接：https://arxiv.org/abs/1910.02490
- 关键词：3D dynamic scene graph, semantic SLAM
- 泛读摘要：Kimera 构建多层 3D scene graph，把几何、对象、场景结构组织在统一图中。
- 入选理由：RSC-Nav 的 landmark graph 可以借鉴 scene graph 的层级组织，但 B 档应保持轻量。
- 重点看：对象层、place 层和图结构，而不是重建细节。

### 13. Neural Map: Structured Memory for Deep Reinforcement Learning

- 链接：https://arxiv.org/abs/1702.08360
- 关键词：structured memory, spatial memory, DRL
- 泛读摘要：Neural Map 提出空间结构化外部记忆，证明比 LSTM 等非结构化记忆更适合部分可观测导航。
- 入选理由：为“显式结构化记忆优于隐式记忆”提供基础文献。
- 重点看：memory write / read 和 LSTM baseline 对照。

### 14. EgoMap: Projective Mapping and Structured Egocentric Memory for Deep RL

- 链接：https://arxiv.org/abs/2002.02286
- 关键词：projective memory, egocentric mapping, Deep RL
- 泛读摘要：EgoMap 将视觉特征投影到 top-down 结构化记忆中，并用自运动更新。
- 入选理由：与 RSC-Nav H1 参考系转换强相关。
- 重点看：投影机制和 ego-motion 更新。

### 15. Memory-Augmented Reinforcement Learning for Image-Goal Navigation

- 链接：https://arxiv.org/abs/2101.05181
- 关键词：episodic memory, attention retrieval, ImageNav
- 泛读摘要：该方法将历史状态编码进 episodic memory，并通过 attention 读取来导航。
- 入选理由：可借鉴 goal-conditioned retrieval，但 RSC-Nav 要把 memory 显式语义-空间化。
- 重点看：attention retrieval 如何使用历史状态。

### 16. What Does the Retrosplenial Cortex Do?

- 链接：https://www.nature.com/articles/nrn2733
- 关键词：RSC, spatial memory, landmarks, scene processing
- 泛读摘要：RSC 综述，讨论其在空间记忆、场景处理、导航和地标使用中的作用。
- 入选理由：RSC-Nav 的认知启发总入口。
- 重点看：RSC 与海马、丘脑、视觉/顶叶区域的连接关系。

### 17. Cues, Context, and Long-Term Memory

- 链接：https://www.frontiersin.org/articles/10.3389/fnhum.2014.00586/full
- 关键词：RSC, context, cues, long-term memory
- 泛读摘要：该文强调 RSC 在环境线索、上下文和长期空间记忆中的整合作用。
- 入选理由：直接支撑 H2 地标锚定和 H4 adaptive update。
- 重点看：context 和 cue 如何影响空间记忆。

### 18. Retrosplenial Cortex and Its Role in Spatial Cognition

- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC5909124/
- 关键词：RSC, spatial cognition, reference frames, landmarks
- 泛读摘要：RSC 空间认知综述，讨论参考系转换、地标使用和导航。
- 入选理由：支撑 H1、H2、H3 的认知合理性。
- 重点看：egocentric / allocentric 转换和 landmark navigation。

### 19. Retrosplenial Cortical Representations of Space and Future Goal Locations Develop with Learning

- 链接：https://www.cell.com/current-biology/fulltext/S0960-9822(19)30511-9
- 关键词：RSC, future goal, learning, spatial representation
- 泛读摘要：研究显示 RSC 中关于空间和未来目标位置的表征会随学习发展。
- 入选理由：支撑“长期记忆状态影响目标检索和 waypoint selection”。
- 重点看：goal location representation 如何随学习形成。

### 20. Understanding Memory through Hippocampal Remapping

- 链接：https://www.sciencedirect.com/science/article/pii/S0166223608001681
- 关键词：hippocampal remapping, place cells, context
- 泛读摘要：该综述总结 hippocampal remapping，说明空间表征会随环境、上下文和任务相关性变化而重组。
- 入选理由：H4 的核心神经科学来源。RSC-Nav 的 adaptive update 可以作为工程转译。
- 重点看：global remapping、rate remapping、context-induced remapping 的区别。

---

## 5. 为什么其余 30 篇暂不进入前 20

### 5.1 数据集/平台类只保留必要项

Habitat、HM3D、HM3DSEM 等很重要，但它们更像工程底座。由于本轮目标是筛出最能塑造研究问题和方法的文献，因此只保留 HM3D-OVON 作为任务/数据核心；Habitat 和 HM3D 系列放在背景文献中继续引用。

### 5.2 纯 3D 重建/SLAM 只保留代表

SemanticFusion、Fusion++、NICE-SLAM、NeuralRecon 等对语义建图有帮助，但 RSC-Nav B 档不做完整 3D 重建。它们适合作为“为什么不直接重 3D 建图”的背景，不进入最优先 20 篇。

### 5.3 通用外部记忆只保留导航相关

NTM、DNC、MERLIN 等是记忆模型的重要背景，但 RSC-Nav 不计划实现通用可微外部记忆系统。因此优先保留 Neural Map、EgoMap、Memory-Augmented RL 这类和导航更贴近的工作。

### 5.4 感知/VLM 方法暂作扩展

ZSON、CoW、ESC 等对 open-vocabulary 感知和语义先验有价值，但 RSC-Nav MVP 优先用 oracle / simulator semantic evidence 验证长期记忆机制。因此它们暂不进入最相关 20 篇。

---

## 6. 建议阅读顺序

### 第一轮：确定论文定位

1. GOAT
2. GOAT-Bench
3. HM3D-OVON
4. VLFM
5. GOSE

目标：明确 RSC-Nav 在 ObjectNav / OVON / lifelong navigation 中的位置。

### 第二轮：确定 baseline 和模块

6. OVRL-V2
7. Recursive Implicit Maps
8. Active Neural SLAM
9. Neural Topological SLAM
10. Lifelong Semantic Mapping

目标：确定显式记忆、隐式记忆、模块化地图和 adaptive update 的边界。

### 第三轮：确定认知叙事

11. What Does the Retrosplenial Cortex Do?
12. Retrosplenial Cortex and Its Role in Spatial Cognition
13. Cues, Context, and Long-Term Memory
14. Retrosplenial Representations of Space and Future Goal Locations Develop with Learning
15. Understanding Memory through Hippocampal Remapping

目标：把 H1-H4 写成计算启发，而不是生物机制复现。

### 第四轮：补模块细节

16. Learning to Map
17. Kimera
18. Neural Map
19. EgoMap
20. Memory-Augmented RL

目标：补足 semantic map、graph memory、structured memory 和 retrieval 细节。

---

## 7. 对 RSC-Nav 的直接结论

基于这 20 篇，RSC-Nav 最稳的文献定位是：

> RSC-Nav belongs to the emerging line of lifelong / multi-goal semantic navigation, but it differs from existing systems by explicitly organizing long-term semantic-spatial memory around RSC-inspired computational hypotheses and by evaluating both memory reuse and memory adaptive update.

中文表述：

> RSC-Nav 属于长期/多目标语义导航方向，但它不是单纯追求 ObjectNav 排榜，也不是重建完整 3D 地图。它的核心价值在于：用 RSC-inspired 计算假设组织长期语义-空间记忆，并通过 memory reuse 与 memory adaptive update 两个必要表现来验证长期记忆增强导航是否成立。

