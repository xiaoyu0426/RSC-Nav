# RSC-Nav 相关文献 50 篇摘要级整理

> 整理日期：2026-06-15  
> 整理方式：基于论文题目、摘要、项目页和公开元信息的非精读归纳。  
> 用途：为 RSC-Nav 的“长期语义-空间记忆增强导航”主线建立文献入口。  

---

## 0. 与 RSC-Nav 的关系框架

RSC-Nav 的唯一主线是：

> 长期语义-空间记忆增强导航。

该长期记忆能力有两个必要表现：

1. **Memory reuse**：过去探索到的空间、语义和地标信息能被后续目标复用。
2. **Memory adaptive update**：旧语义-地标记忆能被新观测确认、削弱、迁移或覆盖，避免 stale memory 误导导航。

因此，文献整理按以下六类组织：

- 仿真环境、数据集与评测协议。
- ObjectNav / OVON / 多目标导航。
- 模块化地图、语义地图与 frontier 探索。
- 长期记忆、实例记忆与动态语义地图。
- 外部记忆、结构化神经记忆与导航策略。
- RSC、海马体 remapping 与认知启发。

---

## 1. 仿真环境、数据集与评测协议

### 01. Habitat: A Platform for Embodied AI Research

- 年份：2019
- 方向：具身 AI 仿真平台
- 链接：https://arxiv.org/abs/1904.01201
- 摘要级归纳：Habitat 提供高效、可复现的 3D 室内导航仿真环境，支持 PointNav、ObjectNav 等任务，是后续 Habitat-Lab / Habitat-Sim 生态的基础。它强调标准化任务接口、传感器、动作空间、评估指标和大规模并行仿真。
- 对 RSC-Nav 的启发：RSC-Nav 不需要重写仿真底座，应优先继承 Habitat 的环境接口、episode 管理和评估体系。

### 02. Habitat 2.0: Training Home Assistants to Rearrange their Habitat

- 年份：2021
- 方向：交互式具身仿真、重排任务
- 链接：https://arxiv.org/abs/2106.14405
- 摘要级归纳：Habitat 2.0 扩展到具备物理交互、可操作物体和家庭助理任务的仿真环境，并引入 ReplicaCAD 和 HAB benchmark。它说明 Habitat 生态可支持更复杂的动态与交互任务。
- 对 RSC-Nav 的启发：B 档不做完整交互式动态 3D SLAM，但可将动态对象扰动作为轻量协议；C 档可参考 Habitat 2.0 的交互能力。

### 03. Habitat-Matterport 3D Dataset (HM3D): 1000 Large-scale 3D Environments for Embodied AI

- 年份：2021
- 方向：大规模室内 3D 数据集
- 链接：https://arxiv.org/abs/2109.08238
- 摘要级归纳：HM3D 提供 1000 个真实室内环境的高质量 3D 重建，规模、视觉质量和导航空间都超过许多早期数据集。论文显示在 HM3D 上训练的导航 agent 具有较强跨数据集泛化能力。
- 对 RSC-Nav 的启发：HM3D 是 RSC-Nav 的主要场景来源，适合构造 same-scene sequential goal protocol。

### 04. Habitat-Matterport 3D Semantics Dataset

- 年份：2022
- 方向：HM3D 语义标注
- 链接：https://arxiv.org/abs/2210.05633
- 摘要级归纳：HM3DSEM 为 HM3D 场景提供密集对象实例和房间语义标注，支持 ObjectNav 等任务。它提供了大量对象实例、类别和空间语义信息。
- 对 RSC-Nav 的启发：MVP 阶段可使用 HM3DSEM / Habitat semantic sensor 作为语义证据来源，先验证记忆机制而不是感知模型。

### 05. Matterport3D: Learning from RGB-D Data in Indoor Environments

- 年份：2017
- 方向：室内 RGB-D 扫描数据集
- 链接：https://arxiv.org/abs/1709.06158
- 摘要级归纳：Matterport3D 提供大规模室内扫描、RGB-D 视角和语义信息，是早期视觉导航、ObjectNav 和 3D 语义理解的重要数据来源。
- 对 RSC-Nav 的启发：MP3D 是许多导航 baseline 的历史评测场景，可作为对比文献背景，不作为首选主数据集。

### 06. Gibson Env: Real-World Perception for Embodied Agents

- 年份：2018
- 方向：真实扫描环境导航仿真
- 链接：https://arxiv.org/abs/1808.10654
- 摘要级归纳：Gibson 提供真实世界 3D 扫描环境，支持具身 agent 在逼真室内场景中学习视觉导航。它推动了从简单迷宫到真实空间的导航评测。
- 对 RSC-Nav 的启发：Gibson 常用于 PointNav / ImageNav baseline，可作为方法迁移和泛化背景。

### 07. The Replica Dataset: A Digital Replica of Indoor Spaces

- 年份：2019
- 方向：高保真室内 3D 数据集
- 链接：https://arxiv.org/abs/1906.05797
- 摘要级归纳：Replica 提供高质量室内场景重建，支持语义分割、导航和重建任务。其视觉质量适合评估感知与几何模块。
- 对 RSC-Nav 的启发：Replica 可作为 C 档跨数据集或可视化验证场景，但主线仍建议聚焦 HM3D / HM3D-OVON。

### 08. AI2-THOR: An Interactive 3D Environment for Visual AI

- 年份：2017
- 方向：交互式室内仿真
- 链接：https://arxiv.org/abs/1712.05474
- 摘要级归纳：AI2-THOR 提供可交互家庭场景，支持对象操作、导航和视觉问答等任务。它强调对象状态变化和交互动作。
- 对 RSC-Nav 的启发：动态语义对象变化可从交互仿真思路获得启发，但 B 档不需要迁移到 AI2-THOR。

---

## 2. ObjectNav、OVON 与多目标导航

### 09. Object Goal Navigation using Goal-Oriented Semantic Exploration

- 年份：2020
- 方向：ObjectNav、语义地图、模块化探索
- 链接：https://arxiv.org/abs/2007.00643
- 摘要级归纳：GOSE / SemExp 提出构建 episodic semantic map，并根据目标类别学习语义先验以选择探索方向。它在 Habitat ObjectNav Challenge 中表现强，是模块化 ObjectNav 的代表。
- 对 RSC-Nav 的启发：BEV semantic map 和目标导向探索可借鉴，但 RSC-Nav 进一步关注同场景连续目标下长期记忆复用和 adaptive update。

### 10. GOAT: GO to Any Thing

- 年份：2023
- 方向：多模态目标、长期导航、实例语义记忆
- 链接：https://arxiv.org/abs/2311.06430
- 摘要级归纳：GOAT 是一个可处理类别、图像和语言目标的通用导航系统，强调 lifelong experience 和 continually augmented instance-aware semantic memory。它在真实家庭实验中显示随着经验增加，导航成功率提高。
- 对 RSC-Nav 的启发：GOAT 是长期记忆增强导航的重要近邻；RSC-Nav 可借鉴实例级语义记忆，但论文主张应聚焦 RSC-inspired adaptive semantic-spatial memory。

### 11. GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation

- 年份：2024
- 方向：多目标、多模态、lifelong navigation benchmark
- 链接：https://arxiv.org/abs/2404.06609
- 摘要级归纳：GOAT-Bench 将目标指定扩展到类别、语言描述和图像，并要求 agent 在序列目标中导航。论文分析显式/隐式 scene memory、目标噪声鲁棒性和 lifelong 场景中的记忆影响。
- 对 RSC-Nav 的启发：B 档 same-scene sequential goal protocol 与 GOAT-Bench 同向，但应控制工作量；C 档可向 GOAT-Bench 靠近。

### 12. HM3D-OVON: A Dataset and Benchmark for Open-Vocabulary Object Goal Navigation

- 年份：2024
- 方向：开放词汇 ObjectNav 数据集
- 链接：https://arxiv.org/abs/2409.14296
- 摘要级归纳：HM3D-OVON 基于 HM3DSEM，提供大量开放词汇对象类别和实例，用于训练和评估自由文本目标 ObjectNav。论文比较多类方法，并展示开放词汇 ObjectNav 的训练可行性。
- 对 RSC-Nav 的启发：RSC-Nav B 档可直接基于 HM3D-OVON 子集构建目标序列和语义证据。

### 13. VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation

- 年份：2023
- 方向：VLM、frontier exploration、zero-shot ObjectNav
- 链接：https://arxiv.org/abs/2312.03275
- 摘要级归纳：VLFM 用深度构建 occupancy map 和 frontier，再用视觉语言模型生成 language-grounded value map，选择最有希望找到目标的 frontier。方法 zero-shot，并在多个数据集和真实机器人上表现强。
- 对 RSC-Nav 的启发：VLFM 是强模块化 baseline；RSC-Nav 可借鉴 frontier/value map，但核心区别是长期语义-空间记忆的复用与 adaptive update。

### 14. OVRL-V2: A Simple State-of-Art Baseline for ImageNav and ObjectNav

- 年份：2023
- 方向：学习式 ObjectNav / ImageNav 强 baseline
- 链接：https://arxiv.org/abs/2303.07798
- 摘要级归纳：OVRL-V2 使用预训练视觉表征、ViT patch compression、卷积和 LSTM policy，在 ImageNav 和 ObjectNav 上取得强结果。它强调无需显式检测、分割、地图或规划模块。
- 对 RSC-Nav 的启发：OVRL-V2 可作为强 learned baseline；RSC-Nav 应强调可解释、可更新的显式长期记忆，而不是纯隐式策略表现。

### 15. ZSON: Zero-Shot Object-Goal Navigation using Multimodal Goal Embeddings

- 年份：2022
- 方向：zero-shot ObjectNav、多模态 goal embedding
- 链接：https://arxiv.org/abs/2206.12403
- 摘要级归纳：ZSON 通过图像目标导航训练 agent，并把语言目标投影到同一多模态语义空间，从而实现零样本 ObjectNav。它降低对 ObjectNav 标注和奖励的依赖。
- 对 RSC-Nav 的启发：目标文本编码可借鉴冻结多模态 embedding；但 RSC-Nav 的主要创新不在语言泛化，而在长期记忆增强。

### 16. Object Goal Navigation with Recursive Implicit Maps

- 年份：2023
- 方向：隐式空间地图、ObjectNav
- 链接：https://arxiv.org/abs/2308.05602
- 摘要级归纳：该方法用 transformer 递归更新 implicit spatial map，并通过辅助任务重建显式地图、视觉特征和语义标签，以增强 ObjectNav 的空间推理。
- 对 RSC-Nav 的启发：说明地图可以是显式或隐式的；RSC-Nav 选择显式 BEV + landmark graph，是为了可解释和可消融。

### 17. PONI: Potential Functions for ObjectGoal Navigation with Interaction-free Learning

- 年份：2022
- 方向：ObjectNav、潜在函数、无需交互训练
- 链接：https://arxiv.org/abs/2201.06840
- 摘要级归纳：PONI 学习 object-goal navigation 的潜在函数，用于在地图上预测目标相关区域和导航价值，减少昂贵的交互式 RL 训练。
- 对 RSC-Nav 的启发：目标条件 BEV 区域评分可借鉴；RSC-Nav 可将 graph-to-BEV attention 理解为一种记忆检索后的区域价值分配。

### 18. ESC: Exploration with Soft Commonsense Constraints for Zero-shot Object Navigation

- 年份：2023
- 方向：常识约束、zero-shot ObjectNav
- 链接：https://arxiv.org/abs/2301.13166
- 摘要级归纳：ESC 利用大模型或常识知识引导探索，使 agent 在没有专门训练的情况下更合理地寻找目标对象。它强调对象与房间/场景之间的常识约束。
- 对 RSC-Nav 的启发：可作为语义先验或目标条件检索参考；但 RSC-Nav 的核心记忆来自场景内亲身探索，而不是只依赖外部常识。

### 19. CoW: Catching Objects in the Wild for Object Goal Navigation

- 年份：2023
- 方向：open-vocabulary perception、ObjectNav
- 链接：https://arxiv.org/abs/2303.11718
- 摘要级归纳：CoW 使用开放词汇检测和视觉语言模型帮助 agent 在开放环境中寻找对象，强调感知模块在 ObjectNav 中的重要性。
- 对 RSC-Nav 的启发：可作为 C 档替换 oracle semantic evidence 的感知来源，但不应在 MVP 中抢占记忆机制验证。

### 20. LagMemo: Language 3D Gaussian Splatting Memory for Multi-modal Open-vocabulary Multi-goal Visual Navigation

- 年份：2025
- 方向：3D language memory、多目标开放词汇导航
- 链接：https://arxiv.org/abs/2510.24118
- 摘要级归纳：LagMemo 构建 language 3D Gaussian Splatting memory，用于多模态开放词汇多目标导航。系统通过探索建立统一 3D 语言记忆，并在目标到来时查询、验证和导航。
- 对 RSC-Nav 的启发：代表“更重的 3D 语言记忆”路线；RSC-Nav 可在论文中强调轻量 BEV + landmark graph 的可控性和 adaptive update。

### 21. EvoMemNav: Efficient Self-Evolving Fine-Grained Memory for Zero-Shot Embodied Navigation

- 年份：2026
- 方向：自演化细粒度记忆、zero-shot embodied navigation
- 链接：https://arxiv.org/abs/2606.03509
- 摘要级归纳：EvoMemNav 构建 Visual-Semantic Memory Graph，将 raw views、语义线索和拓扑关系组织成层级记忆，并通过反思式 write-back 更新经验。它关注细粒度记忆、预算化检索和多目标导航。
- 对 RSC-Nav 的启发：该方向与 RSC-Nav 的 adaptive memory 很近；RSC-Nav 应突出 RSC/海马 remapping 启发和 carried-stale vs carried-adaptive 机制验证。

---

## 3. 模块化地图、语义地图与 frontier 探索

### 22. Active Neural SLAM

- 年份：2020
- 方向：可学习 SLAM、探索与导航
- 链接：https://arxiv.org/abs/2004.05155
- 摘要级归纳：Active Neural SLAM 将神经网络地图预测和经典规划结合，用于探索和 PointNav。系统构建局部/全局地图，并选择长期目标指导探索。
- 对 RSC-Nav 的启发：RSC-Nav 的 BEV memory 和 waypoint policy 可参考其 modular navigation pipeline。

### 23. Neural Topological SLAM for Visual Navigation

- 年份：2020
- 方向：拓扑地图、语义节点、ImageNav
- 链接：https://arxiv.org/abs/2005.12256
- 摘要级归纳：该方法构建带有语义特征和粗几何关系的拓扑图，用于长程视觉导航。它强调拓扑表示在含噪动作下的鲁棒性。
- 对 RSC-Nav 的启发：landmark-topological memory 可借鉴其节点/边设计，但 RSC-Nav 需要增加 confidence、freshness 和 status。

### 24. Learning to Map for Active Semantic Goal Navigation

- 年份：2021
- 方向：主动语义地图预测、ObjectNav
- 链接：https://arxiv.org/abs/2106.15648
- 摘要级归纳：该方法学习预测视野外的语义地图，并利用未观测区域的不确定性选择探索目标。它通过空间预测学习室内语义先验。
- 对 RSC-Nav 的启发：semantic map 不只记录已见内容，也可以预测未知区域；RSC-Nav MVP 可先只做已观测证据，C 档再加入预测。

### 25. Occupancy Anticipation for Efficient Exploration and Navigation

- 年份：2020
- 方向：occupancy map 预测、探索
- 链接：https://arxiv.org/abs/2008.09285
- 摘要级归纳：Occupancy Anticipation 让 agent 从 RGB-D 观测推断不可见区域的 occupancy，从而更快建立空间意识并提升探索效率。
- 对 RSC-Nav 的启发：可作为 BEV geometry memory 的增强方向；主线先保持 deterministic / log-odds map。

### 26. SemanticFusion: Dense 3D Semantic Mapping with Convolutional Neural Networks

- 年份：2017
- 方向：3D 语义地图融合
- 链接：https://arxiv.org/abs/1609.05130
- 摘要级归纳：SemanticFusion 将 CNN 语义预测融合进 dense SLAM 地图，形成一致的 3D semantic map。它是语义 SLAM 的经典方法之一。
- 对 RSC-Nav 的启发：semantic confidence fusion 和多帧证据整合可借鉴；但 RSC-Nav B 档不做完整 dense 3D SLAM。

### 27. Fusion++: Volumetric Object-Level SLAM

- 年份：2018
- 方向：对象级 SLAM、语义实例地图
- 链接：https://arxiv.org/abs/1808.08378
- 摘要级归纳：Fusion++ 将对象检测、实例分割和体素融合结合，构建对象级地图。它强调对象实例作为地图中的一等公民。
- 对 RSC-Nav 的启发：landmark node 可以借鉴对象级地图思想；但 RSC-Nav 更轻量，并关心目标条件检索。

### 28. Kimera: From SLAM to Spatial Perception with 3D Dynamic Scene Graphs

- 年份：2019
- 方向：3D 动态场景图、语义 SLAM
- 链接：https://arxiv.org/abs/1910.02490
- 摘要级归纳：Kimera 构建从度量几何到语义对象、场景图的多层空间感知系统，支持动态场景图表示。
- 对 RSC-Nav 的启发：RSC-Nav 的 landmark graph 可以视作轻量、导航导向的语义-拓扑图，不追求完整 3D scene graph。

### 29. Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction

- 年份：2022
- 方向：实时 3D scene graph、空间感知
- 链接：https://arxiv.org/abs/2201.13360
- 摘要级归纳：Hydra 实时构建多层 3D scene graph，包含 places、objects、rooms 等语义结构，用于机器人空间理解。
- 对 RSC-Nav 的启发：room / region goal 和层级图结构可作为 C 档扩展；B 档先保留 keyframe + landmark。

### 30. NICE-SLAM: Neural Implicit Scalable Encoding for SLAM

- 年份：2022
- 方向：神经隐式 SLAM
- 链接：https://arxiv.org/abs/2112.12130
- 摘要级归纳：NICE-SLAM 使用层级神经隐式表示进行可扩展室内重建和跟踪。它代表较重的神经 3D 建图路线。
- 对 RSC-Nav 的启发：可用于论证为什么 RSC-Nav 不选择完整 3D implicit reconstruction 作为 B 档核心。

### 31. NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video

- 年份：2021
- 方向：学习式 3D 重建
- 链接：https://arxiv.org/abs/2104.00681
- 摘要级归纳：NeuralRecon 从单目视频实时预测体素 TSDF，完成稠密 3D 重建。它展示学习式重建可用于空间感知。
- 对 RSC-Nav 的启发：3D 重建能力强但工程重；RSC-Nav 应说明 B 档选择 BEV memory 是研究记忆机制而非追求重建精度。

### 32. Lifelong Semantic Mapping in Indoor Environments

- 年份：2020
- 方向：长期语义地图、动态环境
- 链接：https://arxiv.org/abs/2010.08846
- 摘要级归纳：lifelong semantic mapping 关注机器人长期运行中如何更新语义地图、处理新对象和变化环境。它通常涉及语义迁移、动态对象和地图维护。
- 对 RSC-Nav 的启发：confidence、timestamp、negative evidence 和 stale status 可以从 lifelong semantic mapping 中借鉴。

### 33. Long-term Visual Localization Revisited

- 年份：2019
- 方向：长期视觉定位、环境变化
- 链接：https://arxiv.org/abs/1912.01207
- 摘要级归纳：长期视觉定位研究跨季节、光照、视角和环境变化下的定位鲁棒性，强调视觉记忆不能假设环境静态不变。
- 对 RSC-Nav 的启发：支持 RSC-Nav 引入 freshness / adaptive update，而不是只保留静态地图。

---

## 4. 外部记忆、结构化神经记忆与导航策略

### 34. Neural Map: Structured Memory for Deep Reinforcement Learning

- 年份：2017
- 方向：结构化外部记忆、深度强化学习
- 链接：https://arxiv.org/abs/1702.08360
- 摘要级归纳：Neural Map 提出 2D 空间结构化 memory image，让 DRL agent 在部分可观测环境中存储长期信息，优于简单 LSTM 或短历史堆叠。
- 对 RSC-Nav 的启发：支持“空间结构化记忆优于非结构化隐状态”的基本动机。

### 35. EgoMap: Projective Mapping and Structured Egocentric Memory for Deep RL

- 年份：2020
- 方向：egocentric structured memory、DRL
- 链接：https://arxiv.org/abs/2002.02286
- 摘要级归纳：EgoMap 将 CNN 特征投影到 top-down map，并用 ego-motion 更新结构化记忆，在 3D 任务上优于 recurrent agent。
- 对 RSC-Nav 的启发：与 H1 参考系转换相关；RSC-Nav 进一步把记忆扩展到语义、地标和 adaptive status。

### 36. Memory-Augmented Reinforcement Learning for Image-Goal Navigation

- 年份：2021
- 方向：episodic memory、ImageNav
- 链接：https://arxiv.org/abs/2101.05181
- 摘要级归纳：该方法将历史访问状态嵌入 episodic memory，并通过 attention 读取，用于 RGB-only image-goal navigation。
- 对 RSC-Nav 的启发：goal-conditioned memory retrieval 可借鉴 attention 读记忆思想；RSC-Nav 的区别是记忆显式语义-空间化。

### 37. MERLIN: Unsupervised Predictive Memory in a Goal-Directed Agent

- 年份：2018
- 方向：预测式记忆、目标导向 agent
- 链接：https://arxiv.org/abs/1803.10760
- 摘要级归纳：MERLIN 将记忆形成与预测建模结合，在严重部分可观测的 3D 任务中学习长期记忆。它强调“记忆格式”对长期任务成功至关重要。
- 对 RSC-Nav 的启发：支持 RSC-Nav 将记忆设计成可检索、可更新的结构，而不是简单 LSTM hidden state。

### 38. Neural Turing Machines

- 年份：2014
- 方向：可微外部记忆
- 链接：https://arxiv.org/abs/1410.5401
- 摘要级归纳：Neural Turing Machine 将神经网络控制器与可读写外部 memory 结合，引入基于 attention 的 differentiable read/write。
- 对 RSC-Nav 的启发：可读写记忆的思想可作为 learned memory update gate 的远期背景，但 B 档无需实现通用 NTM。

### 39. Differentiable Neural Computers

- 年份：2016
- 方向：动态外部记忆、可微计算
- 链接：https://www.nature.com/articles/nature20101
- 摘要级归纳：DNC 扩展 NTM，提供更强的外部 memory allocation、temporal links 和 content-based retrieval，能处理图结构和长程依赖任务。
- 对 RSC-Nav 的启发：动态 read/write/erase 是 adaptive memory 的高层思想来源；实现上应转化为轻量规则或小型 gate。

### 40. Episodic Curiosity through Reachability

- 年份：2019
- 方向：episodic memory、探索奖励
- 链接：https://arxiv.org/abs/1810.02274
- 摘要级归纳：该方法使用 episodic memory 判断当前状态是否新颖，并将“不可达/新颖”状态作为探索奖励，缓解稀疏奖励环境中的探索问题。
- 对 RSC-Nav 的启发：visit_count、explored memory 和 revisit success 可借鉴 episodic novelty 的思想。

### 41. Generalization of Reinforcement Learners with Working and Episodic Memory

- 年份：2018
- 方向：工作记忆、情节记忆、RL 泛化
- 链接：https://arxiv.org/abs/1805.12018
- 摘要级归纳：该研究比较工作记忆和情节记忆对 RL 泛化的帮助，强调不同记忆形式服务不同时间尺度的决策。
- 对 RSC-Nav 的启发：LSTM baseline 可视为工作记忆，RSC-Nav 的 BEV / landmark memory 更接近外部长期情节-语义记忆。

---

## 5. RSC、海马体 remapping 与认知启发

### 42. What Does the Retrosplenial Cortex Do?

- 年份：2009
- 方向：RSC 综述
- 链接：https://www.nature.com/articles/nrn2733
- 摘要级归纳：该综述总结 RSC 在空间记忆、场景处理、导航、情景记忆和地标使用中的作用，强调其与海马、丘脑、视觉/顶叶区域的连接。
- 对 RSC-Nav 的启发：为 RSC 作为感知证据、地标和长期空间记忆枢纽提供认知基础。

### 43. Cues, Context, and Long-Term Memory: The Role of the Retrosplenial Cortex in Spatial Cognition

- 年份：2014
- 方向：RSC、线索、语境和长期空间记忆
- 链接：https://www.frontiersin.org/articles/10.3389/fnhum.2014.00586/full
- 摘要级归纳：该文强调 RSC 在环境线索、上下文和长期空间记忆之间的整合作用，尤其适合解释地标与目标导向导航之间的关系。
- 对 RSC-Nav 的启发：支持 H2 地标锚定和 H4 context-dependent adaptive update。

### 44. Retrosplenial Cortex and Its Role in Spatial Cognition

- 年份：2018
- 方向：RSC 空间认知综述
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC5909124/
- 摘要级归纳：该综述讨论 RSC 在参考系转换、地标使用、空间定位和导航中的作用，并强调 RSC 与海马系统及视觉空间处理网络的联系。
- 对 RSC-Nav 的启发：H1 参考系转换、H2 地标锚定和 H3 目标条件检索均可从该综述获得支持。

### 45. Retrosplenial Cortex Maps the Conjunction of Internal and External Spaces

- 年份：2015
- 方向：RSC 表征、自身运动与外部空间
- 链接：https://www.nature.com/articles/nn.4058
- 摘要级归纳：研究显示 RSC 神经元编码内部运动状态和外部空间线索的结合，说明 RSC 不只是静态地标区，而是整合自我运动与环境参照的区域。
- 对 RSC-Nav 的启发：支持将 egocentric observation + pose 转换到 allocentric memory 的 H1。

### 46. Retrosplenial Cortical Representations of Space and Future Goal Locations Develop with Learning

- 年份：2019
- 方向：RSC、目标位置、学习过程
- 链接：https://www.cell.com/current-biology/fulltext/S0960-9822(19)30511-9
- 摘要级归纳：研究表明 RSC 中关于空间和未来目标位置的表征会随着学习逐渐发展，提示 RSC 参与目标导向空间记忆的形成。
- 对 RSC-Nav 的启发：支持 goal-conditioned retrieval 和 waypoint selection 受长期记忆状态影响。

### 47. Understanding Memory through Hippocampal Remapping

- 年份：2008
- 方向：海马 place cell remapping 综述
- 链接：https://www.sciencedirect.com/science/article/pii/S0166223608001681
- 摘要级归纳：该综述总结 hippocampal remapping，即 place cell 表征会随环境、上下文或任务变化重组。它区分 global remapping、rate remapping 等现象。
- 对 RSC-Nav 的启发：H4 的核心灵感来源；RSC-Nav 将其转译为语义-空间记忆的 confirm / weaken / relocate / overwrite。

### 48. Place Cells, Spatial Maps and the Population Code for Memory

- 年份：2005
- 方向：海马 place cells、空间地图、记忆编码
- 链接：https://www.sciencedirect.com/science/article/pii/S0959438805001378
- 摘要级归纳：该综述讨论 place cells 如何形成空间地图，以及群体编码如何支持情景和空间记忆。
- 对 RSC-Nav 的启发：为“空间记忆不是简单坐标表，而是可被经验塑造的表征”提供背景。

### 49. The Hippocampus as a Cognitive Map

- 年份：1978
- 方向：认知地图理论
- 链接：https://global.oup.com/academic/product/the-hippocampus-as-a-cognitive-map-9780198572060
- 摘要级归纳：O'Keefe 和 Nadel 提出海马作为 cognitive map 的经典理论，奠定了空间记忆和导航研究的基础。
- 对 RSC-Nav 的启发：RSC-Nav 的 allocentric BEV memory 可视为工程层面的认知地图近似，但加入了语义与地标状态。

### 50. Hippocampal Remapping and Its Entorhinal Origin

- 年份：2018
- 方向：海马 remapping、内嗅皮层来源
- 链接：https://www.frontiersin.org/articles/10.3389/fnbeh.2017.00253/full
- 摘要级归纳：该文综述 hippocampal remapping 及其与内嗅皮层输入的关系，强调空间表征会根据环境和上下文变化而重组。
- 对 RSC-Nav 的启发：支持 memory adaptive update 作为长期语义-空间记忆能力的必要表现，而不是附加功能。

---

## 6. 初步综合判断

### 6.1 当前研究现状

ObjectNav / OVON 领域已经形成三条主流路线：

1. **模块化地图路线**：如 GOSE、VLFM、PONI，通过 occupancy / semantic map、frontier 或 value map 实现可解释探索。
2. **学习式强 baseline 路线**：如 OVRL-V2、RIM，用预训练视觉特征、Transformer / LSTM 和策略训练获得强性能。
3. **长期 / 多目标记忆路线**：如 GOAT、GOAT-Bench、LagMemo、EvoMemNav，关注同一环境中跨目标经验积累和实例记忆。

RSC-Nav 应站在第三条路线中，但借鉴第一条路线的可解释地图和第二条路线的训练工程。

### 6.2 RSC-Nav 的差异化空间

RSC-Nav 不应只宣称“建了一张语义地图”，否则容易被 3D SLAM、GOSE、VLFM 或 language 3D memory 方法覆盖。更清晰的差异化是：

- 用 RSC-inspired 假设组织长期语义-空间记忆。
- 将 BEV memory、semantic memory、landmark-topological memory 与 adaptive memory state 统一。
- 通过 `memory reset vs memory carried` 验证 memory reuse。
- 通过 `carried-stale vs carried-adaptive` 验证 memory adaptive update。
- 强调长期记忆增强导航，而不是单目标 ObjectNav 排榜。

### 6.3 对实现路线的建议

- MVP 优先使用 Habitat semantic sensor / oracle semantic evidence。
- BEV / semantic map / landmark graph 先规则实现。
- adaptive update 先做规则版：confirm、weaken、relocate、overwrite。
- 学习模块集中在 retrieval、graph-to-BEV attention、waypoint / stop。
- 强 baseline 用作参照，不要让论文主线变成“追求最高 ObjectNav SPL”。

