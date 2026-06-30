# RSC-Nav Phase 2 阶段性执行文档
*(BEV / Semantic BEV / Object Memory Substrate)*

---

## 1. 阶段定位

Phase 2 的目标不是完成全部导航算法，也不是解决整体环境大变问题，而是建立后续 Phase 3 landmark retrieval 和 Phase 4 context remapping 可以依赖的稳定 semantic-spatial memory substrate。

Phase 2 在全局项目中的作用：

```text
RGB-D + pose + semantic evidence
-> BEV / semantic BEV / object memory
-> stable context for retrieval and later navigation
```

传统 BEV 继续承担实时几何导航底座；semantic BEV 和 object memory 作为后续语义检索、地标检索和长期记忆复用的上下文注入。

---

## 2. 已实现能力

当前 Phase 2 已完成 MVP：

- RGB-D / pose 到 BEV 的投影与可视化。
- occupancy / explored / semantic evidence map。
- object memory store。
- automatic episode runner。
- recorder / summary HTML / GIF artifacts。
- prior/live semantic evidence 分离。
- smooth evidence update weight。
- positive / not observable / expected-visible miss 三态证据更新。

---

## 3. 语义证据更新原则

对象记忆更新分三种情况：

```text
positive observation:
  当前确实看到了对象

not observable:
  当前没看到，但视角、高度、遮挡或覆盖条件不足以判断

expected-visible miss:
  历史对象位置理论上应可见，却连续没有看到
```

只有 expected-visible miss 才算 negative evidence。

MVP 更新直觉：

```text
positive observation:
  confidence increases
  freshness resets
  missed_observation_count clears
  status -> active

not observable:
  confidence unchanged
  freshness slowly decays
  missed count unchanged

expected-visible miss:
  confidence decreases
  freshness decays faster
  missed / negative evidence count increases
```

该机制主要解决一个问题：抬头、低头、遮挡或视锥没有覆盖对象高度时，不能把“没看到”误判为“不存在”。

---

## 4. Smooth Update Weight

证据更新不只按帧数累积，而是结合位移和转向形成平滑权重：

```text
small/no motion:
  weight is low

moderate motion / useful viewpoint change:
  weight increases

too fast / unstable motion:
  weight is capped
```

这样可以避免 agent 原地停留时置信度无限刷高，也避免快速移动时单帧证据造成过强更新。

---

## 5. Phase 2 验证结论

当前可视为已验证：

- semantic BEV 能稳定累积语义证据。
- viewpoint change 不会直接删除对象记忆。
- negative evidence 只在 expected-visible miss 下产生。
- prior/live 分离可观察。
- A->B stress test 中，旧 prior 会下降，新 live 会累积。

当前不作为 Phase 2 结论的内容：

- A->B 不是未来处理整体环境变化的最终 method。
- 大变环境应进入 Phase 4 context remapping。
- context remapping 不应覆写旧环境 memory；旧 context 应保留，新环境写入新的 context_id。
- same-context adaptive update 的正式主实验仍应使用同环境对象删除、移动或语义扰动。

---

## 6. A/B Stress Test 解释

A/B 实验目的：

```text
scene_A:
  build prior semantic memory

scene_B:
  load A as prior
  observe whether old prior weakens and new live evidence accumulates
```

它验证的是 stress condition 下的 prior/live separation 和旧 context weakening，而不是证明“一个环境能靠衰减更新成另一个环境”。

正确论文解释：

```text
same-context local change:
  adaptive update

global context change:
  context remapping
  keep old context memory
  create / switch new context

A->B:
  stress visualization only
```

---

## 7. Curated Artifacts

远端展示页：

```text
http://39.101.65.229:43901/negfix_ab_index.html?v=phase2-context-v4-A3B6
```

v4 文档：

```text
http://39.101.65.229:43901/docs_v4/RSC-Nav_实验协议_v4.html
http://39.101.65.229:43901/docs_v4/RSC-Nav_项目文档_v4.html
http://39.101.65.229:43901/docs_v4/RSC-Nav_Phase2_阶段性执行文档.html
```

长版 A3 -> B6 stress test：

```text
http://39.101.65.229:43901/phase2_curated_assets/phase2_A3pass_then_B6pass_dense_stitch.gif
http://39.101.65.229:43901/phase2_curated_assets/phase2_A3pass_then_B6pass_semantic_only.gif
http://39.101.65.229:43901/phase2_curated_assets/phase2_A3_B6_prior_live_curve.png
http://39.101.65.229:43901/phase2_curated_assets/phase2_A3_B6_report.json
```

---

## 8. 阶段收口判断

Phase 2 不需要继续扩大主功能。后续工作应切到：

```text
Phase 3:
  landmark retrieval

Phase 4:
  context remapping gate
```

Phase 2 后续只保留轻量维护：

- 清理展示页。
- 记录关键 GIF / HTML / report。
- 在论文中作为 semantic-spatial memory substrate 和 update机制可视化证据。

---

> **版本:** phase2-exec-v1  
> **最后更新时间:** 2026-06-27
