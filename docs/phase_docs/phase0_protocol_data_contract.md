# Phase 0: Protocol and Data Contract

## 阶段定位

Phase 0 的目标是固定实验主线、数据契约和阶段边界，避免后续实现偏离“长期语义-空间记忆”这一主变量。

## 当前状态

状态：已建立主线，仍需在后续实验前补齐机器可校验 schema。

说明：已完成的是协议主线和文档边界；schema、validator、dataset spec 等尚未实现部分仍是初步构想，待实际开发时完善。

已完成：

- 项目文档 v4：明确 RSC-Nav 的研究问题、核心机制和实验主张。
- 实验协议 v4：明确 repeated-use episodes、五个 claims、冲突实验分层和阶段路线。
- 明确 same-context adaptive update 与 global context remapping 的边界。
- 明确 A->B 仅作为 global mismatch stress test，不作为环境大变的最终更新 method。
- 明确 context mismatch 后保留旧 `context_id` memory，新环境写入新的 context。

## 核心契约

主变量：

```text
是否存在可检索、可复用、可修正、可按 context 隔离的长期语义-空间记忆
```

默认仿真设置：

- Habitat-Lab / Habitat-Sim
- HM3D / MP3D
- RGB-D observation
- oracle pose / depth
- simulator semantic labels or precomputed semantic evidence

## 参考旧协议的阶段设计

Phase 0 需要冻结的内容：

- Habitat / Habitat-Sim 版本。
- 数据子集和场景选择规则。
- 动作空间。
- BEV 分辨率。
- same-scene sequential goal protocol。
- lightweight dynamic semantic-spatial update protocol。
- 语义证据来源。
- memory item 字段和 update rule。

最小产物：

- `dataset_spec.md`。
- episode 生成脚本。
- 小样本 episode 可视化。
- 至少 10 条可检查轨迹或 episode trace。

## 待补

- episode schema 的 JSON schema。
- memory item schema 的 JSON schema。
- perturbation schema。
- context schema。
- validation script。
- dataset / scene subset spec。
- seed and split policy。

## 相关文档

- `RSC-Nav_项目文档_v4.md`
- `RSC-Nav_实验协议_v4.md`

> **最后更新时间:** 2026-06-27
