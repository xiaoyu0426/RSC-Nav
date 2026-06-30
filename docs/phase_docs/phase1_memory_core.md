# Phase 1: Memory Core

## 阶段定位

Phase 1 的目标是建立长期记忆的核心数据结构和更新接口，为后续 BEV / semantic memory、landmark retrieval 和 context remapping 提供统一底座。

## 当前状态

状态：已有可用 MVP，后续需随 Phase 3 / Phase 4 扩展。

说明：当前 semantic/object memory 已有基础；landmark node、context-aware memory store 和完整 context manager 仍是初步构想，待对应阶段实现时完善。

已完成或已有基础：

- semantic evidence memory。
- object memory store。
- memory state: `active / stale / missing / relocated`。
- confidence / freshness / negative evidence fields。
- prior/live evidence 的来源区分。
- context 相关字段已进入文档要求，后续 Phase 4 需要实现完整 context manager。

## 核心 Memory Item

每个 semantic / landmark memory item 至少包含：

```text
id
semantic_label or semantic_embedding
bev_position
confidence
freshness
last_seen_time
visit_count
negative_evidence_count
status
context_id or context_embedding
source_view_ids
```

## 关键原则

- 同一 context 内的局部冲突走 adaptive update。
- global context mismatch 后不覆写旧 memory。
- `context_id` 是隔离和再检索旧环境记忆的主键。

## 参考旧协议的阶段设计

Phase 1 对应长期记忆核心，而不是某个单独视觉模块。最小闭环应覆盖：

```text
write
-> retrieve
-> perturb
-> stale retrieval
-> adaptive retrieval
```

需要支持的核心操作：

- write：写入 semantic / landmark memory。
- retrieve：按 goal query 检索相关 memory。
- weaken：应见未见时降低旧记忆影响。
- relocate：新位置连续确认后迁移对象或地标状态。
- overwrite：新证据稳定强于旧证据时更新 memory item。

状态字段必须被后续检索使用，而不是只记录不参与排序。

## 关键文件

- `src/object_memory_store.py`
- `src/semantic_bev_memory.py`

## 待补

- memory item schema validator。
- context-aware memory store API。
- landmark node schema。
- memory serialization compatibility tests。
- write / retrieve / perturb smoke test。
- stale vs adaptive retrieval unit test。

> **最后更新时间:** 2026-06-27
