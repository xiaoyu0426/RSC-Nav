# Phase 4: Context Remapping Gate

## 阶段定位

Phase 4 的目标是判断当前观测是否仍属于当前 memory context，还是应该创建或切换到新的 context。

## 当前状态

状态：尚未正式实现。当前内容是 context remapping 的初步构想和设计边界；mismatch score、threshold、context manager API 和回访验证待实现时完善。

## 核心原则

context mismatch 后：

```text
old context:
  keep memory
  archive / downweight for current task
  remain retrievable if context is revisited

new context:
  create new context_id
  write new observations into new memory
  retrieve primarily within current context
```

也就是说，环境大变时不覆写旧环境 memory；旧环境记忆需要保留，并通过 `context_id` 隔离。

## 参考旧协议的扩展设计

旧协议只保留了 `context_id or context_embedding` 字段，v4 将其扩展为显式 remapping gate。该阶段要回答：

```text
current observation belongs to current context?
or should create / switch context?
```

可用信号：

- geometry mismatch。
- expected landmark absence。
- semantic landmark overlap。
- retrieval contradiction。
- repeated global conflict rather than local conflict。

## 计划输入

- geometry mismatch signal。
- semantic landmark overlap。
- expected landmark absence。
- current context confidence。
- retrieval mismatch / stale retrieval evidence。

## 计划输出

- current_context_id。
- context_confidence。
- remap_triggered。
- remap_reason。
- context selection result。

## 关键评估

- Remapping Trigger Accuracy。
- Context Selection Accuracy。
- Cross-Context Interference Rate。
- Old-Context Retrieval Error。
- New-Context Recovery Steps。
- 旧 context 回访时的可复用性。

## 与 A/B Stress Test 的关系

A->B 可作为 global mismatch stress visualization，但不是最终 remapping method。最终方法应显式创建或切换 context，而不是依靠旧 prior 慢慢衰减。

## 待补

- mismatch score definition。
- context manager API。
- archive / switch / retrieve policy。
- A->B with explicit context remapping experiment。
- 回访旧 context 的验证实验。
- context mismatch threshold calibration。
- forced-single-context baseline。

> **最后更新时间:** 2026-06-27
