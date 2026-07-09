# Phase 7: Context Remapping / Scene Variant

## 阶段定位

Phase 7 的目标是补足环境变化相关实验，包括 same-context scene variant 和 global context remapping stress / evaluation。

## 当前状态

状态：部分已有 A->B stress visualization；正式 same-context scene variant 和 explicit context remapping 实验尚未开始。除已完成的 stress visualization 外，当前设计均为初步构想，待实现时完善。

## 已有基础

- A->B stress test 已展示 prior/live 分离和旧 context weakening。
- 已明确 A->B 不作为大变环境最终更新 method。
- 已明确 context mismatch 后保留旧 context memory，新环境写入新的 context_id。

## 计划实验

Same-context variant：

```text
scene_i_v1:
  object_a at location_x
  write memory

scene_i_v2:
  same global layout
  object_a removed / relocated / replaced
  update local memory
```

Global remapping：

```text
scene_A:
  write context_A

scene_B:
  detect mismatch
  create / switch context_B
  keep context_A memory
```

## 参考旧协议的 scene variant 设计

推荐变化：

- target object relocated。
- target object removed。
- landmark-related object relation changed。
- distractor object inserted near old location。

控制变量：

- same base scene。
- same task sequence。
- same start pose when feasible。
- same max steps。
- same success radius。
- same action space。
- same perception source。
- same planner / policy。

核心对照：

```text
carried-stale on scene_i_v2
vs
carried-adaptive on scene_i_v2
vs
memory-reset on scene_i_v2
```

论文表述应是 controlled simulation scene-variant reconfiguration，不表述为真实动态世界部署。

## 关键评估

- same-context adaptive correction latency。
- stale old-location stop rate。
- remapping trigger accuracy。
- context selection accuracy。
- cross-context interference。
- return-to-old-context reuse。

## 待补

- same-scene object removal / relocation setup。
- explicit context remapping implementation。
- forced-single-context vs context-remapping comparison。
- A -> B -> A revisit experiment。
- scene_i_v1 / scene_i_v2 construction notes。
- relation-change visual case。

> **最后更新时间:** 2026-06-27
