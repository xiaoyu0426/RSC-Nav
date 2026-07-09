# Phase 8: Thesis Packaging

## 阶段定位

Phase 8 的目标是把系统、协议、实验结果、可视化和失败案例整理成论文可用材料。

## 当前状态

状态：尚未正式开始，已有项目文档、实验协议和 Phase 2 可视化材料作为基础。当前内容是论文整理阶段的初步构想，待主实验结果出来后完善。

## 计划产物

- method equations。
- system diagram。
- experiment protocol section。
- main results tables。
- ablation tables。
- qualitative visual cases。
- failure analysis。
- limitations and future work。

## 当前可复用材料

- 项目文档 v4。
- 实验协议 v4。
- Phase 2 semantic BEV / A-B stress test GIF。
- negative evidence 机制说明。
- adaptive update vs context remapping 边界说明。

## 参考旧协议的论文材料

至少整理：

- BEV occupancy / explored / semantic map 随时间更新。
- landmark graph 构建过程。
- semantic confidence、freshness、last_seen_time 和 status 曲线。
- goal-to-node top-k retrieval。
- graph-to-BEV attention。
- memory reset vs memory carried 轨迹对比。
- carried-stale vs carried-adaptive 动态修正轨迹对比。
- scene_v1 vs scene_v2 memory correction case。
- 成功与失败案例分析。

失败案例至少分为：

- timeout。
- saw goal but failed stop。
- wrong object stop。
- returned to stale old location。
- stale node ranked above corrected node。
- map / projection failure。
- landmark relation failure。
- irrelevant-memory retrieval。

## 待补

- 论文图目录。
- 表格模板。
- 术语统一表。
- failure taxonomy。
- final experiment checklist。
- contribution wording finalization。
- limitations and future work section。

> **最后更新时间:** 2026-06-27
