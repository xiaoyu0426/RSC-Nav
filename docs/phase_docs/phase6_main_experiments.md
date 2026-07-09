# Phase 6: Main Experiments

## 阶段定位

Phase 6 的目标是完成论文主实验和消融，系统性验证 memory reuse、landmark retrieval、long-horizon retention 和 adaptive update。

## 当前状态

状态：尚未正式开始。当前内容是主实验矩阵和评估口径的初步构想；具体场景划分、任务生成、统计脚本和表格模板待实验实现时完善。

## 主对照

```text
memory reset
vs
memory carried

BEV-only carried
vs
RSC-full carried
vs
RSC-full carried without landmark retrieval

carried-stale
vs
carried-adaptive
vs
memory-reset
```

## 参考旧协议的 baseline 设计

硬 baseline：

- Reactive / frontier-only。
- LSTM / implicit memory。
- BEV-only reset。
- BEV-only carried。
- Graph-only memory。
- RSC-full reset。
- RSC-full carried。
- RSC-full carried-stale。
- RSC-full carried-adaptive。

关键消融：

- without landmark retrieval。
- without freshness penalty。
- without status penalty。
- without negative evidence。
- random retrieval。
- recency-only retrieval。
- BEV-only。
- graph-only。

## 关键指标

- Success Rate。
- SPL。
- Path Length。
- Retrieval Hit@K。
- Anchor Hit@K。
- Long-Horizon Memory Retention。
- Stale Memory Error Rate。
- Memory Reuse Gain。

## 统计与可视化要求

所有主实验优先使用 paired comparison，并报告：

- number of scenes。
- number of episode groups。
- number of tasks。
- number of target categories。
- random seeds。
- mean。
- standard deviation or 95% confidence interval。

每组主实验至少输出：

- BEV occupancy / explored / semantic map。
- landmark graph。
- goal-to-node top-k retrieval。
- retrieval score decomposition。
- confidence / freshness / status curve。
- reset vs carried trajectory。
- stale vs adaptive trajectory。
- failure case report。

## 待补

- experiment matrix。
- scene / task split。
- seed policy。
- reporting script。
- tables and visual cases。
- paired statistical test script。
- failure taxonomy implementation。

> **最后更新时间:** 2026-06-27
