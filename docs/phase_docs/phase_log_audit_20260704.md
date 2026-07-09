# Phase Log Audit: 2026-07-04

## 审计目标

检查 `docs/phase_docs/` 下每个阶段文档是否满足阶段性追溯要求：

```text
阶段目标
当前状态
已完成内容
关键脚本 / 关键文件
验证方式 / 产物路径
结论
边界与待补
```

## 总体结论

当前阶段文档体系已经基本可追溯。

已正式执行过的阶段均能追到对应脚本、输出目录、HTML/GIF/report 或关键 JSON。尚未正式实现的阶段均已标注为初步构想，并保留输入输出、评估口径和待补清单。

需要注意的是：Phase 0 / Phase 1 是较早整理的协议与核心记忆阶段，已有设计和关键文件索引，但历史执行 log 和 artifact 索引不如后续阶段完整。后续若进入论文整理，可把早期 commit / smoke test / schema validator 补成更机器可校验的记录。

## 阶段审计表

| 文档 | 状态 | 可追溯性判断 | 主要证据 | 待补 |
| --- | --- | --- | --- | --- |
| `phase0_protocol_data_contract.md` | 设计阶段已建立，schema 未实现 | 基本可追溯 | 项目/协议主线、数据契约、待补 schema 清单 | JSON schema、dataset spec、episode validator |
| `phase1_memory_core.md` | Memory MVP 已有，后续扩展待补 | 基本可追溯 | `src/object_memory_store.py`、`src/semantic_bev_memory.py`、memory fields | schema validator、serialization tests、write/retrieve/perturb smoke |
| `phase2_bev_semantic_memory.md` | 已完成 Phase2 可视化与负证据 MVP | 可追溯 | Phase2 summary page、A/B stress GIF、negative evidence 设计 | 更标准化的 run manifest 可后补 |
| `phase3_landmark_retrieval.md` | M1 已完成 | 可追溯 | object memory -> landmark nodes -> top-k retrieval，report/output paths | 后续与真实 planner/replan 进一步联动 |
| `m2_nlmap_semantic_bev_bridge.md` | M2 MVP / NLMap bridge 已完成 | 可追溯 | mock/import/NLMap-qwen3 bridge outputs、semantic BEV、retrieval report | 更多真实 NLMap 数据和 gold 对照 |
| `m25_open_vocab_grounding_adapter.md` | M2.5 已完成多轮对照 | 可追溯 | OWLv2/GroundingDINO/SAM、Habitat oracle validation、full-env coverage outputs | 细粒度 cup/bottle/sink 等 affordance class |
| `m3_rgb_geometry_frontend.md` | VGGT geometry MVP 已跑通 | 可追溯 | Habitat RGB sequence -> VGGT -> pose/depth/point cloud -> BEV | 长序列稳定性、窗口化/重叠融合 |
| `m35_semantic_representation_alignment.md` | 表示对齐已完成当前版 | 可追溯 | G/S/O/L 表示、paper-style figure、centroid metrics | 后续论文图统一排版 |
| `phase4_context_remapping_gate.md` | 尚未正式实现 | 规划可追溯 | context remapping 设计、输入输出和评估指标 | mismatch score、context manager、A->B->A 回访实验 |
| `phase5_navigation_policy.md` | Phase5A API planner + Habitat MVP 已完成 | 可追溯 | qwen3-max tests、navmesh validation、case1/case2 videos/GIFs | verify -> negative evidence -> replan，真实 affordance labels |
| `phase6_main_experiments.md` | 尚未正式开始 | 规划可追溯 | baseline / ablation / metrics 矩阵 | experiment matrix、scene/task split、统计脚本 |
| `phase7_context_remapping_scene_variant.md` | A/B stress 有基础，正式实验未开始 | 规划可追溯 | same-context variant / global remapping 设计 | object removal/relocation setup、forced-single-context baseline |
| `phase8_thesis_packaging.md` | 尚未正式开始 | 规划可追溯 | 论文图表、failure taxonomy、材料清单 | final figures/tables、术语表、failure report |

## 当前需要优先补的记录

1. Phase5A 当前新增 demo 需要保持 case registry。
   - case1: `water_then_owner_bed_20260704`
   - case2: `case2_cup_from_dining_table_20260704`
   - 两者均已使用 object-centric semantic evidence / centroids 作为联图语义面板。

2. Phase0 / Phase1 后续最好补机器可校验的 schema 和 smoke tests。
   - memory item schema
   - episode schema
   - context schema
   - write/retrieve/decay serialization tests

3. Phase4 / Phase7 需要在进入 remapping 实现前补一份更明确的 execution plan。
   - context mismatch score
   - old/new context memory isolation
   - A -> B -> A revisit protocol

4. Phase6 / Phase8 暂时保持规划态即可。
   - 当前还不需要过早写死 benchmark 表格。
   - 等 Phase5A replan loop 和 Phase4 remapping 完成后再冻结主实验矩阵。

## 审计结论

```text
active / implemented phases:
  traceable enough for continued development

future phases:
  correctly marked as preliminary design

weak spots:
  Phase0/Phase1 lack machine-checkable schema/test artifacts
  Phase4/Phase7 need execution-level design before implementation
  Phase5A still needs replan loop and richer affordance labels
```

> **最后更新时间:** 2026-07-04
