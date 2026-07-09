# RSC-Nav 阶段文档索引

本目录维护各阶段的阶段性文档。项目文档和实验协议只保留主线与规划；每个阶段的实现细节、已完成证据、运行记录、可视化链接和待补事项统一放在这里。

## 文档列表

- [Phase 0: Protocol and Data Contract](phase0_protocol_data_contract.md)
- [Phase 1: Memory Core](phase1_memory_core.md)
- [Phase 2: BEV and Semantic Memory](phase2_bev_semantic_memory.md)
- [Phase 3: Landmark Retrieval](phase3_landmark_retrieval.md)
- [M2: NLMap-Style Semantic BEV Bridge](m2_nlmap_semantic_bev_bridge.md)
- [M2.5: Open-Vocabulary Grounding Adapter](m25_open_vocab_grounding_adapter.md)
- [M3: RGB Geometry Frontend](m3_rgb_geometry_frontend.md)
- [M3.5: Semantic Representation Alignment](m35_semantic_representation_alignment.md)
- [Phase 4: Context Remapping Gate](phase4_context_remapping_gate.md)
- [Phase 5: Navigation Policy](phase5_navigation_policy.md)
- [Phase 6: Main Experiments](phase6_main_experiments.md)
- [Phase 7: Context Remapping / Scene Variant](phase7_context_remapping_scene_variant.md)
- [Phase 8: Thesis Packaging](phase8_thesis_packaging.md)
- [Phase Log Audit: 2026-07-04](phase_log_audit_20260704.md)

## 当前执行顺序

当前优先顺序统一为：

```text
1. RSC-Nav Phase 3 M1
   object memory -> landmark nodes -> goal query top-k retrieval

2. VGGT-NLMap Semantic BEV Bridge Stage A/B
   NLMap-style semantic map -> BEV / semantic BEV -> RSC object memory
   then connect to Phase 3 retrieval

3. M2.5 Open-Vocabulary Grounding Adapter
   RGB / RGB-D -> semantic candidates -> M2 bridge, with Habitat oracle validation

4. VGGT-NLMap Semantic BEV Bridge Stage C
   VGGT / DUST3R from RGB sequence -> depth / pose / point cloud
   then reuse the same BEV bridge

5. M3.5 Semantic Representation Alignment
   define G/S/O/L representation contract before planner input

6. Return to later RSC-Nav phases
   context remapping / navigation policy / main experiments
```

## 维护规则

- 已完成或正在进行的阶段：记录目标、实现范围、关键文件、验证方式、产物链接、结论和遗留问题。
- 尚未正式开始的阶段：必须明确标注为“初步构想，待实现时完善”，先保留阶段目标、输入输出、计划验证和待补清单。
- 大纲性内容保留在 `RSC-Nav_项目文档_v4.md` 和 `RSC-Nav_实验协议_v4.md`；具体执行信息写入本目录。

> **最后更新时间:** 2026-07-04
