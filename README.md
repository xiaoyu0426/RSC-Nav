# RSC-Nav

Reusable semantic-spatial memory for language-conditioned embodied navigation.

RSC-Nav studies a map-then-task navigation loop: an agent first builds reusable object-centric semantic memory, then an API planner selects executable semantic waypoints from that memory, and execution feedback can update confidence, freshness, and future replanning decisions.

## Demo

### Case: explore from an empty map and find all cups

![RSC-Nav zero-map cup-search demo](outputs/phase5a_sim_demo/zero_map_find_all_cups_hm3d_20260723/final_report/zero_map_find_all_cups.gif)

The GIF shows:

- first-person RGB with GroundingDINO boxes, labels, and confidence scores;
- the live depth observation used for 3D projection;
- a dynamic metric BEV with the explored route and stable object tracks;
- active tabletop scans and multi-view object-memory confirmation.

This run starts with empty map and object memory. Across 523 RGB-D observations,
it projects 1,719 open-vocabulary detections into 3D, retains 19 stable semantic
tracks, and confirms four cup tracks from multiple views. Habitat oracle labels
are used only for post-hoc coverage auditing, not for exploration or memory
updates.

The full public result index is available at:

```text
outputs/paper_public_index.html
```

## Current Position

Current Habitat demos use simulator exact pose for depth projection and semantic evidence accumulation. This validates the semantic-memory-to-planner-to-execution loop, not a full real-world localization system.

The real-world roadmap explicitly includes:

- interest-driven exploration instead of scripted coverage loops;
- Lingbo vision/depth model trials;
- real computed localization with SLAM/VIO/wheel odometry/relocalization uncertainty;
- API planner demo refinement with target verification, negative evidence, memory update, and replanning.

## Core Idea

RSC-Nav is not just another VLN planner or BEV mapping baseline. The main contribution target is:

```text
RGB-D / semantic observations
-> object-centric reusable semantic memory
-> memory-grounded API planner
-> semantic waypoint / stopover execution
-> observe / update / replan-ready loop
```

The memory stores object category, spatial position, confidence, freshness, positive evidence, expected-visible misses, stale/missing status, and planner-facing waypoint candidates.

## Repository Layout

```text
src/
  dense_bev_mapper.py              # metric BEV occupancy / depth projection
  semantic_bev_memory.py           # semantic evidence accumulation
  object_memory_store.py           # reusable object memory and confidence update
  landmark_retrieval.py            # landmark retrieval / ranking
  semantic_grounding_adapter.py    # grounding adapter contracts

scripts/
  phase23_habitat_control_server.py          # live Habitat control UI
  m25_habitat_rgbd_export.py                 # zero-map RGB-D/pose capture
  m25_groundingdino_export.py                # open-vocabulary grounding + 3D projection
  phase5a_active_table_search_capture.py     # close multi-view tabletop search
  phase5a_zero_map_cup_search_report.py      # final GIF/video/metrics report
  phase5a_sim_language_demo.py               # natural-language -> API planner -> Habitat demo
  phase5a_make_grounding_depth_storyboard.py # GIF storyboard generation
  phase5a_api_semantic_planner_eval.py       # API planner evaluation
  phase5a_navmesh_validate.py                # Habitat navmesh validation

docs/
  demo_drink_water_reusable_semantic_memory.md
  phase_docs/phase5_navigation_policy.md
  phase_docs/phase_log_audit_20260704.md

outputs/
  phase5a_sim_demo/                 # curated demo reports/GIFs/videos/traces
  phase5a_api_semantic_planner/      # curated qwen3-max planner reports
  phase5a_navmesh_validation/        # curated navmesh validation reports
  m35_semantic_representation_alignment/
```

Large model weights, third-party downloads, WSL images, and full raw experiment outputs are intentionally ignored by Git.

## API Planner

The selected Phase5A API planner is:

```text
provider: DashScope OpenAI-compatible
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
model: qwen3-max
key env: DASHSCOPE_API_KEY or OPENAI_API_KEY
```

API keys are not stored in this repository. Before running API demos, inject a key in the runtime environment, for example:

```bash
export DASHSCOPE_API_KEY=...
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENAI_MODEL=qwen3-max
```

## Reproduce The Demo

On a development machine with `habitat-sim` and GroundingDINO dependencies
installed, the zero-map case is a four-stage pipeline:

```bash
python scripts/m25_habitat_rgbd_export.py \
  --scene /path/to/17DRP5sb8fy.glb \
  --scene-dataset-config /path/to/mp3d.scene_dataset_config.json \
  --trajectory-mode coverage-loop \
  --lightweight-capture \
  --pitch-scan-every 4 \
  --max-frames 360 \
  --out-dir outputs/zero_map_run/capture

python scripts/m25_groundingdino_export.py \
  --frames-metadata outputs/zero_map_run/capture/frames_metadata.json \
  --labels cup,bottle,table,counter,sink \
  --max-frames 10000 \
  --out-dir outputs/zero_map_run/exploration_grounding
```

Use `phase5a_active_table_search_capture.py` to revisit candidate support
surfaces, run GroundingDINO on those additional views, then build the final
artifact:

```bash
python scripts/phase5a_zero_map_cup_search_report.py \
  --frames-metadata outputs/zero_map_run/active_search/combined_frames_metadata.json \
  --detections-json outputs/zero_map_run/exploration_grounding/detections.json \
  --detections-json outputs/zero_map_run/active_grounding/detections.json \
  --grounding-dir outputs/zero_map_run/exploration_grounding \
  --grounding-dir outputs/zero_map_run/active_grounding \
  --out-dir outputs/zero_map_run/final_report
```

## Key Reports

- `outputs/phase5a_api_semantic_planner/api_model_benchmark_20260703/api_model_benchmark.html`
- `outputs/phase5a_api_semantic_planner/qwen3_max_20case_noleak_20260703/qwen3_max_20case_noleak_report.html`
- `outputs/phase5a_navmesh_validation/qwen3_max_find5_20260704/navmesh_validation_report.html`
- `outputs/m35_semantic_representation_alignment/paper_semantic_bev_index_20260703.html`

## Status

Current status: demo-ready research prototype.

Validated:

- qwen3-max semantic waypoint selection;
- Habitat navmesh reachability for selected waypoint targets;
- first-person execution trace with stop-and-look behavior;
- empty-map RGB-D exploration with active tabletop scans;
- GroundingDINO-to-depth-to-world projection and multi-view cup memory;
- dynamic BEV + grounding + depth + confidence storyboard;
- explicit backlog for localization, interest exploration, Lingbo model trials, and API planner demo refinement.
