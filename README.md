# RSC-Nav

RSC-Nav studies long-term semantic-spatial memory augmentation for embodied navigation.

Current repository stage:

- Phase 0: protocol and data contracts.
- Phase 1: minimal memory core smoke test.
- Phase 2: synthetic BEV memory projection smoke test.

## Phase 0+1 Smoke Test

Run:

```powershell
python scripts\phase01_smoke_test.py
```

Outputs:

- `outputs/phase01/phase01_smoke_log.json`
- `outputs/phase01/phase01_smoke_visualization.png`

The smoke test demonstrates:

```text
write -> retrieve -> perturb -> stale retrieval -> reconfigured retrieval
```

It is synthetic by design and does not require Habitat. Habitat integration starts after the memory core and schema are stable.

## Phase 2 BEV Smoke Test

Run:

```powershell
python scripts\phase02_bev_smoke_test.py
```

Outputs:

- `outputs/phase02/phase02_log.json`
- `outputs/phase02/bev_occupancy.png`
- `outputs/phase02/bev_semantic.png`
- `outputs/phase02/bev_memory_overlay.png`
- `outputs/phase02/bev_update_sequence.png`

This smoke test demonstrates:

```text
synthetic observation + pose
-> BEV projection
-> occupancy / explored map
-> semantic evidence map
-> long-term memory retrieval
```

It is still synthetic. The purpose is to stabilize the BEV memory interface before replacing synthetic observations with Habitat RGB-D, pose, and semantic detector outputs.
