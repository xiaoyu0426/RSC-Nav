# RSC-Nav

RSC-Nav studies long-term semantic-spatial memory augmentation for embodied navigation.

Current repository stage:

- Phase 0: protocol and data contracts.
- Phase 1: minimal memory core smoke test.

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
