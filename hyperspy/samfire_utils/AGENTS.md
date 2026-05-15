<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# samfire_utils

## Purpose
SAMFire (Smart Adaptive Multi-dimensional Fitting) infrastructure. SAMFire fits multidimensional datasets by using already-fitted neighbouring pixels to seed parameter guesses, dramatically improving convergence and speed.

## Key Files

| File | Description |
|------|-------------|
| `strategy.py` | Base fitting strategy |
| `local_strategies.py` | Local (neighbour-based) strategies |
| `global_strategies.py` | Global (statistics-based) strategies |
| `samfire_kernel.py` | Core SAMFire fitting loop |
| `samfire_pool.py` | Worker pool management |
| `samfire_worker.py` | Individual worker process |
| `fit_tests.py` | Convergence acceptance tests |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `segmenters/` | Spatial segmentation algorithms for strategy guidance |
| `goodness_of_fit_tests/` | Statistical tests for fit acceptance |
| `weights/` | Weighting schemes for strategy selection |

## For AI Agents

### Working In This Directory

- SAMFire is accessed via `signal.create_samfire(workers=N)` — do not instantiate SAMFire classes directly.
- Strategy objects control the order in which pixels are fitted; local strategies use fitted neighbours, global strategies use parameter distributions.
- Worker processes use pickle — ensure all objects passed to workers are picklable.

### Testing Requirements

```bash
pytest hyperspy/tests/samfire/
```

## Dependencies

### Internal
- `hyperspy/samfire.py` — public entry point
- `hyperspy/models/` — models being fitted

### External
- `multiprocessing` — parallel worker pool

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
