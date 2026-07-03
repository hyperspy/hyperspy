<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# misc

## Purpose
Internal utility modules not intended for direct public use. Houses helpers for array manipulation, axis tools, slicing, dask, and other cross-cutting concerns. Underscored files (`_*.py`) are private.

## Key Files

| File | Description |
|------|-------------|
| `array_tools.py` | Array reshape/broadcast helpers |
| `_array_tools.py` | Private array utilities |
| `axis_tools.py` | Axis value conversion and indexing helpers |
| `slicing.py` | Slicing machinery for `.isig` / `.inav` |
| `dask_utils.py` | Dask array utilities |
| `_dask_utils.py` | Private dask helpers |
| `math_tools.py` | Mathematical utilities (FFT helpers, etc.) |
| `hist_tools.py` | Histogram computation |
| `machine_learning.py` | ML result containers |
| `model_tools.py` | Model inspection utilities |
| `signal_tools.py` | Internal signal manipulation helpers |
| `_signal_tools.py` | Private signal utilities |
| `_markers.py` | Internal markers helpers |
| `_utils.py` | General internal utilities |
| `export_dictionary.py` | Dictionary serialisation for metadata |
| `ipython_tools.py` | IPython/Jupyter helpers |
| `lowess_smooth.py` | LOWESS implementation |
| `tv_denoise.py` | Total-variation denoising |
| `utils.py` | Catch-all internal utils |
| `test_utils.py` | Test helper utilities |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `eels/` | EELS cross-section and background utilities |
| `dask_widgets/` | Jupyter dask progress widgets |

## For AI Agents

### Working In This Directory

- Do not add public API here — use `hyperspy/utils/` for user-facing functions.
- Underscored files are private; non-underscored files may be imported by other internal modules.
- Avoid circular imports: `misc/` should not import from `hyperspy/signal.py`.

## Dependencies

### External
- `numpy`, `dask`, `scipy`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
