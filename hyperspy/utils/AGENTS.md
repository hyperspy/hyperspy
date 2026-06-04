<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# utils

## Purpose
Public utility functions exposed in `hyperspy.utils.*`. Organised by domain. These are part of the public API.

## Key Files

| File | Description |
|------|-------------|
| `plot.py` | `hs.plot.*` — `plot_spectra`, `plot_images`, `plot_signals` |
| `roi.py` | `hs.roi.*` — ROI classes (`RectangularROI`, `CircleROI`, `Line2DROI`, etc.) |
| `model.py` | `hs.model.*` — `components1D`, `components2D` namespace |
| `model_tools.py` | Model utility helpers (goodness of fit, parameter statistics) |
| `signal_tools.py` | Signal utility helpers |
| `axis_tools.py` | Axis manipulation utilities |
| `array_tools.py` | Array manipulation (broadcast, reshape helpers) |
| `dask_utils.py` | Dask computation helpers |
| `math_tools.py` | Mathematical utilities |
| `hist_tools.py` | Histogram helpers |
| `machine_learning.py` | `hs.utils.machine_learning.*` — decomposition result access |
| `export_dictionary.py` | Signal metadata dict export |
| `slicing.py` | Slicing utilities shared by `.isig` / `.inav` |
| `lowess_smooth.py` | LOWESS smoothing helper |
| `tv_denoise.py` | Total-variation denoising |
| `ipython_tools.py` | Jupyter/IPython display helpers |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `eels/` | EELS-specific utilities (Hartree-Slater cross-sections, background, etc.) |
| `dask_widgets/` | Dask progress widgets for Jupyter |

## For AI Agents

### Working In This Directory

- These are **public API** — changes must be backward-compatible or use the `@deprecated` decorator.
- New utilities should be added here (not in `misc/`) if they are intended for end-users.
- Keep functions stateless and pure where possible.

### Testing Requirements

```bash
pytest hyperspy/tests/utils/
```

## Dependencies

### Internal
- Used throughout the package; avoids importing from `hyperspy/signal.py` to prevent circular imports.

### External
- `numpy`, `scipy`, `matplotlib`, `dask`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
