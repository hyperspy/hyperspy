<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# examples

## Purpose
sphinx-gallery example scripts that are automatically executed at doc-build time and rendered into the online gallery. Each subdirectory maps to a gallery section.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `create_signal/` | Signal construction from arrays and tabular data |
| `data_visualization/` | Plotting, image stacks, multi-signal displays |
| `extensions/` | Using HyperSpy extensions and third-party plugins |
| `io/` | Loading and saving files |
| `Markers/` | All marker types: arrows, circles, lines, polygons, text, etc. |
| `model_fitting/` | Component fitting, residual plots, simple fits |
| `plotting/` | Advanced plot customisation |
| `processing/` | Baseline removal, smoothing, peak finding |
| `region_of_interest/` | ROI creation and interactive use |
| `simple_simulations/` | Building simulated signals via the model framework |

## For AI Agents

### Working In This Directory

- Every `.py` file must start with a module-level docstring that becomes the gallery page title and description (sphinx-gallery convention).
- Scripts must be **fully runnable** without user interaction — avoid `plt.show()` without a non-interactive backend guard.
- Use ALL-DIFFERENT array dimensions in examples so axis reversals are visible: e.g. `np.random.random((12, 25, 48))` not `(64, 64, 128)`.
- Prefer `hs.signals.Signal1D(data)` over lower-level constructors.
- Each subdirectory should have a `README.rst` (gallery section header).

### Testing Requirements

Examples are executed as part of the documentation build. Ensure they run cleanly:

```bash
cd doc && make html   # will execute all examples
```

### Common Patterns

```python
"""
Example Title
=============
Brief description for the gallery.
"""
import numpy as np
import hyperspy.api as hs

data = np.random.random((12, 25, 48))   # (ny, nx, energy) — all different!
s = hs.signals.Signal1D(data)           # displays as (25, 12 | 48)
s.axes_manager.navigation_axes.set(units='nm', scale=0.5, name=('x', 'y'))
s.axes_manager.signal_axes[0].name = 'energy'
```

## Dependencies

### Internal
- `hyperspy/` — all examples import `hyperspy.api as hs`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
