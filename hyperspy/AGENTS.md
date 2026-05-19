<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# hyperspy (main package)

## Purpose
The core Python package. Contains the `BaseSignal` class hierarchy, `AxesManager`, model framework, component library, I/O layer, machine learning tools, drawing/plotting engine, and all utilities. Public API is re-exported through `hyperspy/api.py`.

## Key Files

| File | Description |
|------|-------------|
| `api.py` | Public API surface — what `import hyperspy.api as hs` exposes |
| `signal.py` | `BaseSignal` — base class for all signal types |
| `axes.py` | `AxesManager`, `DataAxis`, `UniformDataAxis`, `FunctionalDataAxis` |
| `component.py` | `Component` base class for model components |
| `model.py` | `BaseModel` — shared model fitting logic |
| `components1d.py` | Public re-export of 1D components from `_components/` |
| `components2d.py` | Public re-export of 2D components from `_components/` |
| `signals.py` | Public re-export of signal types from `_signals/` |
| `io.py` | File load/save dispatcher |
| `roi.py` | Region-of-interest classes |
| `samfire.py` | SAMFire (Smart Adaptive Multi-dimensional Fitting) entry point |
| `events.py` | Event system used for reactive UI updates |
| `exceptions.py` | Custom exception classes |
| `extensions.py` | Extension registry for third-party hyperspy plugins |
| `decorators.py` | Internal decorators (`@lazify`, `@deprecated`, etc.) |
| `defaults_parser.py` | Preferences system backed by `hyperspy_extension.yaml` |
| `interactive.py` | `interactive()` helper for live-linked signal operations |
| `logger.py` | Package-level logger setup |
| `ui_registry.py` | Registry for GUI widgets |
| `_lazy_signals.py` | Lazy signal type re-exports |
| `conftest.py` | Pytest fixtures shared across test suite |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `_components/` | 1D and 2D model component implementations (see `_components/AGENTS.md`) |
| `_signals/` | Concrete signal type implementations (see `_signals/AGENTS.md`) |
| `data/` | Built-in sample data files bundled with the package |
| `datasets/` | Loaders for built-in and downloadable example datasets |
| `docstrings/` | Shared docstring templates injected via `_docstring.py` pattern |
| `drawing/` | Matplotlib-based plotting engine (see `drawing/AGENTS.md`) |
| `external/` | Vendored third-party code (mpfit, astropy units, etc.) |
| `io_plugins/` | File format I/O plugins (see `io_plugins/AGENTS.md`) |
| `learn/` | Matrix decomposition and BSS algorithms (see `learn/AGENTS.md`) |
| `misc/` | Internal utility modules (see `misc/AGENTS.md`) |
| `models/` | Model subclasses for Signal1D and Signal2D (see `models/AGENTS.md`) |
| `samfire_utils/` | SAMFire strategy, segmenter, and worker code (see `samfire_utils/AGENTS.md`) |
| `signal_tools/` | Interactive signal editing dialogs (see `signal_tools/AGENTS.md`) |
| `tests/` | Full test suite (see `tests/AGENTS.md`) |
| `utils/` | Public utility functions (see `utils/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- **Never modify `api.py` casually** — it defines the public contract; changes may break downstream code.
- When adding a new signal type, add it to both `_signals/` and re-export it in `signals.py`.
- When adding a new component, add it to `_components/` and re-export in `components1d.py` or `components2d.py`.
- `AxesManager` (`axes.py`) stores axes in **NumPy array order** internally; the `_axes` list in natural/display order is a reversed view. Do not conflate these.
- `BaseSignal.data` is always a NumPy (or dask) array in **NumPy order** — never display order.

### Navigation Dimension Guard Pattern

```python
if signal.axes_manager.navigation_dimension == 0:
    component.param.value = scalar_value          # single position
else:
    component.param.map["values"] = array         # shape (ny, nx, ...)
    component.param.map["is_set"] = True
```

### Testing Requirements

```bash
pytest hyperspy/tests/
ruff check hyperspy/
```

New code must have corresponding tests. Mirror the test structure to the source structure (e.g., `hyperspy/tests/signals/` for code in `hyperspy/_signals/`).

### Common Patterns

- Lazy evaluation: wrap computations with dask; use `signal._lazy` flag to branch.
- Events: use `signal.events.data_changed.trigger()` after mutating `.data` in-place.
- Deprecation: use the `@deprecated` decorator from `hyperspy/decorators.py`, not bare warnings.

## Dependencies

### Internal
- All subpackages depend on `signal.py` and `axes.py` as the foundation.

### External
- `numpy`, `dask`, `scipy`, `matplotlib`, `pint`, `rsciio`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
