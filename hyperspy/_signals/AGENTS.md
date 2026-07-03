<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# _signals

## Purpose
Concrete signal type implementations. Each file adds domain-specific methods on top of `BaseSignal`. Public API is re-exported via `hyperspy/signals.py`.

## Key Files

| File | Description |
|------|-------------|
| `signal1d.py` | `Signal1D` — 1D signal axis; core EELS/EDX/spectrum-image class |
| `signal2d.py` | `Signal2D` — 2D signal axis; images, diffraction patterns, 4D-STEM |
| `common_signal1d.py` | Mixin with methods shared by Signal1D and its lazy variant |
| `common_signal2d.py` | Mixin with methods shared by Signal2D and its lazy variant |
| `lazy_signal1d.py` | `LazySignal1D` — dask-backed Signal1D |
| `lazy_signal2d.py` | `LazySignal2D` — dask-backed Signal2D |
| `lazy.py` | `LazySignal` base mixin for all lazy signals |
| `complex_signal.py` | `ComplexSignal` — complex-valued data |
| `complex_signal1d.py` | `ComplexSignal1D` |
| `complex_signal2d.py` | `ComplexSignal2D` |
| `lazy_complex_signal.py` | `LazyComplexSignal` |
| `lazy_complex_signal1d.py` | `LazyComplexSignal1D` |
| `lazy_complex_signal2d.py` | `LazyComplexSignal2D` |
| `_signal1d_tool.py` | Internal helpers for Signal1D operations |

## For AI Agents

### Signal Class Hierarchy

```
BaseSignal (signal.py)
├── Signal1D          ← most EELS/EDX/spectral work
│   └── LazySignal1D
├── Signal2D          ← images, diffraction
│   └── LazySignal2D
└── ComplexSignal
    ├── ComplexSignal1D / ComplexSignal2D
    └── Lazy variants
```

### Axis Convention (critical)

The last N axes of the NumPy array are the signal axes. HyperSpy reverses both navigation and signal dimensions for display:

```python
# Signal1D: NumPy (ny, nx, E) → display (nx, ny | E)
# Signal2D: NumPy (ny, nx, qy, qx) → display (nx, ny | qx, qy)
```

### Adding Methods to Signal Types

- Methods that apply to both Signal1D and LazySignal1D belong in `common_signal1d.py`.
- Lazy-specific overrides go in `lazy_signal1d.py`.
- Always accept `axis` as a named parameter and resolve it via `self.axes_manager[axis]`.
- Return a new signal (don't mutate in-place) unless the method is explicitly `_inplace`.

### Testing Requirements

```bash
pytest hyperspy/tests/signals/
```

Mirror test file names to source files (e.g. `test_signal1d.py` for `signal1d.py`).

## Dependencies

### Internal
- `hyperspy/signal.py` — `BaseSignal` base class
- `hyperspy/axes.py` — `AxesManager`
- `hyperspy/learn/mva.py` — decomposition methods mixed into Signal

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
