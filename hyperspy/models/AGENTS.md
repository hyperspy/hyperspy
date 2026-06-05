<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# models

## Purpose
Signal-type-specific `Model` subclasses. `Model1D` (for `Signal1D`) and `Model2D` (for `Signal2D`) extend `BaseModel` with dimension-appropriate fitting logic. Created via `signal.create_model()`.

## Key Files

| File | Description |
|------|-------------|
| `model1d.py` | `Model1D` — fitting engine for 1D signal axis |
| `model2d.py` | `Model2D` — fitting engine for 2D signal axis |

## For AI Agents

### Model Workflow

```python
m = signal.create_model()               # returns Model1D or Model2D

# Add components (see hyperspy/_components/)
m.append(hs.model.components1D.Gaussian(name="Peak"))
m.append(hs.model.components1D.PowerLaw(name="Background"))

# Initialize parameters
m.components.Peak.estimate_parameters(signal, x1=80, x2=120)

# Single position fit
m.fit(bounded=True)

# Fit all navigation positions
m.multifit(bounded=True, show_progressbar=True)

# Evaluate
sim = m.as_signal()
residual = signal - sim
```

### Parameter Map Convention

Parameter maps are always in **NumPy array order** — not HyperSpy display order:

```python
# Navigation shape (80, 50) in HyperSpy display → (50, 80) in NumPy array order
centre_map = m.components.Peak.centre.map  # shape (50, 80) = (ny, nx)
centre_map[j, i]  # [y-index, x-index] — NumPy convention
```

### Navigation Dimension Guard

```python
if signal.axes_manager.navigation_dimension == 0:
    m.components.Peak.centre.value = 100.0     # scalar
else:
    m.components.Peak.centre.map["values"] = np.full(
        signal.axes_manager.navigation_shape[::-1], 100.0  # NumPy order
    )
    m.components.Peak.centre.map["is_set"] = True
```

### Testing Requirements

```bash
pytest hyperspy/tests/model/
```

## Dependencies

### Internal
- `hyperspy/model.py` — `BaseModel`
- `hyperspy/_components/` — all component classes
- `hyperspy/signal.py` — signal creates the model

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
