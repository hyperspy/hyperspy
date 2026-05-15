<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# _components

## Purpose
Concrete implementations of 1D and 2D model components. Each file defines one `Component` subclass. Components are publicly re-exported via `hyperspy/components1d.py` and `hyperspy/components2d.py`.

## Key Files

| File | Description |
|------|-------------|
| `gaussian.py` | Gaussian peak (A, centre, sigma) |
| `gaussian2d.py` | 2D Gaussian for image fitting |
| `gaussianhf.py` | Gaussian in height-FWHM parameterisation |
| `lorentzian.py` | Lorentzian peak |
| `voigt.py` | Voigt profile (convolution of Gaussian + Lorentzian) |
| `split_voigt.py` | Asymmetric Voigt with independent left/right widths |
| `skew_normal.py` | Skew-normal distribution peak |
| `power_law.py` | Power-law background (EELS standard) |
| `polynomial.py` | Polynomial component (configurable order) |
| `expression.py` | Generic `Expression` component — any SymPy-parseable formula |
| `offset.py` | Constant offset component |
| `exponential.py` | Exponential decay/growth |
| `logistic.py` | Logistic (sigmoid) function |
| `arctan.py` | Arctangent step function |
| `error_function.py` | Error function (erf) step |
| `heaviside.py` | Heaviside step function |
| `doniach.py` | Doniach-Šunjić lineshape for XPS core-level peaks |
| `bleasdale.py` | Bleasdale-Nelder function |
| `rc.py` | RC circuit response |
| `scalable_fixed_pattern.py` | Fixed spectral pattern with a scalar multiplier |

## For AI Agents

### Adding a New Component

1. Subclass `hyperspy.component.Component`.
2. Define `__init__` with all parameters as `Parameter` instances.
3. Implement `function(self, x)` (1D) or `function(self, x, y)` (2D) returning a NumPy array.
4. Optionally implement `grad_*` methods for analytical Jacobians (speeds up fitting).
5. Optionally implement `estimate_parameters(signal, x1, x2)` for auto-initialisation.
6. Add a public re-export in `hyperspy/components1d.py` or `components2d.py`.
7. Add tests in `hyperspy/tests/component/`.

### Component Pattern

```python
from hyperspy.component import Component, Parameter

class MyComponent(Component):
    def __init__(self, A=1.0, centre=0.0, sigma=1.0):
        super().__init__(["A", "centre", "sigma"])
        self.A.value = A
        self.centre.value = centre
        self.sigma.value = sigma

    def function(self, x):
        return self.A.value * np.exp(-0.5 * ((x - self.centre.value) / self.sigma.value) ** 2)
```

### Prefer Expression for Custom Math

```python
# Instead of a new file, use Expression for one-off custom shapes:
c = hs.model.components1D.Expression(
    "A * exp(-abs(x - centre) / decay)",
    name="Exponential_Peak",
    A=1000, centre=100, decay=5
)
```

### Testing Requirements

```bash
pytest hyperspy/tests/component/
```

Each component should have a test verifying `function()` output and, if present, `estimate_parameters()`.

## Dependencies

### Internal
- `hyperspy/component.py` — base `Component` and `Parameter` classes

### External
- `numpy`, `scipy.special` (for erf, voigt profiles)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
