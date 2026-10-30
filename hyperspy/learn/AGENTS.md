<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# learn

## Purpose
Matrix decomposition, blind source separation (BSS), and machine learning algorithms. The public entry point is `mva.py` which is mixed into `BaseSignal` via the `MVA` mixin.

## Key Files

| File | Description |
|------|-------------|
| `mva.py` | `MVA` mixin — `decomposition()`, `blind_source_separation()`, `get_decomposition_model()` |
| `_svd_pca.py` | SVD/PCA implementation (community standard for HyperSpy) |
| `_mlpca.py` | Maximum-likelihood PCA (Poisson noise model) |
| `_rpca.py` | Robust PCA (L+S decomposition) |
| `_ornmf.py` | Online Robust NMF |
| `_orthomax.py` | Orthomax / varimax rotation for BSS |
| `_mva.py` | Internal MVA helpers |
| `_whitening.py` | Data whitening preprocessing |

## For AI Agents

### Preferred Decomposition Pattern

```python
# ✅ Use SVD — HyperSpy community standard
signal.decomposition(algorithm='SVD')

# ✅ Let HyperSpy estimate the number of significant components
n = signal.learning_results.number_significant_components
signal.plot_explained_variance_ratio(vline=True)  # verify with scree plot

# ✅ Reconstruct and evaluate residuals
for n in [6, 8, 10]:
    rec = signal.get_decomposition_model(components=n).as_signal()
    residual = signal - rec
    print(f"n={n}: residual std = {np.std(residual.data):.4f}")
```

### BSS Pattern

```python
signal.decomposition(algorithm='SVD')
signal.blind_source_separation(number_of_components=n)
# Results: signal.learning_results.bss_components / bss_scores
```

### Data Preprocessing

```python
# Guard against negative values before NMF / Poisson-based methods
if signal.data.min() < 0:
    signal.data = signal.data - signal.data.min() + 1e-6
```

### Testing Requirements

```bash
pytest hyperspy/tests/learn/
```

### Common Patterns

- `learning_results` is a `LearningResults` dataclass on the signal; do not access private attributes directly.
- Lazy signals use incremental / online algorithms internally; the public API is identical.
- `sklearn_pca` algorithm requires `scikit-learn`; guard with `try/except ImportError`.

## Dependencies

### Internal
- Mixed into `BaseSignal` through `hyperspy/signal.py`

### External
- `numpy`, `scipy` — core linear algebra
- `scikit-learn` — optional, for `sklearn_pca` and related algorithms
- `dask` — lazy decomposition paths

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
