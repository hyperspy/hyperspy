<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# tests

## Purpose
Full pytest test suite for HyperSpy. Mirrors the source tree structure under `hyperspy/`.

## Key Files

| File | Description |
|------|-------------|
| `test_data.py` | Tests for built-in data and dataset loaders |
| `test_decorators.py` | Tests for `@deprecated` and other decorators |
| `test_events.py` | Tests for the event system |
| `test_io.py` | Tests for file I/O (load/save roundtrip) |
| `test_import.py` | Smoke tests for the public API import |
| `test_interactive.py` | Tests for `interactive()` helper |
| `test_extension_registry.py` | Tests for the extension plugin system |
| `test_progressbar.py` | Tests for progress bar utilities |
| `test_non-uniform_not-implemented.py` | Tests ensuring non-uniform axis limitations are enforced |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `axes/` | Tests for `AxesManager` and `DataAxis` |
| `component/` | Tests for individual model components |
| `doc_docstr_examples/` | Doctests extracted from docstrings |
| `drawing/` | Tests for the plotting engine |
| `external/` | Tests for vendored external code |
| `learn/` | Tests for decomposition algorithms |
| `misc/` | Tests for internal utility modules |
| `model/` | Tests for model fitting |
| `samfire/` | Tests for SAMFire |
| `signals/` | Tests for signal types |
| `utils/` | Tests for public utility functions |

## For AI Agents

### Running Tests

```bash
pytest hyperspy/tests/                  # full suite
pytest hyperspy/tests/signals/          # signal tests only
pytest hyperspy/tests/ -x               # stop on first failure
pytest hyperspy/tests/ -k "test_name"   # filter by name
```

### Writing Tests

- **Use ALL-DIFFERENT array dimensions** so axis reversals are detectable:
  ```python
  data = np.random.random((12, 25, 48))   # NOT (64, 64, 128)
  s = hs.signals.Signal1D(data)           # clearly shows (25, 12 | 48)
  ```
- Mirror source structure: tests for `hyperspy/_signals/signal1d.py` go in `hyperspy/tests/signals/test_signal1d.py`.
- Use the `conftest.py` fixtures for common signal creation patterns.
- Mark slow tests with `@pytest.mark.slow`; they are skipped in fast CI runs.
- Use `pytest.approx()` for floating-point comparisons.

### Fixtures

Common fixtures are defined in `hyperspy/conftest.py`:
- `signal1d` / `signal2d` — pre-built test signals
- `mpl_cleanup` — ensures matplotlib figures are closed after each test

## Dependencies

### External
- `pytest`, `pytest-cov`, `pytest-mpl` (for image comparison tests)
- `matplotlib` with Agg backend for GUI tests

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
