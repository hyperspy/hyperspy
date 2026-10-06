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

## Plot Testing with Image Comparison (pytest-mpl)

All `@pytest.mark.mpl_image_compare` tests live in `hyperspy/tests/drawing/`.
Without the `--mpl` flag they execute but **skip image comparison** — a green
run without `--mpl` does not validate rendering.

### Pinned matplotlib/freetype (REQUIRED)

Image comparison only passes against the versions pinned in
`conda_environment_dev.yml` (currently `matplotlib-base=3.9.2`,
`freetype=2.12`). Small rendering differences between matplotlib/freetype
versions fail the strict `tolerance=2.0`. Check with:

```bash
python -c "import matplotlib; from matplotlib import ft2font; \
print(matplotlib.__version__, ft2font.__freetype_version__)"
```

### Setup and run (dedicated env recommended)

```bash
conda env create -f conda_environment_dev.yml -n hyperspy-mpl
conda run -n hyperspy-mpl pip install -e ".[tests]"
MPLBACKEND=agg conda run -n hyperspy-mpl python -m pytest \
    hyperspy/tests/drawing/ --mpl
```

Notes:
- `pytest-xdist` must stay `<3.5` (xdist ≥3.5 has a regression that breaks
  reference-image handling in some plot tests; the pin is in the yml).
- The yml does not pin a Python version; the solver picks a compatible one
  (3.13 at time of writing).

### Regenerating baseline images

Do this only in the pinned env above — baselines generated with any other
matplotlib/freetype fail comparison for everyone else.

```bash
MPLBACKEND=agg conda run -n hyperspy-mpl python -m pytest \
    hyperspy/tests/drawing/test_plot_signal1d.py::TestPlotSpectra \
    --mpl --mpl-generate-path=hyperspy/tests/drawing/plot_signal1d -n 0
```

Rules:
- Pass **both** `--mpl` and `--mpl-generate-path`. With only
  `--mpl-generate-path`, the `setup_teardown` fixture in
  `test_plot_signal1d.py` skips its copy-back step and leaves stray
  `-True.png`/`-None.png` per-parameter files next to the canonical baselines.
- Regenerate fig-parametrized `TestPlotSpectra` tests as the **whole class in
  one invocation**. The fixture teardown copies each generated per-parameter
  image back onto the single style-only canonical file (e.g.
  `test_plot_spectra_cascade.png`) and iterates all styles; running a subset
  errors in teardown on the missing per-parameter files. Expect benign
  teardown errors mid-run for the same reason — the last test's teardown does
  the final copy-back and cleanup.
- Use `-n 0` (sequential): teardown does file copies/removals that should not
  race between xdist workers.
- Non-parametrized tests (e.g. `test_plot_spectra_ax`) can be regenerated
  individually by node ID.
- After regenerating, rerun with `--mpl` (whole `hyperspy/tests/drawing/`) and
  confirm zero failures before committing.

### Baseline provenance

All `plot_signal1d` baselines were regenerated with the pinned stack
(mpl 3.9.2 / freetype 2.12.1) on 2026-10-06; the drawing suite passes with
`--mpl` (657 passed, 0 failed). Earlier baselines had mixed
matplotlib/freetype provenance — passed only within tolerance.
