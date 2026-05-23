<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# HyperSpy

## Purpose
HyperSpy is a Python library for multi-dimensional data analysis, specialising in hyperspectral and hyperimage datasets (EELS, EDX, CL, 4D-STEM, etc.). It provides a unified signal framework with calibrated axes, a component-based model fitting system, machine learning decomposition tools, and an extensive I/O plugin ecosystem.

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Build system config, dependencies, optional extras |
| `setup.py` | Legacy setuptools entry point |
| `README.md` | Project overview and quick-start |
| `CHANGES.rst` | Changelog |
| `releasing_guide.md` | How to cut a release |
| `prepare_release.py` | Release preparation script |
| `azure-pipelines.yml` | CI configuration |
| `readthedocs.yaml` | Read the Docs build config |
| `conda_environment.yml` | Conda environment for users |
| `conda_environment_dev.yml` | Conda environment for developers |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `hyperspy/` | Main Python package (see `hyperspy/AGENTS.md`) |
| `doc/` | Sphinx documentation source (see `doc/AGENTS.md`) |
| `examples/` | Gallery example scripts (see `examples/AGENTS.md`) |
| `upcoming_changes/` | Towncrier news fragments for the next release (see `upcoming_changes/AGENTS.md`) |

## For AI Agents

### Critical Concept: HyperSpy Axis Convention

**This is the most important concept when working with HyperSpy code.**

HyperSpy reverses ALL dimensions relative to NumPy order — both navigation AND signal axes:

```python
# NumPy array shape: (A, B, C, D)
# Signal1D → HyperSpy display: (C, B, A | D)   — last 1 axis = signal
# Signal2D → HyperSpy display: (B, A | D, C)   — last 2 axes = signal
# BaseSignal → HyperSpy display: (D, C, B, A |) — all axes = navigation
```

Use ALL-DIFFERENT array dimensions when writing tests or examples so the reversal is immediately visible:
```python
data = np.random.random((12, 25, 48))   # NOT (64, 64, 128)
s = hs.signals.Signal1D(data)           # displays as (25, 12 | 48)
```

### NumPy Order for Arrays and Parameter Maps

All underlying arrays (`.data`, parameter `.map`) use NumPy order, never display order:

```python
# For a 2D spatial scan: HyperSpy shows (nx, ny | ...) but arrays are (ny, nx, ...)
peak_centre_map = model.components.Peak.centre.map  # shape (ny, nx) — NumPy order
mask[y_slice, x_slice] = True                        # always [y, x] indexing
```

### Preferred HyperSpy Patterns

```python
# ✅ Use HyperSpy-native methods, not raw NumPy on .data
signal.max(axis='energy')               # keeps metadata
signal.isig[100.:300.]                  # calibrated slicing
signal / signal.max()                   # preserves structure

# ✅ Name axes immediately after creation
signal.axes_manager[0].name = 'x'

# ✅ Batch-set axis properties
signal.axes_manager.navigation_axes.set(units='nm', scale=0.1, name=('x', 'y'))

# ✅ Access model components by name
model.components.Peak.centre.value = 100
model.set_parameters_value('centre', 100, only_current=False)

# ✅ Use SVD for decomposition (community standard)
signal.decomposition(algorithm='SVD')
n = signal.learning_results.number_significant_components
```

### Working in This Repository

- Run tests: `pytest hyperspy/tests/`
- Lint: `ruff check`
- Add changelog entries for every user-facing change in `upcoming_changes/` (see format in `upcoming_changes/README.rst`)
- Never edit `.rst` files in `doc/_build/` — those are generated
- When adding a new public API, update the relevant docstring template in `hyperspy/docstrings/`

### Testing Requirements

```bash
pytest hyperspy/tests/                  # full suite
pytest hyperspy/tests/signals/          # signal-specific tests
pytest -x hyperspy/tests/              # stop on first failure
```

Use ALL-DIFFERENT array dimensions in new tests so axis transformations are clearly verifiable.

### Common Patterns

- Signal classes live in `hyperspy/_signals/`; public re-exports in `hyperspy/signals.py`
- Model components live in `hyperspy/_components/`; public re-exports in `hyperspy/components1d.py` / `components2d.py`
- Navigation dimension 0 = single position: use `.value`. Navigation dimension > 0: use `.map["values"]` + `.map["is_set"] = True`
- Prefer `Expression` component for custom mathematical functions rather than raw NumPy simulations

## Dependencies

### Internal
- All subpackages interact through `hyperspy/signal.py` (base `BaseSignal`) and `hyperspy/axes.py` (`AxesManager`)

### External
- `numpy`, `scipy` — numerical core
- `matplotlib` — plotting
- `dask` — lazy/out-of-core computation
- `pint` — unit-aware axis calibration
- `rsciio` — file I/O (separate package, accessed via `hyperspy/io_plugins/`)
- `scikit-learn` — optional, for additional decomposition algorithms

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->

## AI Agent Setup

Before implementing any changes, verify that the development environment
is correctly configured.  Do **not** start editing files until these pass:

- ``pre-commit install`` — activates lint, format, and AI co-author checks
  on every commit.
- ``pre-commit run --all-files`` — should pass cleanly.  Fix any reported
  issues before proceeding.

If ``pre-commit`` is not installed: ``pip install pre-commit``.

Add setup steps to the first todo item of every session — do not skip it.

## Agent Completion Checklist

Before claiming any task is complete, verify ALL of the following:

### Code Quality
- [ ] `ruff check` passes on all changed files — zero new errors
- [ ] `ruff format` applied — no formatting inconsistencies

### Testing
- [ ] Affected tests pass: `pytest hyperspy/tests/<affected>/`
- [ ] New code has corresponding tests that mirror the source structure
- [ ] Tests use ALL-DIFFERENT array dimensions (e.g., `(12, 25, 48)`) so axis reversals are verifiable
- [ ] Floating-point comparisons use `pytest.approx()`, not raw `==`

### Changelog
- [ ] Every user-facing change has an `upcoming_changes/<issue>.<type>.rst` entry
- [ ] The `<type>` matches one of: `new`, `bugfix`, `doc`, `deprecation`, `enhancements`, `api`, `maintenance`

### Documentation
- [ ] New public API has updated docstring templates in `hyperspy/docstrings/`
- [ ] Never edit `.rst` files in `doc/_build/` — those are generated
- [ ] Non-obvious design choices are annotated with inline comments explaining intent
- [ ] For structural changes (file moves, renames, splits, new modules), provide a change map for the PR description: what changed, what moved where, and why

### Commits
- [ ] Commit following best practices (atomic units, repo-consistent messages, no secrets)
- [ ] MUST NOT use ``Co-authored-by:`` trailer for AI tools — use ``Assisted-by: <tool>:<model>`` instead
- [ ] Never push unless explicitly asked

### Repository Hygiene
- [ ] Never modify AGENTS.md generated sections — only add notes below `<!-- MANUAL -->` lines
- [ ] Never suppress type/lint errors with blanket ignores (`# type: ignore`, `# noqa` without justification)

### HyperSpy-Specific
- [ ] All axis operations respect the NumPy-vs-display order convention (see "Critical Concept" above)
- [ ] Prefer HyperSpy-native methods over raw NumPy on `.data`
