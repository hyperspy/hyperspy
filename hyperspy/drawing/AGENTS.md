<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-06-08 -->

# drawing

## Purpose
Backend-agnostic interactive plotting engine.  Handles the `signal.plot()` infrastructure, including multi-panel figures, navigator/signal separation, interactive widgets, and the markers system.  All rendering goes through a `PlottingBackend` protocol so that alternative backends (anyplotlib, future backends) can replace matplotlib without touching the core drawing logic.

## Key Files

| File | Description |
|------|-------------|
| `signal.py` | `SignalFigure` — top-level figure manager for any signal |
| `signal1d.py` | 1D signal panel rendering |
| `image.py` | 2D image panel rendering |
| `figure.py` | `BlittedFigure` — base figure with blit-based animation |
| `he.py` / `hse.py` / `hie.py` | `HyperExplorer` base + 1D / 2D abstract explorer classes |
| `markers.py` | Public markers entry point |
| `widget.py` | Navigator widget base |
| `widgets.py` | Concrete interactive widgets (crosshair, range, etc.) |
| `tiles.py` | Tiled signal display |
| `utils.py` | Plot utility helpers |

## Backend System

### Key Files

| File | Description |
|------|-------------|
| `backends/_protocol.py` | `PlottingBackend` Protocol + `BackendCapabilityError` |
| `backends/_registry.py` | Entry-point discovery via `importlib.metadata` |
| `backends/__init__.py` | `get_backend()` / `register_backend()` |
| `backends/mpl/` | Matplotlib backend (default) |
| `backends/anyplotlib/` | anyplotlib backend |
| `backends/_magic.py` | `%anyplotlib` IPython magic |

### How the active backend is selected

1. `hyperspy/drawing/__init__.py` loads `"matplotlib"` at import time.
2. `preferences.Plot.backend` is an open `Str` trait; changing it fires `_on_backend_pref_change` which calls `load_backend(name)`.
3. `load_backend` looks up the `"hyperspy.backends"` entry-point group; external packages register there.

### Adding a new backend (summary)

1. Create a package with `pyproject.toml`:
   ```toml
   [project.entry-points."hyperspy.backends"]
   mybackend = "hyperspy_mybackend.backend:MyBackend"
   ```
2. Implement every method in `backends/_protocol.py` (76 methods).  Unsupported features should raise `BackendCapabilityError`.
3. **Override `get_explorer`** — the Protocol default returns the abstract `HyperExplorer` base class and produces broken plots.  Subclass `HyperSignal1D_Explorer` + `HyperImage_Explorer` and implement their four abstract methods.  See `backends/anyplotlib/_explorers.py` for a full example.
4. Set `ax.hspy_fig = <Signal1DFigure>` on your axes objects (done by `signal1d.py` automatically) — your axes must accept arbitrary attribute assignment.
5. Verify: `isinstance(MyBackend(), PlottingBackend)` and run `pytest hyperspy/tests/drawing/`.

### Known remaining MPL coupling

The items below have not yet been routed through the backend.  A new backend whose axes are plain Python objects (accept monkey-patching) will not hit these, but they are tracked for cleanup:

- `_markers/_` — collections are matplotlib objects (`LineCollection`, `PolyCollection`, etc.); a non-MPL backend's `add_collection` should raise `BackendCapabilityError`.
- `_widgets/line2d.py`, `_widgets/circle.py` — `plt.Line2D` / `plt.Circle` widget construction is MPL-only; these widgets are not used by anyplotlib.
- `signal1d.py` — `ax.hspy_fig.ax_markers` accesses the parent figure's marker list (not a backend call).
- `utils.py` — `plt.gcf()` / `plt.gca()` used by standalone plot utilities.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `backends/` | Backend implementations and protocol |
| `_markers/` | Individual marker type implementations (Arrow, Circle, Line, etc.) |
| `_widgets/` | Low-level interactive widget implementations |

## For AI Agents

### Working In This Directory

- All rendering goes through the active backend: `get_backend().<method>()`.  **Never call `ax.transData`, `plt.gca()`, `ax.hspy_fig._background = None`, or similar MPL-specific calls from outside `backends/mpl/`.**
- Use `BackendCapabilityError` for features a backend does not yet support; callers in `markers.py` and `hse.py` degrade gracefully on this exception.
- Interactive updates rely on `events.py` — do not poll; connect/disconnect event handlers.
- Use `backend.render_figure_from_ax(ax)` for performance-critical repaints; use `backend.invalidate_blit_background(ax)` before `draw_patch()` when a patch is structurally removed/added.

### Testing Requirements

```bash
uv run pytest hyperspy/tests/drawing/
```

Use `matplotlib.use('Agg')` or the `mpl_cleanup` fixture for non-interactive test rendering.

## Dependencies

### Internal
- `hyperspy/events.py` — reactive event system
- `hyperspy/roi.py` — ROI widgets connect to drawing widgets

### External
- `matplotlib` — default rendering stack (via `backends/mpl/`)
- `anyplotlib` — optional alternative backend (via `backends/anyplotlib/`)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
