# Backend Abstraction — Implementation Plan

**Branch:** `alternate-backend`  
**Status:** Draft  
**Scope:** Finish the plotting backend abstraction so that (a) external packages can register
new backends (pyqtgraph, fastplotlib, napari, …) without modifying hyperspy's source, (b) the
generic drawing layer contains zero direct matplotlib imports, and (c) unsupported features are
handled gracefully rather than with bare `NotImplementedError`.

---

## Table of Contents

1. [Motivation & Goals](#motivation--goals)
2. [Terminology](#terminology)
3. [Specification](#specification)
   - [S1 – Protocol completeness](#s1--protocol-completeness)
   - [S2 – Explorer factory](#s2--explorer-factory)
   - [S3 – Entry-points registry](#s3--entry-points-registry)
   - [S4 – Extensible backend preference](#s4--extensible-backend-preference)
   - [S5 – Unsupported-feature handling](#s5--unsupported-feature-handling)
   - [S6 – Generic layer purity](#s6--generic-layer-purity)
   - [S7 – Widget & marker backend routing](#s7--widget--marker-backend-routing)
   - [S8 – Connection-ID lifecycle](#s8--connection-id-lifecycle)
   - [S9 – `anyplotlib` as optional dependency](#s9--anyplotlib-as-optional-dependency)
4. [Implementation Phases](#implementation-phases)
   - [Phase 0 – Bug fixes (no design changes)](#phase-0--bug-fixes-no-design-changes)
   - [Phase 1 – Protocol completeness](#phase-1--protocol-completeness)
   - [Phase 2 – Explorer factory](#phase-2--explorer-factory)
   - [Phase 3 – Entry-points registry & extensible preference](#phase-3--entry-points-registry--extensible-preference)
   - [Phase 4 – Generic layer purity](#phase-4--generic-layer-purity)
   - [Phase 5 – Widget & marker routing](#phase-5--widget--marker-routing)
   - [Phase 6 – Unsupported-feature polish](#phase-6--unsupported-feature-polish)
5. [Test Specification](#test-specification)
6. [Compliance Checklist](#compliance-checklist)
7. [File Change Map](#file-change-map)

---

## Motivation & Goals

The PR already extracts a `PlottingBackend` protocol and routes most drawing calls through it.
The audit identified gaps that prevent a new backend from working end-to-end without modifying
hyperspy core:

| Issue | Blocker for new backends? |
|-------|--------------------------|
| Explorer classes hardcoded to MPL in `signal.py` | **Yes** |
| `hasattr` duck-typing for anyplotlib-only methods | **Yes** |
| `plt.figure()` / `subfigures()` in `signal.py` | Yes (MPL import in generic code) |
| Hardcoded `if name == "anyplotlib"` dispatch | **Yes** |
| Closed enum blocks external backend names | **Yes** |
| `canvas.mpl_connect` in MPL subclasses | No (MPL code, acceptable) |
| `canvas.draw_idle()` in `mpl_hse.py` | No (MPL subclass) |
| MPL pick simulation in `WidgetBase` | Soft yes (degrades gracefully) |
| `on_figure_window_close` in `WidgetBase` | Yes (MPL canvas direct call) |
| Inline `hasattr(ax, "axvline")` in `VerticalLineWidget` | Yes |
| `remove_right_pointer` mutation-during-iteration bug | Bug (all backends) |
| `markers.py` imports `mpl_collections` | Soft (NotImplementedError today) |
| Key handler cids not stored/disconnected | Bug |
| No `@abstractmethod` on template methods | Low |

**Non-goals:** Rewriting the markers system (out of scope — they raise `NotImplementedError`
on non-MPL backends today and that is acceptable for a first pass). No UI/API surface changes
visible to end users.

---

## Terminology

- **Backend** — a class implementing `PlottingBackend` that wraps one rendering library.
- **Explorer** — a `HyperExplorer` subclass that orchestrates signal+navigator layout for a
  specific backend.
- **Entry point** — a `[project.entry-points."hyperspy.backends"]` declaration in a
  package's `pyproject.toml`.
- **Generic layer** — files that must work regardless of backend: `signal.py`, `figure.py`,
  `widget.py`, `he.py`, `hie.py`, `hse.py`, and `_widgets/`.

---

## Specification

### S1 – Protocol completeness

Every method called on the backend object **must** be declared in `PlottingBackend`.  
No code outside `backends/` may call `hasattr(backend, "some_method")` as a branching
condition.

Add the following methods to `PlottingBackend`, all with default no-op or `None`-return
implementations so existing backends that don't override them still satisfy the protocol:

```python
# Combined multi-panel layout (optional capability)
def create_combined_figure_panels(
    self, figsize=None
) -> tuple[Any, Any] | None:
    """Return (nav_fig, signal_fig) for a combined layout, or None.

    Backends that want to show navigator + signal in one window implement
    this.  Returning None tells signal.py to create two separate figures.
    """
    return None

# Post-plot display hook (optional capability — needed for Jupyter backends)
def ensure_displayed(self, fig: Any) -> None:
    """Called by signal.py after plot() completes.

    Backends that defer display (e.g. anyplotlib awaiting panel countdown)
    implement this to force the final render.
    """

# Close-event connection (needed by WidgetBase.connect)
def connect_close_event(self, fig: Any, fn: Callable) -> Any:
    """Connect fn to the figure's close/destroy event; return a cid."""

# Explorer factory (see S2)
def get_explorer(self, signal_dim: int) -> type[HyperExplorer]:
    """Return the HyperExplorer subclass appropriate for signal_dim.

    signal_dim == 0  →  base HyperExplorer (0-D / navigation-only)
    signal_dim == 1  →  HyperSignal1D_Explorer subclass
    signal_dim == 2  →  HyperImage_Explorer subclass
    """
```

`MplBackend` implements all four.  `AnyplotlibBackend` implements
`create_combined_figure_panels`, `ensure_displayed`, and `get_explorer`; it inherits the
no-op `connect_close_event` for now (anyplotlib close handling is done via `on_close=`
kwarg already).

---

### S2 – Explorer factory

`signal.py` must not import any backend-specific explorer class.  Instead:

```python
# signal.py — current (wrong)
from hyperspy.drawing.backends.mpl.mpl_hse import MPL_HyperSignal1D_Explorer
self._plot = MPL_HyperSignal1D_Explorer()

# signal.py — target
from hyperspy.drawing.backends import get_backend
self._plot = get_backend().get_explorer(axes_manager.signal_dimension)()
```

Each backend's `get_explorer` returns its own subclass:

| Backend | `signal_dim` | Returns |
|---------|-------------|---------|
| `MplBackend` | 0 | `MPL_HyperExplorer` |
| `MplBackend` | 1 | `MPL_HyperSignal1D_Explorer` |
| `MplBackend` | 2 | `MPL_HyperImage_Explorer` |
| `AnyplotlibBackend` | 0 | `HyperExplorer` (base, sufficient for 0-D) |
| `AnyplotlibBackend` | 1 | `Apl_HyperSignal1D_Explorer` (new) |
| `AnyplotlibBackend` | 2 | `Apl_HyperImage_Explorer` (new) |
| `PyQtGraphBackend` (external) | any | `PG_Hyper*Explorer` defined in that package |

A backend that only supports 1-D signals may raise `ValueError` for `signal_dim=2` —
`signal.py` lets this propagate (same as today's "Plotting not supported" path).

---

### S3 – Entry-points registry

External backends register under the `"hyperspy.backends"` entry-point group:

```toml
# In an external package's pyproject.toml:
[project.entry-points."hyperspy.backends"]
pyqtgraph = "hyperspy_pyqtgraph.backend:PyQtGraphBackend"
```

HyperSpy's own built-in backends are listed in its own `pyproject.toml`:

```toml
[project.entry-points."hyperspy.backends"]
matplotlib  = "hyperspy.drawing.backends.mpl:MplBackend"
anyplotlib  = "hyperspy.drawing.backends.anyplotlib:AnyplotlibBackend"
```

`drawing/__init__.py` becomes:

```python
import importlib.metadata

def _load_backend(name: str):
    eps = importlib.metadata.entry_points(group="hyperspy.backends")
    matched = [ep for ep in eps if ep.name == name]
    if not matched:
        known = [ep.name for ep in eps]
        raise ValueError(
            f"Unknown backend {name!r}. "
            f"Available: {known}. "
            "External backends must declare a 'hyperspy.backends' entry point."
        )
    return matched[0].load()()   # instantiate

def _on_backend_pref_change(change=None):
    name = change.new if change is not None else _pref.Plot.backend
    _register_backend(_load_backend(name))
```

This replaces the current `if name == "anyplotlib": ... else: MplBackend()` block entirely.

---

### S4 – Extensible backend preference

`defaults_parser.py` currently uses a closed `t.Enum`:

```python
backend = t.Enum(["matplotlib", "anyplotlib"], ...)   # blocks external names
```

Replace with `t.Str` and validate against discovered entry points at set-time:

```python
backend = t.Str("matplotlib", label="Plotting backend", desc="...")

@t.observe("backend")
def _validate_backend(self, change):
    from hyperspy.drawing.backends._registry import available_backends
    if change.new not in available_backends():
        raise t.TraitError(
            f"{change.new!r} is not a registered backend. "
            f"Available: {available_backends()}"
        )
```

`available_backends()` is a helper in `backends/_registry.py`:

```python
import importlib.metadata

def available_backends() -> list[str]:
    return [ep.name for ep in importlib.metadata.entry_points(group="hyperspy.backends")]
```

---

### S5 – Unsupported-feature handling

Replace bare `raise NotImplementedError(...)` in non-MPL backends with a typed exception
that callers can catch:

```python
# hyperspy/drawing/backends/_protocol.py (add near top)
class BackendCapabilityError(NotImplementedError):
    """Raised when the active backend does not support a requested feature.

    Callers may catch this to degrade gracefully or show a user-facing warning.
    """
```

Usage in `AnyplotlibBackend`:

```python
from hyperspy.drawing.backends._protocol import BackendCapabilityError

def add_right_axis(self, ax, color="black"):
    raise BackendCapabilityError(
        "The anyplotlib backend does not support twin-y axes. "
        "The right-pointer feature is unavailable."
    )
```

Callers that are best-effort (e.g., the right-pointer toggle) catch `BackendCapabilityError`
and warn rather than crash:

```python
# hse.py
import warnings
from hyperspy.drawing.backends._protocol import BackendCapabilityError

def add_right_pointer(self, **kwargs):
    try:
        ...
        self._add_right_line(**kwargs)
    except BackendCapabilityError as e:
        warnings.warn(
            f"Right pointer not available with the current backend: {e}",
            UserWarning,
            stacklevel=2,
        )
        return
```

Do **not** add a `supports(feature)` predicate — that pattern leads to TOCTOU races and
duplicated logic. Catch `BackendCapabilityError` at the call site instead.

---

### S6 – Generic layer purity

Files that are **not** inside `backends/` must contain zero direct matplotlib imports.
Specifically:

| File | Violation | Fix |
|------|-----------|-----|
| `signal.py:3066` | `import matplotlib.pyplot as plt; plt.figure(...)` | Remove the `use_subfigure` MPL block; `create_combined_figure_panels` covers the same need backend-agnostically |
| `widget.py:208` | `from matplotlib.backend_bases import MouseEvent, PickEvent` | Move `select()` into `MplWidgetBase` in `backends/mpl/` |
| `widget.py:228` | `on_figure_window_close` → `figure.canvas.mpl_connect(...)` | Replace with `get_backend().connect_close_event(ax.figure, self.close)` |
| `_widgets/vertical_line.py:43` | `if not hasattr(ax, "axvline")` | Remove; call `get_backend().add_vline_widget(ax, x, color)` unconditionally; let `MplBackend` use `ax.axvline` internally |

The `use_subfigure` preference in `PlotConfig` should be **deprecated** and its code path
removed. `create_combined_figure_panels` returning non-`None` is the backend-agnostic
equivalent. `MplBackend.create_combined_figure_panels` may internally use subfigures if
`preferences.Plot.use_subfigure` is True during a transition period, but `signal.py` only
calls the protocol method.

---

### S7 – Widget & marker backend routing

**`WidgetBase`** changes:

1. Remove the `select()` method from `WidgetBase`. Add an overrideable `_do_select()` hook:
   ```python
   def _do_select(self):
       pass  # no-op in base; MPL subclass simulates a pick event
   ```
   `MplWidgetBase` (new, in `backends/mpl/`) overrides `_do_select` with the current
   `matplotlib.backend_bases` code.  All existing MPL widget classes inherit from
   `MplWidgetBase`.

2. Replace `on_figure_window_close` call:
   ```python
   # Before
   from hyperspy.drawing.utils import on_figure_window_close
   on_figure_window_close(ax.figure, self.close)
   # After
   get_backend().connect_close_event(ax.figure, self.close)
   ```

**`VerticalLineWidget._add_patch_to`**: remove the `if not hasattr(ax, "axvline")` block
entirely. Both MPL and non-MPL backends implement `add_vline_widget` in the protocol; let
each backend handle the implementation detail.

**Markers (`markers.py`)**: no change in this phase. The `BackendCapabilityError` raised by
non-MPL backends is the correct behaviour for now. A future phase can introduce a
`BackendMarkerCollection` protocol.

---

### S8 – Connection-ID lifecycle

Every `canvas.mpl_connect` / `backend.connect_*` call in the MPL explorer subclasses must
store the returned cid and disconnect it in `close()`.  

Current violations:
- `mpl_he.py:_connect_key_nav` — stores nothing, leaks on close
- `mpl_hse.py:_connect_key_handler` — stores nothing, leaks on close

Pattern to follow (already correct in some places):

```python
class MPL_HyperExplorer(HyperExplorer):
    def __init__(self):
        super().__init__()
        self._key_nav_cids: list[tuple[Any, Any]] = []   # (canvas, cid)

    def _connect_key_nav(self, figure):
        if figure.figure is not None and self.axes_manager.navigation_axes:
            cid = get_backend().connect_key_press(
                figure.figure, self.axes_manager.key_navigator
            )
            self._key_nav_cids.append((figure.figure, cid))

    def close(self):
        for fig, cid in self._key_nav_cids:
            get_backend().disconnect_event(fig, cid)
        self._key_nav_cids.clear()
        super().close()
```

---

### S9 – `anyplotlib` as optional dependency

`pyproject.toml` currently lists `anyplotlib` as a **required** dependency.  It should be
optional so that users who only use matplotlib do not need it:

```toml
# pyproject.toml
[project.optional-dependencies]
anyplotlib = ["anyplotlib"]

# Remove "anyplotlib" from [project.dependencies]
```

Imports of anyplotlib inside `AnyplotlibBackend` are already guarded by `import anyplotlib`
inside each method.  The backend module itself may be imported; the `import anyplotlib` only
fires when the backend is actually used.

---

## Implementation Phases

Each phase is independently mergeable.  Complete Phase 0 first (no API changes, just bug
fixes), then proceed in order.

---

### Phase 0 – Bug fixes (no design changes)

**Goal:** fix correctness bugs with zero API or architecture changes.

| File | Change |
|------|--------|
| `hse.py:129` | Fix `remove_right_pointer` — use `list(self.signal_plot.right_ax_lines)` snapshot |
| `mpl_he.py` | Store key-nav cids; disconnect in `close()` |
| `mpl_hse.py:96` | Replace `self.signal_plot.figure.canvas.draw_idle()` with `get_backend().draw_idle(self.signal_plot.figure)` |
| `mpl_hse.py:78` | Replace `canvas.mpl_connect(...)` with `get_backend().connect_key_press(figure.figure, fn)` |
| `mpl_he.py:130` | Replace `canvas.mpl_connect(...)` with `get_backend().connect_key_press(figure.figure, ...)` |
| `he.py`, `hse.py`, `hie.py` | Add `from abc import ABC, abstractmethod`; annotate abstract methods |

**Tests added in Phase 0:**
- `test_remove_right_pointer_removes_all_lines` — assert all lines removed, not every other
- `test_key_nav_cids_disconnected_on_close` — connect mock handler, close, assert disconnected

---

### Phase 1 – Protocol completeness

**Goal:** remove all `hasattr(backend, ...)` guards from `signal.py`.

**Changes:**

1. `_protocol.py` — add `create_combined_figure_panels`, `ensure_displayed`,
   `connect_close_event`, `get_explorer` (with default implementations as shown in S1).

2. `backends/mpl/__init__.py` (`MplBackend`) — implement all four new methods:
   - `connect_close_event`: `return fig.canvas.mpl_connect("close_event", lambda e: fn())`
   - `get_explorer`: return appropriate `MPL_Hyper*Explorer` class
   - `create_combined_figure_panels`: return `None` by default; when
     `preferences.Plot.use_subfigure` is True, create two matplotlib `SubFigure` objects and
     return them (migrate the current `signal.py` logic here)
   - `ensure_displayed`: no-op

3. `backends/anyplotlib/__init__.py` (`AnyplotlibBackend`) — `get_explorer` returns
   `HyperExplorer` / `HyperSignal1D_Explorer` / `HyperImage_Explorer` base classes until
   anyplotlib-specific subclasses are written; `connect_close_event` → no-op.

4. `signal.py` — replace `hasattr(_backend, "create_combined_figure_panels")` with direct
   call; replace `hasattr(_backend, "ensure_displayed")` with direct call; remove the
   `use_subfigure` block (now inside `MplBackend.create_combined_figure_panels`).

5. `_protocol.py` — add `BackendCapabilityError` class.

**Tests added in Phase 1:**
- `test_protocol_has_all_required_methods` — inspect `PlottingBackend` attrs; assert each
  method name exists
- `test_mpl_backend_create_combined_returns_tuple_when_use_subfigure` — set
  `preferences.Plot.use_subfigure = True`, call `create_combined_figure_panels`, assert tuple
- `test_mpl_backend_create_combined_returns_none_default` — assert `None` when
  `use_subfigure` is False
- `test_mpl_backend_get_explorer_signal_dim_0_1_2` — assert correct class for each dim
- `test_connect_close_event_fires_on_plt_close` — connect a mock fn, call `plt.close(fig)`,
  assert fn was called
- `test_backend_capability_error_is_notimplementederror` — `issubclass(BackendCapabilityError, NotImplementedError)`

---

### Phase 2 – Explorer factory

**Goal:** remove all `from hyperspy.drawing.backends.mpl.*` imports from `signal.py`.

**Changes:**

1. `signal.py:3071–3097` — replace the three `if signal_dim == 0 / 1 / 2` import blocks:

   ```python
   if axes_manager.signal_dimension > 2:
       raise ValueError(...)   # unchanged
   self._plot = get_backend().get_explorer(axes_manager.signal_dimension)()
   ```

2. `backends/mpl/__init__.py` — complete `get_explorer` (already done in Phase 1).

3. `backends/anyplotlib/__init__.py` — create `Apl_HyperSignal1D_Explorer` and
   `Apl_HyperImage_Explorer` as minimal subclasses of the generic base classes, implementing
   `_make_signal_figure`, `_make_image_figure`, `_connect_key_handler`, `_add_right_line`,
   `_redraw_signal_figure` using `get_backend()` calls or `BackendCapabilityError` where not
   yet possible.

**Tests added in Phase 2:**
- `test_signal_plot_uses_backend_explorer` — set a mock backend with known `get_explorer`
  return value; call `signal.plot()`; assert the mock class was instantiated
- `test_signal_plot_with_anyplotlib_explorer` — with anyplotlib backend, `signal.plot()` must
  not raise; `isinstance(self._plot, HyperExplorer)` must be True

---

### Phase 3 – Entry-points registry & extensible preference

**Goal:** external packages can register and use a custom backend without modifying hyperspy.

**Changes:**

1. Create `hyperspy/drawing/backends/_registry.py`:

   ```python
   import importlib.metadata

   def available_backends() -> list[str]:
       return [
           ep.name
           for ep in importlib.metadata.entry_points(group="hyperspy.backends")
       ]

   def load_backend(name: str):
       eps = importlib.metadata.entry_points(group="hyperspy.backends")
       matched = [ep for ep in eps if ep.name == name]
       if not matched:
           raise ValueError(
               f"Unknown backend {name!r}. Available: {[e.name for e in eps]}. "
               "External backends must declare a 'hyperspy.backends' entry point."
           )
       return matched[0].load()()
   ```

2. `pyproject.toml` — add entry points for built-in backends:

   ```toml
   [project.entry-points."hyperspy.backends"]
   matplotlib = "hyperspy.drawing.backends.mpl:MplBackend"
   anyplotlib  = "hyperspy.drawing.backends.anyplotlib:AnyplotlibBackend"
   ```

3. `drawing/__init__.py` — replace `if name == "anyplotlib"` dispatch:

   ```python
   from hyperspy.drawing.backends._registry import load_backend

   def _on_backend_pref_change(change=None):
       name = change.new if change is not None else _pref.Plot.backend
       _register_backend(load_backend(name))
   ```

4. `defaults_parser.py` — change `t.Enum(["matplotlib", "anyplotlib"])` to `t.Str("matplotlib")`.
   Remove the `desc` hardcoding of "anyplotlib".  Validation against `available_backends()`
   is added to `PlotConfig._validate_backend`.

5. `pyproject.toml` — move `anyplotlib` from `[project.dependencies]` to
   `[project.optional-dependencies]` under the key `anyplotlib`.

**Tests added in Phase 3:**
- `test_available_backends_includes_matplotlib_and_anyplotlib` — assert both names present
- `test_load_backend_matplotlib` — returns `MplBackend` instance
- `test_load_backend_unknown_raises_valueerror` — `load_backend("nonexistent")` → `ValueError`
- `test_external_backend_via_entry_point` — use `importlib.metadata` mocking to inject a fake
  entry point; assert `load_backend("fake")` instantiates the fake class; assert
  `preferences.Plot.backend = "fake"` does not raise `TraitError`
- `test_plot_config_str_backend_accepts_known_name` — `preferences.Plot.backend = "matplotlib"`
  succeeds
- `test_plot_config_str_backend_rejects_unknown_name` — `preferences.Plot.backend = "bogus"`
  raises `TraitError`

---

### Phase 4 – Generic layer purity

**Goal:** zero `import matplotlib` in files outside `backends/`.

**Changes:**

1. `signal.py` — remove `import matplotlib.pyplot as plt` (the `use_subfigure` block was
   already migrated into `MplBackend.create_combined_figure_panels` in Phase 1).

2. `widget.py` — replace `on_figure_window_close` call:
   ```python
   # Before
   from hyperspy.drawing.utils import on_figure_window_close
   on_figure_window_close(ax.figure, self.close)
   # After
   get_backend().connect_close_event(ax.figure, self.close)
   ```

3. `widget.py` — extract `select()` into a `_do_select()` hook:
   ```python
   class WidgetBase:
       def select(self):
           ...
           self._do_select()   # backend-specific pick simulation
           self.picked = False

       def _do_select(self):
           pass   # no-op; MPL subclass overrides
   ```
   Create `hyperspy/drawing/backends/mpl/mpl_widget.py`:
   ```python
   from hyperspy.drawing.widget import WidgetBase

   class MplWidgetMixin:
       def _do_select(self):
           from matplotlib.backend_bases import MouseEvent, PickEvent
           ...  # current select() body
   ```
   All existing widget classes in `_widgets/` that need pick simulation inherit from
   `MplWidgetMixin` in addition to their current base.

4. `_widgets/vertical_line.py` — remove `if not hasattr(ax, "axvline")` block; call
   `get_backend().add_vline_widget(ax, x, color)` and store the returned handle in
   `self._vline_handle`; `_update_patch_position` sets position via
   `get_backend().update_vline(self._vline_handle, x)`.

5. `figure.py` — the one remaining MPL import:
   ```python
   import matplotlib.figure   # only to test SubFigure
   ```
   Replace with duck-typing (`hasattr(figure, "figure")`) or move to `MplBackend`.

**Tests added in Phase 4:**
- `test_generic_files_have_no_matplotlib_import` — ast-parse each file in the generic layer;
  assert no `import matplotlib` or `from matplotlib` nodes
- `test_widget_connect_close_event_used` — mock `get_backend().connect_close_event`; attach
  widget to an ax; assert the mock was called
- `test_vertical_line_uses_backend_add_vline_widget` — mock backend; call
  `VerticalLineWidget.set_mpl_ax(ax)`; assert `add_vline_widget` was called on the mock

---

### Phase 5 – Widget & marker routing

**Goal:** all `_widgets/` subclasses work correctly with any backend that supports their
required primitives.

**Changes:**

1. All `_widgets/*.py` — audit each for any remaining `import matplotlib` or direct
   `ax.method()` calls outside of `MplWidgetMixin`; route each through the backend.

2. `backends/mpl/__init__.py` — ensure all backend methods called by widgets are implemented
   (most are already; verify `add_vline_widget`, `add_rect_widget`, `create_rect_patch`,
   `get_data_transform_inverse`, `transform_point`).

3. `backends/anyplotlib/__init__.py` — replace bare `raise NotImplementedError` on widget
   methods that anyplotlib now supports (if any); replace others with `BackendCapabilityError`.

4. `markers.py` — no structural changes yet; add a module-level guard:
   ```python
   from hyperspy.drawing.backends._protocol import BackendCapabilityError
   # In Markers.plot():
   try:
       get_backend().add_collection(ax, self._collection)
   except BackendCapabilityError:
       warnings.warn(
           "The active backend does not support markers. "
           "Markers will not be displayed.",
           UserWarning,
           stacklevel=2,
       )
       return
   ```

**Tests added in Phase 5:**
- `test_markers_warn_on_unsupported_backend` — use a stub backend that raises
  `BackendCapabilityError`; assert `UserWarning` is issued, no exception raised
- `test_vertical_line_widget_position_update_uses_backend` — mock backend; move widget;
  assert `update_vline` called with correct x value

---

### Phase 6 – Unsupported-feature polish

**Goal:** every `NotImplementedError` in non-MPL backends is replaced with
`BackendCapabilityError`; higher-level callers handle it gracefully.

**Changes:**

1. `backends/anyplotlib/__init__.py` — replace all remaining `raise NotImplementedError(...)` with
   `raise BackendCapabilityError(...)`.

2. `hse.py:add_right_pointer` — catch `BackendCapabilityError`; warn and return (see S5).

3. Add a `CAPABILITIES` section to the backend docstring documenting which features are
   supported (informational, not programmatic).

4. (Optional) Add `preferences.Plot.warn_on_unsupported_feature = True` to suppress warnings
   for users who intentionally use a limited backend.

**Tests added in Phase 6:**
- `test_anyplotlib_add_right_axis_raises_backend_capability_error`
- `test_add_right_pointer_warns_when_backend_unsupported` — with a stub backend, `right_pointer_on = True` emits `UserWarning`, does not raise

---

## Test Specification

All tests live in `hyperspy/tests/drawing/`.

### Unit tests (`test_backend_protocol.py`)

```python
def test_protocol_declares_all_required_methods():
    required = [
        "create_figure", "close_figure", "draw_idle", "supports_blit",
        "copy_background", "restore_background", "blit",
        "connect_draw_event", "disconnect_event",
        "create_axes", "set_xlabel", "set_ylabel", "set_title",
        "set_xlim", "set_ylim", "get_ylim", "get_xbound",
        "set_axis_off", "set_aspect", "add_right_axis", "remove_right_axis",
        "plot_line", "update_line", "remove_line", "set_line_props",
        "line_get_xdata", "line_get_color",
        "add_text", "update_text", "remove_text",
        "plot_image", "plot_mesh", "image_set_data", "image_set_extent",
        "image_set_clim", "image_set_norm", "get_image_handle",
        "add_colorbar", "colorbar_set_label", "colorbar_remove", "colorbar_redraw",
        "connect_key_press", "connect_mouse_move", "connect_mouse_press",
        "connect_mouse_release", "connect_pick",
        "draw_animated_artists",
        "add_vline_widget", "update_vline", "add_rect_widget", "update_rect",
        "remove_widget_patch", "set_patch_animated", "set_patch_color",
        "set_patch_alpha", "add_artist", "create_rect_patch",
        "get_data_transform_inverse", "transform_point",
        "add_collection", "collection_update", "collection_remove",
        # New in this spec:
        "create_combined_figure_panels", "ensure_displayed",
        "connect_close_event", "get_explorer",
    ]
    import inspect
    from hyperspy.drawing.backends._protocol import PlottingBackend
    members = {name for name, _ in inspect.getmembers(PlottingBackend)}
    for name in required:
        assert name in members, f"PlottingBackend missing: {name}"


def test_mpl_backend_satisfies_protocol():
    from hyperspy.drawing.backends._protocol import PlottingBackend
    from hyperspy.drawing.backends.mpl import MplBackend
    assert isinstance(MplBackend(), PlottingBackend)


def test_anyplotlib_backend_satisfies_protocol():
    pytest.importorskip("anyplotlib")
    from hyperspy.drawing.backends._protocol import PlottingBackend
    from hyperspy.drawing.backends.anyplotlib import AnyplotlibBackend
    assert isinstance(AnyplotlibBackend(), PlottingBackend)


def test_backend_capability_error_is_notimplementederror():
    from hyperspy.drawing.backends._protocol import BackendCapabilityError
    assert issubclass(BackendCapabilityError, NotImplementedError)


def test_default_backend_get_explorer_all_dims():
    import hyperspy.drawing  # noqa: F401
    from hyperspy.drawing.backends import get_backend
    from hyperspy.drawing.he import HyperExplorer
    b = get_backend()
    for dim in (0, 1, 2):
        cls = b.get_explorer(dim)
        assert issubclass(cls, HyperExplorer), f"dim={dim} returned non-HyperExplorer"
```

### Integration tests (`test_backend_integration.py`)

```python
@pytest.mark.parametrize("signal_dim", [1, 2])
def test_signal_plot_uses_backend_explorer(signal_dim, tmp_path):
    """signal.plot() must instantiate the class returned by get_explorer()."""
    import hyperspy.drawing  # noqa: F401 — register default
    import hyperspy.api as hs
    import numpy as np
    from hyperspy.drawing.backends import get_backend
    from unittest.mock import patch

    data = np.random.random((3, 4, 5, 6)) if signal_dim == 2 else np.random.random((3, 4, 5))
    s = hs.signals.Signal2D(data) if signal_dim == 2 else hs.signals.Signal1D(data)

    original_get_explorer = get_backend().get_explorer
    instantiated = []

    def tracking_get_explorer(dim):
        cls = original_get_explorer(dim)
        class Tracking(cls):
            def __init__(self_inner):
                super().__init__()
                instantiated.append(cls)
        return Tracking

    with patch.object(get_backend(), "get_explorer", tracking_get_explorer):
        s.plot(navigator="slider")

    assert len(instantiated) == 1
    assert issubclass(instantiated[0], get_backend().get_explorer(signal_dim))


def test_external_backend_via_entry_point(monkeypatch):
    """A fake backend registered via entry points can be set as the active backend."""
    import importlib.metadata
    from hyperspy.drawing.backends._registry import load_backend
    from hyperspy.drawing.backends._protocol import PlottingBackend

    class _FakeBackend:
        def get_explorer(self, dim): ...
        def create_combined_figure_panels(self, figsize=None): return None
        def ensure_displayed(self, fig): pass
        def connect_close_event(self, fig, fn): return None
        # (must satisfy runtime_checkable protocol — add remaining stubs)

    class _FakeEP:
        name = "fake"
        def load(self): return _FakeBackend

    def _fake_eps(group):
        real = importlib.metadata.entry_points(group=group)
        if group == "hyperspy.backends":
            return list(real) + [_FakeEP()]
        return real

    monkeypatch.setattr(importlib.metadata, "entry_points", _fake_eps)
    backend = load_backend("fake")
    assert isinstance(backend, _FakeBackend)


def test_load_backend_unknown_raises_valueerror():
    from hyperspy.drawing.backends._registry import load_backend
    with pytest.raises(ValueError, match="Unknown backend"):
        load_backend("__nonexistent__")


def test_right_pointer_warns_on_unsupported_backend(tmp_path):
    """BackendCapabilityError from add_right_axis → UserWarning, not crash."""
    import hyperspy.api as hs
    import numpy as np
    from hyperspy.drawing.backends._protocol import BackendCapabilityError
    from hyperspy.drawing.backends import get_backend
    from unittest.mock import patch

    s = hs.signals.Signal1D(np.random.random((4, 5, 8)))
    s.plot(navigator="slider")
    explorer = s._plot

    with patch.object(get_backend(), "add_right_axis",
                      side_effect=BackendCapabilityError("no twin")):
        with pytest.warns(UserWarning, match="Right pointer not available"):
            explorer.right_pointer_on = True
```

### Static purity tests (`test_generic_layer_purity.py`)

```python
import ast
import importlib
from pathlib import Path

GENERIC_FILES = [
    "hyperspy/signal.py",
    "hyperspy/drawing/figure.py",
    "hyperspy/drawing/widget.py",
    "hyperspy/drawing/he.py",
    "hyperspy/drawing/hie.py",
    "hyperspy/drawing/hse.py",
    "hyperspy/drawing/_widgets/vertical_line.py",
    "hyperspy/drawing/_widgets/range.py",
    "hyperspy/drawing/_widgets/rectangles.py",
    "hyperspy/drawing/_widgets/horizontal_line.py",
    "hyperspy/drawing/_widgets/scalebar.py",
    "hyperspy/drawing/_widgets/polygon.py",
]

@pytest.mark.parametrize("filepath", GENERIC_FILES)
def test_no_direct_matplotlib_import(filepath):
    """Generic-layer files must not import matplotlib directly."""
    repo_root = Path(__file__).parents[3]
    source = (repo_root / filepath).read_text()
    tree = ast.parse(source, filename=filepath)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("matplotlib"), (
                        f"{filepath}: direct `import matplotlib` at line {node.lineno}"
                    )
            else:
                if node.module and node.module.startswith("matplotlib"):
                    raise AssertionError(
                        f"{filepath}: `from matplotlib` import at line {node.lineno}"
                    )
```

### Registry tests (`test_backend_registry.py`)

```python
def test_available_backends_includes_builtins():
    from hyperspy.drawing.backends._registry import available_backends
    names = available_backends()
    assert "matplotlib" in names
    assert "anyplotlib" in names


def test_plot_config_accepts_known_backend():
    from hyperspy.defaults_parser import preferences
    original = preferences.Plot.backend
    try:
        preferences.Plot.backend = "matplotlib"   # must not raise
    finally:
        preferences.Plot.backend = original


def test_plot_config_rejects_unknown_backend():
    import traits.api as t
    from hyperspy.defaults_parser import preferences
    with pytest.raises(t.TraitError):
        preferences.Plot.backend = "__not_a_real_backend__"
```

---

## Compliance Checklist

Run this checklist before marking the PR ready for review.

### Protocol
- [ ] `PlottingBackend` declares all methods in the unit-test `required` list
- [ ] `MplBackend` passes `isinstance(MplBackend(), PlottingBackend)`
- [ ] `AnyplotlibBackend` passes `isinstance(AnyplotlibBackend(), PlottingBackend)`
- [ ] `BackendCapabilityError` is defined and is a subclass of `NotImplementedError`

### Explorer factory
- [ ] `signal.py` contains zero imports from `hyperspy.drawing.backends.mpl`
- [ ] `MplBackend.get_explorer(0/1/2)` returns the correct class
- [ ] `signal.plot()` uses `get_backend().get_explorer(...)()` to instantiate the explorer

### Entry-points registry
- [ ] `pyproject.toml` has `[project.entry-points."hyperspy.backends"]` for `matplotlib` and `anyplotlib`
- [ ] `drawing/__init__.py` has no hardcoded `if name == "anyplotlib"` dispatch
- [ ] `_registry.py` exists with `available_backends()` and `load_backend(name)`
- [ ] `load_backend("__nonexistent__")` raises `ValueError` with helpful message

### Extensible preference
- [ ] `PlotConfig.backend` is `t.Str`, not `t.Enum`
- [ ] Setting an unknown backend name raises `TraitError` at assignment time
- [ ] `anyplotlib` removed from `[project.dependencies]`; added under `[project.optional-dependencies]`

### Unsupported features
- [ ] All `NotImplementedError` in `AnyplotlibBackend` replaced with `BackendCapabilityError`
- [ ] `hse.add_right_pointer` catches `BackendCapabilityError` and emits `UserWarning`
- [ ] `markers.py` catches `BackendCapabilityError` from `add_collection` and emits `UserWarning`

### Generic layer purity
- [ ] `test_no_direct_matplotlib_import` passes for every file in `GENERIC_FILES`
- [ ] `signal.py` has no `import matplotlib` at any level
- [ ] `widget.py` has no `from matplotlib` outside of `_do_select()` override
- [ ] `_widgets/vertical_line.py` has no `if not hasattr(ax, "axvline")` check

### Widget routing
- [ ] `WidgetBase.connect` calls `get_backend().connect_close_event(...)`
- [ ] `VerticalLineWidget._add_patch_to` calls `get_backend().add_vline_widget(...)`
- [ ] `WidgetBase._do_select` is a no-op in base; `MplWidgetMixin._do_select` has MPL logic

### Connection lifecycle
- [ ] `MPL_HyperExplorer._connect_key_nav` stores cids; `close()` disconnects them
- [ ] `MPL_HyperSignal1D_Explorer._connect_key_handler` stores cids; `close()` disconnects them
- [ ] `test_key_nav_cids_disconnected_on_close` passes

### Bug fixes
- [ ] `hse.remove_right_pointer` uses `list(...)` snapshot before iterating
- [ ] `test_remove_right_pointer_removes_all_lines` passes

### Abstract methods
- [ ] `HyperExplorer`, `HyperSignal1D_Explorer`, `HyperImage_Explorer` use `ABC` +
  `@abstractmethod` on their template methods

### Tests
- [ ] All new tests in `hyperspy/tests/drawing/` pass under `pytest`
- [ ] `ruff check` reports zero new errors on all changed files
- [ ] `ruff format` applied

---

## File Change Map

| File | Phase | Action |
|------|-------|--------|
| `hyperspy/drawing/backends/_protocol.py` | 0, 1 | Add `BackendCapabilityError`; add 4 new protocol methods |
| `hyperspy/drawing/backends/_registry.py` | 3 | **New file** — `available_backends`, `load_backend` |
| `hyperspy/drawing/backends/__init__.py` | 3 | Remove `if name ==` dispatch; call `_registry.load_backend` |
| `hyperspy/drawing/backends/mpl/__init__.py` | 1, 2 | Add `get_explorer`, `connect_close_event`, `create_combined_figure_panels`, `ensure_displayed` |
| `hyperspy/drawing/backends/mpl/mpl_he.py` | 0, 4 | Store+disconnect key-nav cids; use `get_backend()` for connect calls |
| `hyperspy/drawing/backends/mpl/mpl_hse.py` | 0 | Use `get_backend()` for `draw_idle` and `connect_key_press` |
| `hyperspy/drawing/backends/mpl/mpl_widget.py` | 4 | **New file** — `MplWidgetMixin._do_select` |
| `hyperspy/drawing/backends/anyplotlib/__init__.py` | 1, 2, 6 | Implement `get_explorer`; add `Apl_Hyper*Explorer`; replace `NotImplementedError` with `BackendCapabilityError` |
| `hyperspy/drawing/he.py` | 0 | Add `ABC`, `@abstractmethod` |
| `hyperspy/drawing/hie.py` | 0 | Add `@abstractmethod` |
| `hyperspy/drawing/hse.py` | 0, 5 | Fix `remove_right_pointer`; catch `BackendCapabilityError` in `add_right_pointer`; add `@abstractmethod` |
| `hyperspy/drawing/widget.py` | 4 | Replace `on_figure_window_close` with `connect_close_event`; extract `_do_select` hook |
| `hyperspy/drawing/_widgets/vertical_line.py` | 4, 5 | Remove `if not hasattr(ax, "axvline")` block |
| `hyperspy/drawing/_widgets/*.py` | 5 | Audit remaining direct matplotlib calls |
| `hyperspy/drawing/markers.py` | 5 | Catch `BackendCapabilityError`; emit `UserWarning` |
| `hyperspy/drawing/__init__.py` | 1, 3 | Remove `if name == "anyplotlib"`; use `_registry.load_backend` |
| `hyperspy/drawing/figure.py` | 4 | Remove `import matplotlib.figure` SubFigure duck-type |
| `hyperspy/defaults_parser.py` | 3 | Change `t.Enum` → `t.Str`; add validator |
| `hyperspy/signal.py` | 1, 2 | Remove `plt.figure`/`subfigures`; remove `hasattr` guards; use `get_explorer` |
| `pyproject.toml` | 3, 9 | Add `entry-points`; move `anyplotlib` to optional |
| `hyperspy/tests/drawing/test_backend.py` | all | Expand with test cases listed per phase |
| `hyperspy/tests/drawing/test_backend_integration.py` | 2, 5 | **New file** |
| `hyperspy/tests/drawing/test_generic_layer_purity.py` | 4 | **New file** |
| `hyperspy/tests/drawing/test_backend_registry.py` | 3 | **New file** |
