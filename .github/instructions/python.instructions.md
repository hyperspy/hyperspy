---
applyTo: "**/*.py"
---

# Python Code Review Standards — HyperSpy Specific

When reviewing Python files in this repository, apply these checks on top of the repository-wide standards.

## Lazy Evaluation Branching
Use `signal._lazy` flag to branch. Signal methods must handle both numpy and dask paths.

**Flag:** Code assuming numpy without checking `_lazy`. Dask ops missing chunk handling. `.compute()` on non-lazy data without guard. Methods missing dask path.

## Events System
After mutating `.data` in-place: `self.events.data_changed.trigger(obj=self)` required for UI reactivity.

**Flag:** In-place `.data` mutations without trigger. `trigger()` missing `obj=self`. Every `connect()` must have a corresponding `disconnect()` on `close()` — event listener leaks are HyperSpy's most common bug class (PR #3355, #3640, #3629, #3648). Test both connection AND disconnection.

## Deprecation Protocol
Use `@deprecated(since="...", alternative="...", removal="...")` from `hyperspy/decorators.py`. Never bare `warnings.warn()` — HyperSpy uses `VisibleDeprecationWarning`.

**Flag:** Bare `warnings.warn()` for API deprecation. `@deprecated` missing `alternative=`/`removal=`. Python's built-in `DeprecationWarning`.

## AxesManager Internals
`axes_manager._axes` in NumPy order (not display). Use `[::-1]` on `navigation_shape`, `signal_shape`, `indices` for array operations.

**Flag:** Code assuming `_axes` in display order. Mutating `_axes` without `_update_attributes()`. Missing `[::-1]` on `indices` for map indexing.

## Signal Subclass Guard
Methods with dimension requirements must raise `SignalDimensionError` at entry.

**Flag:** New methods that don't raise `SignalDimensionError` on wrong signal type.

## NumPy-vs-HyperSpy Methods
Prefer HyperSpy-native methods that preserve metadata: `signal.max(axis='energy')` not `np.max(signal.data)`. `signal.isig[100.:300.]` not `signal.data[:, 100:300]`. `signal / signal.max()` not `signal.data / signal.data.max()`.

**Flag raw NumPy on `.data` where a HyperSpy method exists.**

## _assign_subclass() Pattern
After data shape/dtype/dimension changes, `_assign_subclass()` recomputes signal class. Missing → wrong subclass.

**Flag data transformations without `_assign_subclass()`.**

## Extension & Config Stability
`hyperspy_extension.yaml` changes must be backward-compatible with `defaults_parser.py`.

**Flag:** Changes breaking existing installations. New required keys without migration notes.

## function_nd Vectorization
`Component.function_nd()` is the vectorized performance path. New components should implement it when feasible (PR #3476).

**Flag:** Missing `function_nd` where vectorized possible. Code not handling mixed components.

## Expression Component Pattern
New components should subclass `Expression` (not raw `Component`) for auto-gradients and less boilerplate (#2134, #2135, #2137, #2157). Check sympy reserved names: `gamma`, `beta`, `zeta` silently break Expression compilation (#2134, #2269).

**Flag:** Hand-coded `function()` where `Expression` works. Unverified parameter names in Expression subclasses.

## Canonical Inplace Pattern
`inplace` methods MUST: `sig = self if inplace else self.deepcopy()`, operate on `sig`, `return sig if not inplace` (PR #3590). **Flag deviations.**

## Re-entrance Guard Comments
Boolean guards (`_updating_twin`, `_updating_indices_from_drag`) suppress re-entrant callbacks. Reviewers require comments explaining each guard's purpose (PR #3631). **Flag guards without inline comments.**

## Private & Dynamic Docstrings
New private functions need docstrings (PR #3587). Dynamic docstrings (`_docstring.py` templates, `.format()`) often have rendering bugs (PR #3525, #3475) — verify in readthedocs PR build.

**Flag:** Private functions without docstrings. Dynamic docstring changes without rendered output check.
