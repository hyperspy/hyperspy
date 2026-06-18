# HyperSpy Code Review Standards

When performing a code review, check every change against these HyperSpy-specific rules. Flag violations as blocking.

## 1. Axis Convention — CRITICAL
HyperSpy reverses ALL dimensions vs NumPy order. Internal arrays (`.data`, `.map`) always NumPy order; display is reversed.

**Flag:** `axes_manager.indices` without `[::-1]` for array indexing. `navigation_shape` without `[::-1]` creating arrays. `.data[...]` in display order not NumPy. `parameter.map` indexed `[y, x]` (NumPy order) not display. Descending/reversed axes (negative `scale`) — many operations assume positive scale.

## 2. Public API Integrity
`hyperspy/api.py` `__all__` defines the public contract. Never modify casually.

**Flag:** Signature changes to `load()`, `stack()`, `transpose()`, `preferences`. New functions missing `__all__`/`_import_mapping`. New signal/component types not re-exported in `signals.py`/`components1d.py`/`components2d.py`.

## 3. Navigation Dimension Guard
Components MUST branch: `nav_dim == 0` → `param.value` (scalar); `nav_dim > 0` → `param.map["values"]` (array) + `param.map["is_set"] = True`.

**Flag any `estimate_parameters()` missing this guard.** Works on single-spectrum, fails silently on multi-dimensional.

## 4. Parameter Map Write Patterns
**Flag:** `map["is_set"] = True` without `[:]` — canonical is `map["is_set"][:] = True`. Same for `map["values"]`. Missing `is_set` after `values` set. Map writes without lazy (dask) guard.

## 5. AI Attribution
Pre-commit blocks `Co-authored-by:` for AI tools. Must use `Assisted-by: <tool>:<model>`.

**Flag any `Co-authored-by:` + AI tool (claude, copilot, cursor, codex, gemini, deepseek).**

## 6. Changelog Required
Every user-facing change needs `upcoming_changes/<issue>.<type>.rst`. Types: `new`, `bugfix`, `doc`, `deprecation`, `enhancements`, `api`, `maintenance`.

**Flag:** Modified `.py` files without changelog entry. Wrong type in filename. Developer-facing text instead of user-facing. Multi-paragraph entries for non-`new` types.

## 7. Test Quality
**Flag:** Symmetric test dimensions — use ALL-DIFFERENT dims. Raw `==` float compares — use `np.testing.assert_allclose`. Test classes missing `@lazifyTestClass`. New code without mirrored test files. (See `tests.instructions.md` for full rules.)

## 8. Post-Mutation Events
After in-place `.data` mutation: `self.events.data_changed.trigger(obj=self)` required for UI reactivity.

**Flag any in-place data mutation without this trigger.**

## 9. Code Hygiene
**Flag:** `# type: ignore`/`# noqa` without justification. Bare `except:`/`except Exception:` without logging. Raw NumPy on `.data` where HyperSpy methods exist. (See `python.instructions.md` for deprecation/events/lazy.)

## 10. Scope & Structure
**Flag:** PRs >15 commits touching unrelated modules — split for reviewability. Missing standard checklist (`docstring`, `user guide`, `changelog`, `tests`, `ready for review`). File moves/renames/splits without a change map.

## 11. AI Content Quality
**Flag:** AI-generated PRs summarizing WHAT without explaining WHY — reviewers need design rationale (PR #3622). References to non-existent files — common AI hallucination. Commit messages describing diff instead of intent. For bugfixes: does the fix address the root cause or just patch a symptom? Prefer minimal root-cause fixes over workaround guards (PR #2863). Check sibling methods for the same bug class (PR #3384).

## 12. Documentation & Dependencies
**Flag:** Docstring changes without rendered output check (PR #3548, #3525). Examples misplaced between gallery and user guide (PR #3587). New dependencies without maintenance vetting (PR #3621). Changes breaking downstream extensions (pyxem, exspy, lumispy) without noting impact.

## 13. Traits Patterns (Enthought `traits` >= 7.0)
HyperSpy uses Enthought `traits` (not Jupyter `traitlets`). All trait handlers must use the modern `@observe` pattern.

**Flag:**
- `on_trait_change()` — deprecated; use `self.observe(handler, name)` or `@t.observe("name")` decorator
- `_name_changed(self, old, new)` auto-discovery handlers — must use `@t.observe("name")` decorator with `(self, event=None)` signature
- `_name_fired(self)` button handlers — must use `@t.observe("name")` decorator with `(self, event=None)` signature
- `t.Unicode()` — deprecated; use `t.Str()`
- `t.Either()` — deprecated; use `t.Union()`
- `@t.observe("name")` decorator without corresponding signature change to `(self, event=None)` — the decorator passes an event object, not positional `(old, new)` args
- `@t.observe("name")` on a plain class (not `HasTraits`) — decorator won't register; use `self.observe()` in `__init__` with `hasattr` guard
- ROI validation/revert handlers using `self.trait = old` — must use `self.trait_setq(trait=old)` to prevent infinite recursion
- `observe()` with string paths like `"_axes.items.scale"` when traits may not exist on list items — must use expression API with `optional=True`
- Handlers that fire during `__init__` when they shouldn't — add `post_init=True` to `@t.observe`
