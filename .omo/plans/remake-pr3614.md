# Remake PR #3614 — Lazy Decomposition Overhaul

**Plan**: remake-pr3614
**Created**: 2026-06-04
**Status**: IN PROGRESS
**Base Branch**: `origin/RELEASE_next_minor`
**Source Branch**: `FIX_lazy_SVD_v2` (84 commits, 28 files, +5878/-267 — REJECTED)
**Fork**: `francisco-dlp/hyperspy`

---

## Rationale

PR #3614 was rejected for being far too large for human review. Reviewers explicitly requested:
1. **"Divide into separately scoped PRs"** — @CSSFrancis, @ericpre
2. **"Deprecations must use the `@deprecated` decorator with a removal version"** — @CSSFrancis
3. **Address 7 Copilot-identified bugs** (4 axis-order issues, 3 parity/performance issues)

This plan restructures the single monolith into **7 focused, sequentially-dependency-ordered PRs** (with housekeeping folded into PR1 and PR7) that build on each other.

## HyperSpy Critical Context

### Axis Convention
```
NumPy array shape:  (A, B, C, D)
Signal1D display:   (C, B, A | D)
Signal2D display:   (B, A | D, C)
```
**ALL underlying arrays use NumPy order** — masks from the API arrive in `navigation_shape` (display) order and MUST be transposed (`.T`) before use with dask/NumPy arrays. This is the root cause of Copilot bugs #3-6.

### @deprecated Decorator
```python
from hyperspy.decorators import deprecated
@deprecated(since="2.5", alternative="ORPCA.partial_fit", alternative_is_function=False, removal="3.0")
def fit(self, X, batch_size=None): ...
```
The branch currently uses bare `warnings.warn()` — must be replaced.

### Changelog Format
`upcoming_changes/<issue>.<type>.rst` — plain backticks ONLY, never `:meth:`, `:class:`, `:attr:`.

---

## PR Dependency Order

```
PR1 (bugfixes+housekeeping) → PR2 (deprecation) → PR3 (ISVD) → PR4 (svd_solver)
                                                          ↘
                                                        PR5 (lazy enhancements)
                                                          ↘
                                                        PR6 (lazy kwarg refactor)
                                                          ↘
                                                        PR7 (docs+housekeeping)
```

---

## TODOs

### PR 1: Bugfixes (target: RELEASE_next_patch, milestone: v2.4.1)

- [x] 1. Fix Poissonian noise normalisation silent no-op (#3607) — `hyperspy/_signals/lazy.py`: assign `coeff.map_blocks()` result back to data. Test: `test_lazy_decomposition.py -k poisson`. Changelog: `3607.bugfix.rst`.
- [x] 2. Fix signal left permanently unfolded after lazy SVD (#3608) — `hyperspy/_signals/lazy.py`: change `is False` to `= False`. Test: verify `_unfolded4decomposition` is False after decomposition. Changelog: `3608.bugfix.rst`.
- [x] 3. Fix navigation mask axis order for multi-dim navigation (#3609) — `hyperspy/_signals/lazy.py`: ravel mask in array-axis order matching `unfold()` C-order. Axis verification: confirm ravel order matches. Test: `TestLazyDecompositionMaskTypes` with 2D nav `(3,4)`. Changelog: `3609.bugfix.rst`.
- [x] 4. Fix `_block_iterator` reading only first signal chunk (#3610) — `hyperspy/_signals/lazy.py`: rechunk signal axis to single chunk before iteration. Test: `TestSubSignalChunking`. Changelog: `3610.bugfix.rst`.
- [x] 5. Fix ORNMF hang on negative-mean data (#3611) — `hyperspy/learn/_ornmf.py`: `abs(X.mean())` in `_setup()`, fix iteration cap 1e9→1e6. Test: ORNMF with negative-mean input. Changelog: `3611.bugfix.rst`.
- [x] 6. Fix NaN-fill guards using fragile string comparison (#3612) — `hyperspy/_signals/lazy.py`: skipped, not applicable to baseline (no NaN-fill string guards exist yet).
- [x] 7. Fix validation error messages — `hyperspy/learn/_mva.py`: change `navigation_size < 2` from `AttributeError` to `ValueError`. Test: verify correct exception type/message.
- [x] 8. Fix `get_bss_model()` corruption of `learning_results` — `hyperspy/learn/_mva.py`: work on local copies, don't mutate `lr.factors`/`lr.loadings`. Test: BSS model mutation regression.

**PR1 Acceptance**: all 8 pass, `pytest hyperspy/tests/learn/` clean, 100% cov, `ruff check` clean, branch `pr1-bugfixes`, labels `type:bug,type:bug-fix,status:needs review,milestone:v2.4.1`.

**PR1 Housekeeping**: fix URL encoding in `CHANGES.rst` (percent-encoded quotes in GitHub issue links) — included here since this is the first bugfix PR targeting `RELEASE_next_patch`.

### PR 2: ORPCA/ORNMF sklearn API + Deprecation (target: RELEASE_next_minor, milestone: v2.5.0)

- [x] 9. Add sklearn-compatible API to ORPCA (#3651)
- [x] 10. Add sklearn-compatible API to ORNMF (#3651)
- [x] 11. Deprecate legacy API using `@deprecated` decorator (#3651)
- [x] 12. Update internal callers to use new API (#3651)

**PR2 Acceptance**: all 4 pass, `pytest hyperspy/tests/learn/test_{rpca,ornmf}.py` clean, 100% cov, `ruff check` clean, branch `pr2-sklearn-api`, labels `type:API change,type:deprecation,status:needs review,milestone:v2.5.0`, `@deprecated` from `hyperspy/decorators.py` (NOT bare `warnings.warn`), removal version specified.

### PR 3: ISVD — Incremental SVD (target: RELEASE_next_minor, milestone: v2.5.0)

- [x] 13. Create `hyperspy/learn/incremental_svd.py` (#3652)
- [x] 14. Add ISVD tests (#3652)
- [x] 15. Add API reference docs (#3652)

**PR3 Acceptance**: all 3 pass, `pytest hyperspy/tests/learn/test_incremental_svd.py` clean, 100% cov, `ruff check` clean, branch `pr3-incremental-svd`, labels `type:new feature,status:needs review,milestone:v2.5.0`.

### PR 4: Lazy SVD Pipeline (target: RELEASE_next_minor, milestone: v2.5.0)

- [x] 16. Add `svd_solver` parameter (#3653)
- [x] 17. Add `centre` parameter (#3653)
- [x] 18. Add `reproject` parameter (#3653)
- [x] 19. Add `normalize_poissonian_noise()` standalone method (#3653)
- [x] 20. Add `_validate_decomposition_inputs()` shared helper (#3653)
- [x] 21. Add `_compute_explained_variance_ratio()` shared helper (#3653)
- [x] 22. Add `_nan_expand_rows()` shared helper (#3653)
- [x] 23. Add lazy SVD tests (#3653)

**PR4 Acceptance**: all 8 pass, `pytest hyperspy/tests/learn/test_lazy_decomposition.py` clean, 100% cov, `ruff check` clean, branch `pr4-lazy-svd`, labels `type:new feature,status:needs review,milestone:v2.5.0`, ALL mask operations verified against axis convention, non-obvious code commented.

### PR 5: Lazy Decomposition Enhancements (target: RELEASE_next_minor, milestone: v2.5.0)

- [x] 24. Add `algorithm='NMF'` support (#3653)
- [x] 25. Accept custom sklearn-like estimator objects (#3653)
- [x] 26. Add NMF and custom estimator tests (#3653)

**PR5 Acceptance**: all 3 pass, `pytest hyperspy/tests/learn/test_lazy_decomposition.py -k "NMF or custom"` clean, 100% cov, `ruff check` clean, branch `pr5-lazy-enhancements`, labels `type:new feature,status:needs review,milestone:v2.5.0`.

### PR 6: Lazy Model Reconstruction (target: RELEASE_next_minor, milestone: v2.5.0)

- [x] 27. Add `lazy`/`chunks` kwargs to `_calculate_recmatrix()` (#3653)
- [x] 28. Update `get_decomposition_model()` and `get_bss_model()` (#3653)
- [x] 29. Add lazy reconstruction tests (#3653)

**PR6 Acceptance**: all 3 pass, `pytest hyperspy/tests/learn/test_{decomposition,bss}.py` clean, 100% cov, `ruff check` clean, branch `pr6-lazy-reconstruct`, labels `type:enhancement,status:needs review,milestone:v2.5.0`.

### PR 7: Documentation Updates (target: RELEASE_next_minor, milestone: v2.5.0)

- [x] 30. Update `doc/user_guide/big_data.rst` (#3653)
- [x] 31. Update `doc/user_guide/mva/decomposition.rst` (#3653)
- [x] 32. Update `doc/user_guide/mva/bss.rst` (#3653)
- [x] 33. Update `doc/reference/base_classes/machine_learning.rst` (#3652)

**PR7 Acceptance**: all 4 pass, `cd doc && make html` builds clean, all code examples use ALL-DIFFERENT dimensions, no `:meth:`/`:class:`/`:attr:` in changelogs, branch `pr7-docs`, labels `type:doc,status:needs review,milestone:v2.5.0`.

**PR7 Housekeeping**: add conda.io and docs.conda.io to `linkcheck_ignore` in `doc/conf.py` — included here since this PR already touches documentation config.

---

## Final Verification Wave

- [x] F1. Goal/Constraint Verification — ALL PRs achieve same functionality as PR #3614, each independently reviewable, deprecation uses `@deprecated`, all 7 Copilot bugs addressed.
- [x] F2. Code Quality — non-obvious code commented (WHY), changelogs use plain backticks, ALL-DIFFERENT array dimensions, `ruff check` clean.
- [x] F3. Axis-Order Verification — every mask function checks array-axis vs navigation_shape ordering, asymmetric shapes used, unfold/fold ravel order matches mask flattening.
- [x] F4. Hands-On QA — 854 tests pass, ruff clean, import clean, no `Co-authored-by:` trailers.

---

## Key Instructions (from prior failed attempt)

1. Python env: conda's `hyperspy-dev` environment
2. Changelogs: plain backticks ONLY — NEVER `:meth:`, `:class:`, `:attr:` cross-references
3. Labels: bugfix → `type:bug,type:bug-fix,status:needs review,milestone:v2.4.1`; feature → `type:enhancement,status:needs review,milestone:v2.5.0`
4. Non-obvious code must be commented explaining WHY (axis reversal, lazy/deferred math, sklearn compat)
5. All branches pushed to `francisco-dlp` fork, PRs opened as drafts against `hyperspy/hyperspy`
6. CI must pass — fix any new failures, do not ignore pre-existing flaky infra failures
7. NEVER commit or push from planning — generate the plan, then await explicit go-ahead
