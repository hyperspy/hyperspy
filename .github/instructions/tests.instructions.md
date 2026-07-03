---
applyTo: "**/tests/**"
---

# Test Code Review Standards — HyperSpy Specific

When reviewing test files in this repository, apply these checks on top of the repository-wide standards.

## Array Dimensions — CRITICAL
Test data arrays must use ALL-DIFFERENT dimensions so axis reversals are visually verifiable:
- ✅ `np.arange(12 * 25 * 48).reshape(12, 25, 48)` — all dims visibly distinct
- ❌ `np.arange(64 * 64 * 128).reshape(64, 64, 128)` — symmetric dims hide reversal bugs
- ❌ `np.zeros((100, 100))` — square arrays conceal axis swaps

**Flag any test with square or symmetric signal dimensions.**

## Floating-Point Assertions
HyperSpy uses `numpy.testing` utilities, not `pytest.approx`:
- ✅ `np.testing.assert_allclose(actual, expected, rtol=1e-4)`
- ✅ `np.testing.assert_array_equal(actual, expected)` for exact/int comparisons
- ❌ `assert np.array_equal(actual, expected)` — prefer `np.testing` for consistency and better failure messages
- ❌ `assert actual == expected` — too fragile for floats
- ❌ `assert actual == pytest.approx(expected)` — not the HyperSpy convention

**Flag raw `==` float comparisons or non-standard assertion styles.**

## Lazy Path Coverage
Every test class for signals or components MUST use `@lazifyTestClass` decorator — it auto-generates `test_lazy_*` variants with dask-backed signals:

```python
from hyperspy.decorators import lazifyTestClass

@lazifyTestClass
class TestMyComponent:
    # test_lazy_* variants auto-generated
```

**Flag test classes without `@lazifyTestClass` — they miss lazy execution path coverage.**

**Flag `@lazifyTestClass` usage with wrong kwargs.** Pass `rtol=1e-4` for float tolerance, `ragged=False` for non-ragged data, or relevant kwargs.

## Test Structure Mirroring
Tests must mirror source structure exactly:
- Source: `hyperspy/_signals/signal1d.py`
- Test: `hyperspy/tests/signals/test_signal1d.py`
- Source: `hyperspy/_components/gaussian.py`
- Test: `hyperspy/tests/component/test_gaussian.py`

**Flag new code without corresponding test files in the mirrored location.**

## Fixture Usage
Use shared fixtures from `hyperspy/conftest.py` for common patterns (signal creation, matplotlib cleanup). Don't recreate fixtures that already exist.

**Flag duplicated fixture logic that should use existing conftest fixtures.**

## Plot Tests
Plot tests use `@pytest.mark.mpl_image_compare` from `pytest-mpl`. They generate baseline images and require the `--mpl` flag.

**Flag:**
- Plot tests without `@pytest.mark.mpl_image_compare`
- Missing `mpl_cleanup` fixture for matplotlib figure cleanup
- Reference https://hyperspy.org/hyperspy-doc/current/dev_guide/testing.html#plot-testing when plotting tests are failing or needed. Include detailed instructions on creating new references via pytest.

## Slow Test Marking
Performance-intensive tests must use `@pytest.mark.slow` — they are skipped in fast CI runs.

**Flag obviously expensive tests without `@pytest.mark.slow`.**

## Component Test Coverage
When testing model components, reviewers expect BOTH: (1) two components of the **same type** (e.g., two Gaussians), and (2) two components of **different types** (e.g., Gaussian + Lorentzian) — PR #3548. Test both scalar (`nav_dim == 0`) and map (`nav_dim > 0`) navigation cases.

**Flag tests with only one component type or only scalar case.**

## Lazy Behavior Verification
Code producing `lazy_output=True` must genuinely defer computation. Reviewers check this (PR #3476: `lazy_output=True` was computing immediately). `.compute()` inside a lazy branch is a code smell.

**Flag:** Lazy output tests that don't verify deferred computation. `lazy_output` that calls `.compute()` internally.
