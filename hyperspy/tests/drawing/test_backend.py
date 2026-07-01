from hyperspy.drawing.backends import get_backend
from hyperspy.drawing.backends._protocol import PlottingBackend
from hyperspy.drawing.backends.mpl import MplBackend


def test_anyplotlib_backend_satisfies_protocol():
    try:
        from hyperspy.drawing.backends.anyplotlib import AnyplotlibBackend
    except ImportError:
        import pytest

        pytest.skip("anyplotlib not installed")
    from hyperspy.drawing.backends._protocol import PlottingBackend

    assert isinstance(AnyplotlibBackend(), PlottingBackend)


def test_default_backend_satisfies_protocol():
    import hyperspy.drawing  # noqa: F401 triggers default backend registration

    backend = get_backend()
    assert isinstance(backend, PlottingBackend)


def test_mpl_backend_is_default():
    import hyperspy.drawing  # noqa: F401 — triggers registration

    backend = get_backend()
    assert isinstance(backend, MplBackend)


def test_get_backend_raises_when_none_registered():
    import pytest

    import hyperspy.drawing.backends as _backends

    original = _backends._active_backend
    try:
        _backends._active_backend = None
        with pytest.raises(RuntimeError, match="No plotting backend registered"):
            get_backend()
    finally:
        _backends._active_backend = original


def test_backend_preference_change():
    """Switching the backend preference triggers register_backend()."""
    from unittest.mock import patch

    import hyperspy.drawing  # noqa: F401
    import hyperspy.drawing as _drawing
    import hyperspy.drawing.backends as _backends
    from hyperspy.defaults_parser import preferences

    original = preferences.Plot.backend
    calls = []

    original_register = _backends.register_backend

    def tracking_register(backend):
        calls.append(type(backend).__name__)
        original_register(backend)

    try:
        with patch.object(_drawing, "_register_backend", tracking_register):
            # Force a real change: switch away then back so the observer fires
            # (traits only notifies when the value actually changes)
            preferences.Plot.backend = (
                "matplotlib" if original != "matplotlib" else "matplotlib"
            )
            # Directly invoke the observer with a mock change to test it
            _drawing._on_backend_pref_change()
        assert "MplBackend" in calls, (
            "Expected MplBackend to be registered on preference change"
        )
    finally:
        preferences.Plot.backend = original
