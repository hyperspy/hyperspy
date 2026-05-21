from hyperspy.drawing.backends import get_backend
from hyperspy.drawing.backends._protocol import PlottingBackend
from hyperspy.drawing.backends.mpl import MplBackend


def test_default_backend_satisfies_protocol():
    import hyperspy.drawing  # noqa: F401 triggers default backend registration

    backend = get_backend()
    assert isinstance(backend, PlottingBackend)


def test_mpl_backend_is_default():
    import hyperspy.drawing  # noqa: F401 — triggers registration

    backend = get_backend()
    assert isinstance(backend, MplBackend)


def test_backend_preference_change():
    """Switching the preference updates the active backend."""
    import hyperspy.drawing  # noqa: F401
    from hyperspy.defaults_parser import preferences
    from hyperspy.drawing.backends import get_backend
    from hyperspy.drawing.backends.mpl import MplBackend

    original = preferences.Plot.backend
    try:
        preferences.Plot.backend = "matplotlib"
        assert isinstance(get_backend(), MplBackend)
    finally:
        preferences.Plot.backend = original
