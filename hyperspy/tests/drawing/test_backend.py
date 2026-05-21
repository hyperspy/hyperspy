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
