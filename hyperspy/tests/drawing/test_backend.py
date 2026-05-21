from hyperspy.drawing.backends import get_backend
from hyperspy.drawing.backends._protocol import PlottingBackend


def test_default_backend_satisfies_protocol():
    backend = get_backend()
    assert isinstance(backend, PlottingBackend)
