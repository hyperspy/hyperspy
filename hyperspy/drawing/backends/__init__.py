# hyperspy/drawing/backends/__init__.py
from hyperspy.drawing.backends._protocol import PlottingBackend

_active_backend: PlottingBackend | None = None


def register_backend(backend: PlottingBackend) -> None:
    """Set the active plotting backend."""
    global _active_backend
    _active_backend = backend


def get_backend() -> PlottingBackend:
    """Return the currently active plotting backend."""
    if _active_backend is None:
        raise RuntimeError(
            "No plotting backend registered. "
            "Import hyperspy.drawing to load the default matplotlib backend."
        )
    return _active_backend
