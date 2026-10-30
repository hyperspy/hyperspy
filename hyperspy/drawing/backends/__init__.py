from hyperspy.drawing.backends._protocol import ArtistBackend, PlottingBackend

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
            "Call register_backend() or import hyperspy.drawing "
            "after the default backend is configured."
        )
    return _active_backend


def get_artist_backend() -> ArtistBackend | None:
    """Return the active backend if it implements :class:`ArtistBackend`.

    Returns ``None`` for backends with native widgets, so callers can skip
    artist-only features.
    """
    backend = get_backend()
    return backend if isinstance(backend, ArtistBackend) else None


def _tight_layout(fig) -> None:
    """Re-pack *fig* on artist backends; native backends lay out themselves."""
    artist_backend = get_artist_backend()
    if artist_backend is not None:
        artist_backend.tight_layout(fig)
