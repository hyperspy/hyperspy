"""Backend discovery and loading via importlib entry points.

External packages register under the ``"hyperspy.backends"`` group::

    [project.entry-points."hyperspy.backends"]
    pyqtgraph = "hyperspy_pyqtgraph.backend:PyQtGraphBackend"

Built-in backends (matplotlib, anyplotlib) are registered by hyperspy's own
pyproject.toml so they are discovered the same way as external ones.
"""

from __future__ import annotations

import importlib.metadata


def available_backends() -> list[str]:
    """Return the names of all registered plotting backends."""
    return [
        ep.name for ep in importlib.metadata.entry_points(group="hyperspy.backends")
    ]


def load_backend(name: str):
    """Instantiate and return the backend registered under *name*.

    Raises
    ------
    ValueError
        If no entry point with *name* exists in the ``"hyperspy.backends"``
        group.
    """
    eps = list(importlib.metadata.entry_points(group="hyperspy.backends"))
    matched = [ep for ep in eps if ep.name == name]
    if not matched:
        known = [ep.name for ep in eps]
        raise ValueError(
            f"Unknown plotting backend {name!r}. "
            f"Available: {known}. "
            "External backends must declare a 'hyperspy.backends' entry point "
            "in their package's pyproject.toml."
        )
    return matched[0].load()()
