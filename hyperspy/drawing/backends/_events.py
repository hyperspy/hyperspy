"""Backend-neutral input events.

Every backend converts its native input events into the dataclasses defined
here before calling a handler connected through ``connect_key_press``,
``connect_mouse_move``, ``connect_mouse_press``, ``connect_mouse_release`` or
``connect_pick``.  The generic drawing layer only ever reads these fields; it
never sees a backend's native event (which stays available as ``native`` for
backend-specific code and debugging).

Field names deliberately follow the vocabulary most HyperSpy code and
downstream widget subclasses already use (``xdata``, ``inaxes``, ``key``,
``button``), but their meaning is defined here, not by matplotlib:

``inaxes``
    The backend axes handle the event happened over, or ``None`` when the
    pointer is outside every axes.
``xdata``, ``ydata``
    Position in data coordinates of ``inaxes``; ``None`` outside an axes.
``x``, ``y``
    Position in the backend's display space (the ``CoordSpace.DISPLAY``
    space of :meth:`PlottingBackend.convert_coords`); ``None`` when unknown.
``button``
    ``1`` (left/primary), ``2`` (middle), ``3`` (right), or ``None`` for pure
    motion.  Backends that number buttons from 0 must shift them.
``key``
    Canonical key name, see :func:`canonical_key`.
``modifiers``
    Held modifier keys, a frozenset drawn from ``{"ctrl", "alt", "shift",
    "super"}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Modifier names, in the order they prefix a canonical key name.
MODIFIER_ORDER = ("ctrl", "alt", "super", "shift")

# Non-printable key names, mapped from the spellings backends commonly report
# (browser ``KeyboardEvent.key`` values, Qt names) to canonical names.  The
# canonical names are the ones HyperSpy's navigation preferences store.
_KEY_ALIASES = {
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "arrowdown": "down",
    "pageup": "pageup",
    "pagedown": "pagedown",
    "page_up": "pageup",
    "page_down": "pagedown",
    "prior": "pageup",
    "next": "pagedown",
    "enter": "enter",
    "return": "enter",
    "escape": "escape",
    "esc": "escape",
    "backspace": "backspace",
    "delete": "delete",
    "del": "delete",
    "tab": "tab",
    "home": "home",
    "end": "end",
    "insert": "insert",
    "control": "control",
    "ctrl": "control",
    "alt": "alt",
    "shift": "shift",
    "meta": "super",
    "os": "super",
    "super": "super",
    "cmd": "super",
}
_MODIFIER_ALIASES = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "shift": "shift",
    "meta": "super",
    "super": "super",
    "cmd": "super",
    "os": "super",
}


def canonical_modifiers(modifiers) -> frozenset:
    """Normalise an iterable of modifier names to the canonical frozenset."""
    if not modifiers:
        return frozenset()
    out = set()
    for m in modifiers:
        name = _MODIFIER_ALIASES.get(str(m).lower())
        if name is not None:
            out.add(name)
    return frozenset(out)


def canonical_key(name, modifiers=()) -> str | None:
    """Return the canonical name for a key press.

    The base name is lower-case for named keys (``"left"``, ``"pageup"``,
    ``"enter"``, ``"f5"``) and the character itself for printable keys
    (``"a"``, ``"A"``, ``"+"``).  Held modifiers prefix it in the order
    ``ctrl``, ``alt``, ``super``, ``shift``, joined with ``+``
    (``"ctrl+left"``, ``"ctrl+alt+right"``).  ``shift`` is only added to
    named keys: a shifted printable key already reports the shifted
    character.  A key that is itself a modifier is returned bare
    (``"shift"``, ``"control"``).

    Parameters
    ----------
    name : str or None
        The key name as reported by the backend.
    modifiers : iterable of str
        Modifier names held during the press, in any common spelling.
    """
    if name is None or name == "":
        return None
    name = str(name)
    lowered = name.lower()
    if lowered in _KEY_ALIASES:
        base = _KEY_ALIASES[lowered]
    elif len(lowered) > 1 and lowered[0] == "f" and lowered[1:].isdigit():
        base = lowered  # function keys: "F5" -> "f5"
    elif len(name) == 1:
        base = name  # printable character, case preserved
    else:
        base = lowered
    if base in ("control", "alt", "shift", "super"):
        return base
    mods = canonical_modifiers(modifiers)
    printable = len(base) == 1
    prefix = [
        m for m in MODIFIER_ORDER if m in mods and not (m == "shift" and printable)
    ]
    return "+".join(prefix + [base])


@dataclass(frozen=True)
class KeyEvent:
    """A key press.

    Attributes
    ----------
    key : str or None
        Canonical key name including modifier prefixes, see
        :func:`canonical_key`.
    inaxes : any
        Axes handle under the pointer, or ``None``.
    xdata, ydata : float or None
        Pointer position in data coordinates of ``inaxes``.
    native : any
        The backend's own event object.
    """

    key: str | None
    inaxes: Any = None
    xdata: float | None = None
    ydata: float | None = None
    native: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class PointerEvent:
    """A pointer move, press or release.

    Attributes
    ----------
    kind : {"move", "press", "release"}
    xdata, ydata : float or None
        Position in data coordinates of ``inaxes``.
    inaxes : any
        Axes handle under the pointer, or ``None``.
    x, y : float or None
        Position in display space.
    button : int or None
        ``1`` left, ``2`` middle, ``3`` right; ``None`` for motion without a
        pressed button.
    modifiers : frozenset of str
        Held modifiers from ``{"ctrl", "alt", "shift", "super"}``.
    dblclick : bool
        True for the second press of a double click.
    native : any
        The backend's own event object.
    """

    kind: str
    xdata: float | None = None
    ydata: float | None = None
    inaxes: Any = None
    x: float | None = None
    y: float | None = None
    button: int | None = None
    modifiers: frozenset = frozenset()
    dblclick: bool = False
    native: Any = field(default=None, repr=False, compare=False)

    @property
    def key(self) -> str | None:
        """Held modifiers as one ``+``-joined string (``"ctrl+shift"``).

        Provided for handlers written against the older event shape; new
        code should test ``"shift" in event.modifiers``.
        """
        mods = [m for m in MODIFIER_ORDER if m in self.modifiers]
        return "+".join(mods) if mods else None


@dataclass(frozen=True)
class PickEvent:
    """A press that hit a pickable widget artist.

    Only backends whose widgets are drawn as pickable artists emit pick
    events (see :class:`~.backends._protocol.ArtistBackend`); backends with
    native widgets report drags through ``connect_widget_drag`` instead.

    Attributes
    ----------
    artist : any
        The picked handle, as returned by the backend when the widget was
        created.
    xdata, ydata, inaxes, x, y, button, modifiers
        The pointer state at the press, as in :class:`PointerEvent`.
    native : any
        The backend's own event object.
    """

    artist: Any
    xdata: float | None = None
    ydata: float | None = None
    inaxes: Any = None
    x: float | None = None
    y: float | None = None
    button: int | None = None
    modifiers: frozenset = frozenset()
    native: Any = field(default=None, repr=False, compare=False)
