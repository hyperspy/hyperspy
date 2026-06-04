# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""

Functions that operate on Signal instances and other goodies.

    stack
        Stack Signal instances.

Subpackages:

    material
        Tools related to the material under study.
    plot
        Tools for plotting.
    eds
        Tools for energy-dispersive X-ray data analysis.

"""

import importlib


def print_known_signal_types(style=None):
    r"""Print all known `signal_type`\s

    This includes `signal_type`\s from all installed packages that
    extend HyperSpy.

    Parameters
    ----------
    style : prettytable style or None
        If None, the default prettytable style will be used.

    Examples
    --------
    >>> hs.print_known_signal_types() # doctest: +SKIP
    +--------------------+---------------------+--------------------+----------+
    |    signal_type     |       aliases       |     class name     | package  |
    +--------------------+---------------------+--------------------+----------+
    | DielectricFunction | dielectric function | DielectricFunction |  exspy   |
    |      EDS_SEM       |                     |   EDSSEMSpectrum   |  exspy   |
    |      EDS_TEM       |                     |   EDSTEMSpectrum   |  exspy   |
    |        EELS        |       TEM EELS      |    EELSSpectrum    |  exspy   |
    |      hologram      |                     |   HologramImage    | holospy  |
    |      MySignal      |                     |      MySignal      | hspy_ext |
    +--------------------+---------------------+--------------------+----------+

    """
    from prettytable import PrettyTable

    from hyperspy.misc.utils import display
    from hyperspy.ui_registry import ALL_EXTENSIONS

    table = PrettyTable()
    table.field_names = ["signal_type", "aliases", "class name", "package"]
    if style is not None:
        table.set_style(style)
    for sclass, sdict in ALL_EXTENSIONS["signals"].items():
        # skip lazy signals and non-data-type specific signals
        if sdict["lazy"] or sdict.get("hidden", False) or not sdict["signal_type"]:
            continue
        aliases = (
            ", ".join(sdict["signal_type_aliases"])
            if "signal_type_aliases" in sdict
            else ""
        )
        package = sdict["module"].split(".")[0]
        table.add_row([sdict["signal_type"], aliases, sclass, package])
        table.sortby = "class name"

    display(table)


def show_keybindings(platform=None):
    """Print all keyboard shortcuts with their current key bindings

    Displays a table grouped by category (Navigation, Plot Interaction,
    Widget Resize, Model Plot).  The shortcut column shows the actual
    key or key-chord currently active — change any entry by setting the
    corresponding ``hs.preferences.Plot.<name>`` trait.

    On macOS the shortcut column shows the physical keys to press
    (e.g. ``⌘+←/→`` instead of ``super+←/→``).

    Parameters
    ----------
    platform : {"macos", "standard", "auto"} or None, default None
        Override the display style for the shortcut column:

        - ``"macos"`` — use macOS symbols (⌃⌘⌥⇧) and MacBook key aliases.
        - ``"standard"`` — use raw ASCII modifier names (ctrl, shift, alt).
        - ``"auto"`` — detect from ``sys.platform``.
        - ``None`` (default) — use the current
          ``hs.preferences.Plot.platform_shortcuts`` preference, falling
          back to ``sys.platform`` when set to ``"auto"``.
          Useful when the server platform differs from the client keyboard
          (e.g. a Linux server with a macOS client).

    Examples
    --------
    >>> hs.show_keybindings() # doctest: +SKIP
    +------------------+-----------------+---------------------------------------+
    |    Category      |    Shortcut     |                Action                 |
    +------------------+-----------------+---------------------------------------+
    |   Navigation     |    Ctrl+←/→     |        Navigate dimension 0          |
    |   Navigation     |    Ctrl+↑/↓     |        Navigate dimension 1          |
    |   Navigation     |   Shift+←/→     |        Navigate dimension 2          |
    |   Navigation     |   Shift+↑/↓     |        Navigate dimension 3          |
    |   Navigation     |     pg up       |      Increase step multiplier        |
    |   Navigation     |    pg down      |      Decrease step multiplier        |
    | Plot Interaction |        e        |      Toggle second pointer           |
    | Plot Interaction |        h        |  Launch contrast adjustment tool     |
    | Plot Interaction |        l        |    Toggle log/linear scale           |
    |  Widget Resize   |        +        |       Increase widget size           |
    |  Widget Resize   |        -        |       Decrease widget size           |
    |  Widget Resize   |        x        |  Increase rectangle width            |
    |  Widget Resize   |        y        |  Decrease rectangle width            |
    |  Widget Resize   |      shift      |  Jump-to-click on span/cursor        |
    |    Model Plot    |        a        |  Toggle adjust-position lines        |
    |    Model Plot    |        s        |  Toggle plot-components visibility   |
    |    Model Plot    |        d        |    Toggle residual display           |
    +------------------+-----------------+---------------------------------------+

    Notes
    -----
    Keyboard shortcut preferences are in ``hs.preferences.Plot``.
    To change a shortcut, in a Jupyter notebook::

        hs.preferences.Plot.key_toggle_log = "shift+l"

    or edit your ``~/.hyperspy/hyperspyrc`` configuration file.
    """
    import sys

    from prettytable import PrettyTable

    from hyperspy.defaults_parser import preferences
    from hyperspy.misc.utils import display

    # Display style: None uses the platform_shortcuts preference
    # so that remote-client scenarios work correctly (e.g. Linux
    # server + macOS client with platform_shortcuts="macos").
    if platform is None:
        ps = preferences.Plot.platform_shortcuts
        is_macos = (ps == "macos") or (ps == "auto" and sys.platform == "darwin")
    elif platform == "macos":
        is_macos = True
    elif platform == "standard":
        is_macos = False
    elif platform == "auto":
        is_macos = sys.platform == "darwin"
    else:
        raise ValueError(
            f"platform must be 'macos', 'standard', 'auto', or None, got {platform!r}"
        )

    # Modifier key → macOS symbol (other platforms use the raw ASCII name).
    _MOD_SYMBOLS = {
        "ctrl": "\u2303",  # ⌃
        "super": "\u2318",  # ⌘
        "alt": "\u2325",  # ⌥
        "shift": "\u21e7",  # ⇧
    }

    # Keys that have no dedicated physical key on a MacBook.
    _MACBOOK_ALIASES = {
        "pageup": "fn+\u2191",  # fn+↑
        "pagedown": "fn+\u2193",  # fn+↓
        "home": "fn+\u2190",  # fn+←
        "end": "fn+\u2192",  # fn+→
    }

    def _display_key(raw):
        """Translate a raw config key into a platform-native display label."""
        if not is_macos:
            return raw

        # Full-string matches first (pageup, pagedown, etc.).
        if raw.lower() in _MACBOOK_ALIASES:
            return _MACBOOK_ALIASES[raw.lower()]

        # Modifier combos: "super+alt" → "⌘+⌥".
        parts = raw.split("+")
        translated = "+".join(_MOD_SYMBOLS.get(p, p) for p in parts)
        return translated

    def _mod(name):
        return getattr(preferences.Plot, name)

    # →/←/↑/↓ are Unicode arrow characters (not ASCII)
    ARROW_UP = "\u2191"
    ARROW_DOWN = "\u2193"
    ARROW_LEFT = "\u2190"
    ARROW_RIGHT = "\u2192"

    table = PrettyTable()
    table.field_names = ["Category", "Shortcut", "Action"]
    table.align = "l"
    table.align["Shortcut"] = "c"
    table.sortby = "Category"

    # ---- Navigation ----
    category = "Navigation"

    # Dimension navigation — 6 dims × 2 directions = 12 combos,
    # displayed as 6 rows (one per dimension, both directions)
    mod_names = ["modifier_dims_01", "modifier_dims_23", "modifier_dims_45"]

    for dim, mod_name in enumerate(mod_names):
        for sub_dims, arrow_pair in [
            ("024", (ARROW_LEFT, ARROW_RIGHT)),
            ("135", (ARROW_UP, ARROW_DOWN)),
        ]:
            dim_n = dim * 2 + (0 if sub_dims == "024" else 1)
            if dim_n > 5:
                continue
            modifier = _mod(mod_name)
            combo = _display_key(f"{modifier}+{arrow_pair[0]}/{arrow_pair[1]}")
            table.add_row([category, combo, f"Navigate dimension {dim_n}"])

    # Step multiplier
    step_inc = _mod("key_step_increase")
    step_dec = _mod("key_step_decrease")
    if step_inc == step_dec:
        table.add_row(
            [category, _display_key(step_inc), "Increase/decrease step multiplier"]
        )
    else:
        table.add_row([category, _display_key(step_inc), "Increase step multiplier"])
        table.add_row([category, _display_key(step_dec), "Decrease step multiplier"])

    # ---- Plot Interaction ----
    category = "Plot Interaction"
    plot_keys = {
        "key_toggle_pointer": "Toggle second pointer",
        "key_adjust_contrast": "Launch contrast adjustment tool",
        "key_toggle_log": "Toggle log/linear scale",
    }
    for name, action in plot_keys.items():
        table.add_row([category, _display_key(_mod(name)), action])

    # ---- Widget Resize ----
    category = "Widget Resize"
    widget_keys = {
        "key_widget_increase": "Increase widget size",
        "key_widget_decrease": "Decrease widget size",
        "key_rectangle_x_increase": "Increase rectangle width",
        "key_rectangle_x_decrease": "Decrease rectangle width",
        "key_rectangle_y_increase": "Increase rectangle height",
        "key_rectangle_y_decrease": "Decrease rectangle height",
        "key_jump_to_click": "Jump-to-click on span/cursor",
        "key_rotation_snap": "Snap to 30° during line rotation",
    }
    for name, action in widget_keys.items():
        table.add_row([category, _display_key(_mod(name)), action])

    # ---- Model Plot ----
    category = "Model Plot"
    model_keys = {
        "key_toggle_adjust_position": "Toggle adjust-position lines",
        "key_toggle_plot_components": "Toggle plot-components visibility",
        "key_toggle_residual": "Toggle residual display",
    }
    for name, action in model_keys.items():
        table.add_row([category, _display_key(_mod(name)), action])

    display(table)


__all__ = [
    "interactive",
    "markers",
    "model",
    "plot",
    "print_known_signal_types",
    "roi",
    "show_keybindings",
    "samfire",
    "stack",
    "transpose",
]


def __dir__():
    return sorted(__all__)


_import_mapping = {
    "interactive": ".interactive",
    "stack": ".misc.utils",
    "transpose": ".misc.utils",
}


def __getattr__(name):
    if name in __all__:
        if name in _import_mapping.keys():
            import_path = "hyperspy" + _import_mapping.get(name)
            return getattr(importlib.import_module(import_path), name)
        else:
            return importlib.import_module("." + name, "hyperspy.utils")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
