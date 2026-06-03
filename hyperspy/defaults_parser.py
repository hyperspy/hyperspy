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


import configparser
import logging
import os
import sys
import warnings
from pathlib import Path

import traits.api as t

from hyperspy.misc.ipython_tools import turn_logging_off, turn_logging_on
from hyperspy.ui_registry import add_gui_method

config_path = Path("~/.hyperspy").expanduser()
config_path.mkdir(parents=True, exist_ok=True)
defaults_file = Path(config_path, "hyperspyrc")

_logger = logging.getLogger(__name__)


if defaults_file.is_file():
    # Remove config file if obsolated
    with open(defaults_file) as f:
        if "Not really" in f.readline():
            # It is the old config file
            defaults_file_exists = False
        else:
            defaults_file_exists = True
    if not defaults_file_exists:
        # It actually exists, but is an obsoleted unsupported version of it
        # so we delete it.
        _logger.info("Removing obsoleted config file")
        os.remove(defaults_file)
else:
    defaults_file_exists = False

# Defaults template definition starts#####################################
# This "section" is all that has to be modified to add or remove sections and
# options from the defaults

# Due to https://github.com/enthought/traitsui/issues/23 the desc text as
# displayed in the tooltip get "Specifies" prepended.


class GeneralConfig(t.HasTraits):
    logger_on = t.CBool(
        False,
        label="Automatic logging (requires IPython)",
        desc="If enabled, HyperSpy will store a log in the current directory "
        "of all the commands typed",
    )

    show_progressbar = t.CBool(
        True,
        label="Show progress bar",
        desc="If enabled, show a progress bar when available",
    )

    dtb_expand_structures = t.CBool(
        True,
        label="Expand structures in DictionaryTreeBrowser",
        desc="If enabled, when printing DictionaryTreeBrowser (e.g. "
        "metadata), long lists and tuples will be expanded and any "
        "dictionaries in them will be printed similar to "
        "DictionaryTreeBrowser, but with double lines",
    )
    logging_level = t.Enum(
        [
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        ],
        desc="the log level of all hyperspy modules.",
    )

    nb_progressbar = t.CBool(True, desc="Attempt to use ipywidgets progressbar")

    def _logger_on_changed(self, old, new):
        if new is True:
            turn_logging_on()
        else:
            turn_logging_off()


class GUIs(t.HasTraits):
    enable_ipywidgets_gui = t.CBool(
        True,
        desc="Display ipywidgets in the Jupyter Notebook. "
        "Requires installing hyperspy_gui_ipywidgets.",
    )
    enable_traitsui_gui = t.CBool(
        True,
        desc="Display traitsui user interface elements. "
        "Requires installing hyperspy_gui_traitsui.",
    )


_IS_MACOS = sys.platform == "darwin"

# All valid modifier key combinations.  ``super`` is the Command key on macOS and
# the Windows/Logo key on other platforms.
_MODIFIER_OPTIONS = [
    "ctrl",
    "alt",
    "shift",
    "super",
    "ctrl+alt",
    "ctrl+shift",
    "alt+shift",
    "ctrl+alt+shift",
    "super+alt",
    "super+shift",
    "ctrl+super",
]


def _modifier_list(*defaults):
    """Return modifier options with the best platform default first."""
    # Use the second item ("macOS default") on Darwin; the first otherwise.
    default = defaults[1] if _IS_MACOS and len(defaults) > 1 else defaults[0]
    options = [m for m in _MODIFIER_OPTIONS if m != default]
    if _IS_MACOS:
        # ``super`` (Command ⌘) is unusable on macOS for navigation:
        # macOS captures ⌘+arrow for Mission Control and backends use
        # inconsistent key-event prefixes ("cmd", "ctrl") — never "super".
        options = [m for m in options if "super" not in m]
    return [default] + options


class PlotConfig(t.HasTraits):
    # Don't use t.Enum to list all possible matplotlib colormap to
    # avoid importing matplotlib and building the list of colormap
    # when importing hyperpsy
    widget_plot_style = t.Enum(
        ["horizontal", "vertical"], label="Widget plot style: (only with ipympl)"
    )
    use_subfigure = t.CBool(
        False,
        desc="EXPERIMENTAL. Plot navigator and signal on the same figure. "
        "Note that this is slower than using separate figures "
        "and it requires matplotlib >=3.9.",
    )
    cmap_navigator = t.Str(
        "gray",
        label="Color map navigator",
        desc="Set the default color map for the navigator.",
    )
    cmap_signal = t.Str(
        "gray",
        label="Color map signal",
        desc="Set the default color map for the signal plot.",
    )
    # ---- Navigation -------------------------------------------------------
    dims_024_increase = t.Str("right", label="Navigate right", group="Navigation")
    dims_024_decrease = t.Str(
        "left",
        label="Navigate left",
        group="Navigation",
    )
    dims_135_increase = t.Str(
        "down",
        label="Navigate down",
        group="Navigation",
    )
    dims_135_decrease = t.Str(
        "up",
        label="Navigate up",
        group="Navigation",
    )
    # ---- Modifier Keys ----------------------------------------------------
    # Each tuple: (linux/windows default, macOS default)
    # macOS uses only ``alt`` (Option) and ``shift`` for arrow navigation
    # because Command (⌘) and Control (⌃) + arrow are captured by the OS
    # (Mission Control / Spaces) before any application sees them.
    # Additionally, the macosx and Qt5 backends produce different modifier
    # strings for Command/Control (``cmd`` vs ``ctrl``), while ``alt`` and
    # ``shift`` are consistent across all backends.
    modifier_dims_01 = t.Enum(
        _modifier_list("ctrl", "alt"),
        label="Modifier key for 1st and 2nd dimensions",
        group="Navigation",
    )
    modifier_dims_23 = t.Enum(
        _modifier_list("shift", "shift"),
        label="Modifier key for 3rd and 4th dimensions",
        group="Navigation",
    )
    modifier_dims_45 = t.Enum(
        _modifier_list("alt", "alt+shift"),
        label="Modifier key for 5th and 6th dimensions",
        group="Navigation",
    )
    # --- platform shortcut presets ---
    # Convenience methods apply these sets atomically so users don't have
    # to set each modifier individually — especially useful when the machine
    # running HyperSpy (server) differs from the keyboard (client).
    _MACOS_SHORTCUT_DEFAULTS = {
        "modifier_dims_01": "alt",
        "modifier_dims_23": "shift",
        "modifier_dims_45": "alt+shift",
    }
    _STANDARD_SHORTCUT_DEFAULTS = {
        "modifier_dims_01": "ctrl",
        "modifier_dims_23": "shift",
        "modifier_dims_45": "alt",
    }

    def _apply_shortcut_preset(self, preset):
        """Set *all* platform-dependent modifier traits at once."""
        self.trait_set(True, **preset)

    def use_macos_shortcuts(self):
        """Apply macOS-friendly keyboard modifier defaults.

        Useful when the HyperSpy instance runs on a non-macOS server (e.g.
        remote Linux) but the keyboard is macOS.  Call once after import::

            hs.preferences.Plot.use_macos_shortcuts()
        """
        self._apply_shortcut_preset(self._MACOS_SHORTCUT_DEFAULTS)

    def use_standard_shortcuts(self):
        """Apply standard (Linux/Windows) keyboard modifier defaults.

        Useful when the HyperSpy instance runs on macOS but the keyboard is
        Linux/Windows (e.g. remote desktop, X11 forwarding).  Call once
        after import::

            hs.preferences.Plot.use_standard_shortcuts()
        """
        self._apply_shortcut_preset(self._STANDARD_SHORTCUT_DEFAULTS)

    # --- configurable platform preset ---
    platform_shortcuts = t.Enum(
        ["auto", "macos", "standard"],
        default="auto",
        label="Platform shortcut preset",
        desc="Override platform-specific keyboard modifiers. "
        "'auto' uses the detected platform, 'macos' and 'standard' "
        "apply the corresponding preset regardless of platform. "
        "Useful when the machine running HyperSpy differs from the "
        "client machine.",
    )

    @t.observe("platform_shortcuts")
    def _platform_shortcuts_changed(self, event):
        new = event.new
        if new == "auto":
            return
        if new == "macos":
            self.use_macos_shortcuts()
            if not _IS_MACOS:
                warnings.warn(
                    "Plot.platform_shortcuts is set to 'macos' but "
                    "sys.platform is %r (not macOS). This is expected if "
                    "connecting from a macOS client to a remote server." % sys.platform,
                )
        elif new == "standard":
            self.use_standard_shortcuts()
            if _IS_MACOS:
                warnings.warn(
                    "Plot.platform_shortcuts is set to 'standard' but "
                    "sys.platform is %r (macOS). This is expected if "
                    "connecting from a non-macOS client to a remote macOS "
                    "server." % sys.platform,
                )

    # ---- Plot Interaction -------------------------------------------------
    key_toggle_pointer = t.Str(
        "e",
        label="Toggle second pointer key",
        desc="Key to toggle the second pointer on/off in 1D signal plots.",
        group="Plot Interaction",
    )
    key_adjust_contrast = t.Str(
        "h",
        label="Adjust contrast tool key",
        desc="Key to launch the contrast adjustment tool in 2D image plots.",
        group="Plot Interaction",
    )
    key_toggle_log = t.Str(
        "l",
        label="Toggle log/linear key",
        desc="Key to toggle between logarithmic and linear norm or y-scale.",
        group="Plot Interaction",
    )
    key_widget_increase = t.Str(
        "+",
        label="Widget increase size key",
        desc="Key to increase the size of navigator cursors.",
        group="Widget Resize",
    )
    key_widget_decrease = t.Str(
        "-",
        label="Widget decrease size key",
        desc="Key to decrease the size of navigator cursors.",
        group="Widget Resize",
    )
    key_rectangle_x_increase = t.Str(
        "x",
        label="Rectangle x-size increase key",
        desc="Key to increase the x-size of a rectangle widget.",
        group="Widget Resize",
    )
    key_rectangle_x_decrease = t.Str(
        "c",
        label="Rectangle x-size decrease key",
        desc="Key to decrease the x-size of a rectangle widget.",
        group="Widget Resize",
    )
    key_rectangle_y_increase = t.Str(
        "y",
        label="Rectangle y-size increase key",
        desc="Key to increase the y-size of a rectangle widget.",
        group="Widget Resize",
    )
    key_rectangle_y_decrease = t.Str(
        "u",
        label="Rectangle y-size decrease key",
        desc="Key to decrease the y-size of a rectangle widget.",
        group="Widget Resize",
    )
    key_step_increase = t.Str(
        "pageup",
        label="Step increase key",
        desc="Key to increase the navigation step multiplier. "
        "On MacBooks, ``pageup`` is generated by ``fn+up``; "
        "set to ``fn+up`` if the backend supports it.",
        group="Navigation",
    )
    key_step_decrease = t.Str(
        "pagedown",
        label="Step decrease key",
        desc="Key to decrease the navigation step multiplier. "
        "On MacBooks, ``pagedown`` is generated by ``fn+down``; "
        "set to ``fn+down`` if the backend supports it.",
        group="Navigation",
    )
    key_jump_to_click = t.Str(
        "shift",
        label="Jump-to-click modifier key",
        desc="Modifier key held during a click to jump the pointer "
        "to the cursor position in 1D signal and 2D image plots.",
        group="Plot Interaction",
    )
    key_rotation_snap = t.Str(
        "shift",
        label="Rotation snap modifier key",
        desc="Modifier key held during rotation to snap line/wire "
        "rotation to 30-degree increments.",
        group="Plot Interaction",
    )
    # ---- Model Plot Shortcuts ---------------------------------------------
    key_toggle_adjust_position = t.Str(
        "a",
        label="Toggle adjust position key",
        desc="Key to toggle component position adjustment lines in 1D model plots.",
        group="Model Plot",
    )
    key_toggle_plot_components = t.Str(
        "s",
        label="Toggle plot components key",
        desc="Key to toggle component line visibility in 1D model plots.",
        group="Model Plot",
    )
    key_toggle_residual = t.Str(
        "d",
        label="Toggle residual line key",
        desc="Key to toggle the residual (Signal - Model) line in 1D model plots.",
        group="Model Plot",
    )
    pick_tolerance = t.CFloat(
        7.5, label="Pick tolerance", desc="The pick tolerance of ROIs in screen pixels."
    )


template = {
    "General": GeneralConfig(),
    "GUIs": GUIs(),
    "Plot": PlotConfig(),
}


# Set the enums defaults
template["General"].logging_level = "WARNING"
# Defaults template definition ends ######################################


def template2config(template, config):
    for section, traited_class in template.items():
        config.add_section(section)
        for key, item in traited_class.trait_get().items():
            config.set(section, key, str(item))


def config2template(template, config):
    for section, traited_class in template.items():
        config_dict = {}
        for name, value in config.items(section):
            if value == "True":
                value = True
            elif value == "False":
                value = False
            config_dict[name] = value
        traited_class.trait_set(True, **config_dict)


def dictionary_from_template(template):
    dictionary = {}
    for section, traited_class in template.items():
        dictionary[section] = traited_class.get()
    return dictionary


config = configparser.ConfigParser(allow_no_value=True)
template2config(template, config)
rewrite = False
if defaults_file_exists:
    # Parse the config file. It only copy to config the options that are
    # already defined. If the file contains any option that was not already
    # define the config file is rewritten because it is obsolate

    config2 = configparser.ConfigParser(allow_no_value=True)
    config2.read(defaults_file)
    for section in config2.sections():
        if config.has_section(section):
            for option in config2.options(section):
                if config.has_option(section, option):
                    config.set(section, option, config2.get(section, option))
                else:
                    rewrite = True
        else:
            rewrite = True

if not defaults_file_exists or rewrite is True:
    _logger.info("Writing the config file")
    with open(defaults_file, "w") as df:
        config.write(df)

# Use the traited classes to cast the content of the ConfigParser
config2template(template, config)


@add_gui_method(toolkey="hyperspy.Preferences")
class Preferences(t.HasTraits):
    General = t.Instance(GeneralConfig)
    GUIs = t.Instance(GUIs)
    Plot = t.Instance(PlotConfig)

    def save(self):
        config = configparser.ConfigParser(allow_no_value=True)
        template2config(template, config)
        config.write(open(defaults_file, "w"))


preferences = Preferences(
    General=template["General"],
    GUIs=template["GUIs"],
    Plot=template["Plot"],
)


if preferences.General.logger_on:
    turn_logging_on(verbose=0)


def file_version(fname):
    with open(fname, "r") as f:
        for line in f.readlines():
            if "__version__" in line:
                return line[line.find("=") + 1 :].strip()
    return "0"
