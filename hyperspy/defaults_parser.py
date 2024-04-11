# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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


class PlotConfig(t.HasTraits):
    # Don't use t.Enum to list all possible matplotlib colormap to
    # avoid importing matplotlib and building the list of colormap
    # when importing hyperpsy
    widget_plot_style = t.Enum(
        ["horizontal", "vertical"], label="Widget plot style: (only with ipympl)"
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
    dims_024_increase = t.Str("right", label="Navigate right")
    dims_024_decrease = t.Str(
        "left",
        label="Navigate left",
    )
    dims_135_increase = t.Str(
        "down",
        label="Navigate down",
    )
    dims_135_decrease = t.Str(
        "up",
        label="Navigate up",
    )
    modifier_dims_01 = t.Enum(
        [
            "ctrl",
            "alt",
            "shift",
            "ctrl+alt",
            "ctrl+shift",
            "alt+shift",
            "ctrl+alt+shift",
        ],
        label="Modifier key for 1st and 2nd dimensions",
    )  # 0 elem is default
    modifier_dims_23 = t.Enum(
        [
            "shift",
            "alt",
            "ctrl",
            "ctrl+alt",
            "ctrl+shift",
            "alt+shift",
            "ctrl+alt+shift",
        ],
        label="Modifier key for 3rd and 4th dimensions",
    )  # 0 elem is default
    modifier_dims_45 = t.Enum(
        [
            "alt",
            "ctrl",
            "shift",
            "ctrl+alt",
            "ctrl+shift",
            "alt+shift",
            "ctrl+alt+shift",
        ],
        label="Modifier key for 5th and 6th dimensions",
    )  # 0 elem is default
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
