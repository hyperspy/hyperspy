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


def print_known_signal_types():
    r"""Print all known `signal_type`\s

    This includes `signal_type`\s from all installed packages that
    extend HyperSpy.

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
    for sclass, sdict in ALL_EXTENSIONS["signals"].items():
        # skip lazy signals and non-data-type specific signals
        if sdict["lazy"] or not sdict["signal_type"]:
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


__all__ = [
    "interactive",
    "markers",
    "model",
    "plot",
    "print_known_signal_types",
    "roi",
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
