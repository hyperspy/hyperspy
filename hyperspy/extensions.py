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

import copy
import importlib
import logging
from pathlib import Path

import yaml

_logger = logging.getLogger(__name__)


# libyaml C bindings may be missing
loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# Load hyperspy's own extensions
with open(Path(__file__).parent / "hyperspy_extension.yaml", "r") as stream:
    EXTENSIONS = yaml.load(stream, Loader=loader)

EXTENSIONS["GUI"]["widgets"] = {}

# External extensions are not integrated into the API and not
# import unless needed
ALL_EXTENSIONS = copy.deepcopy(EXTENSIONS)

_extensions = importlib.metadata.entry_points(group="hyperspy.extensions")

for _extension in _extensions:
    _logger.info("Enabling extension %s" % _extension.name)
    _path = (
        Path(importlib.util.find_spec(_extension.name).origin).parent
        / "hyperspy_extension.yaml"
    )

    with open(str(_path)) as stream:
        _extension_dict = yaml.load(stream, Loader=loader)
        if "signals" in _extension_dict:
            ALL_EXTENSIONS["signals"].update(_extension_dict["signals"])
        if "components1D" in _extension_dict:
            ALL_EXTENSIONS["components1D"].update(_extension_dict["components1D"])
        if "components2D" in _extension_dict:
            ALL_EXTENSIONS["components2D"].update(_extension_dict["components2D"])
        if "GUI" in _extension_dict:
            if "toolkeys" in _extension_dict["GUI"]:
                ALL_EXTENSIONS["GUI"]["toolkeys"].extend(
                    _extension_dict["GUI"]["toolkeys"]
                )
            if "widgets" in _extension_dict["GUI"]:
                for toolkit, specs in _extension_dict["GUI"]["widgets"].items():
                    if toolkit not in ALL_EXTENSIONS["GUI"]["widgets"]:
                        ALL_EXTENSIONS["GUI"]["widgets"][toolkit] = {}
                    ALL_EXTENSIONS["GUI"]["widgets"][toolkit].update(specs)
