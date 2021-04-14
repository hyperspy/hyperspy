# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import logging
import copy
import pkgutil
import yaml

from pathlib import Path
import importlib_metadata as metadata


_logger = logging.getLogger(__name__)

_ext_f = Path(__file__).resolve().parent.joinpath("hyperspy_extension.yaml")
with open(_ext_f, 'r') as stream:
    EXTENSIONS = yaml.safe_load(stream)
EXTENSIONS["GUI"]["widgets"] = {}

# External extensions are not integrated into the API and not
# import unless needed
ALL_EXTENSIONS = copy.deepcopy(EXTENSIONS)

_external_extensions = [
    entry_point.module
    for entry_point in metadata.entry_points(group="hyperspy.extensions")]

for _external_extension_mod in _external_extensions:
    _logger.info("Enabling extension %s" % _external_extension_mod)
    _path = Path(
        pkgutil.get_loader(_external_extension_mod).get_filename()
    ).resolve().parent.joinpath("hyperspy_extension.yaml")

    if _path.is_file():
        with open(_path, 'r') as stream:
            _external_extension = yaml.safe_load(stream)
            if "signals" in _external_extension:
                ALL_EXTENSIONS["signals"].update(_external_extension["signals"])
            if "components1D" in _external_extension:
                ALL_EXTENSIONS["components1D"].update(
                    _external_extension["components1D"])
            if "components2D" in _external_extension:
                ALL_EXTENSIONS["components2D"].update(
                    _external_extension["components2D"])
            if "GUI" in _external_extension:
                if "toolkeys" in _external_extension["GUI"]:
                    ALL_EXTENSIONS["GUI"]["toolkeys"].extend(
                        _external_extension["GUI"]["toolkeys"])
                if "widgets" in _external_extension["GUI"]:
                    for toolkit, specs in _external_extension["GUI"]["widgets"].items():
                        if toolkit not in ALL_EXTENSIONS["GUI"]["widgets"]:
                            ALL_EXTENSIONS["GUI"]["widgets"][toolkit] = {}
                        ALL_EXTENSIONS["GUI"]["widgets"][toolkit].update(specs)

    else:
        _logger.error(
            "Failed to load hyperspy extension from {0}. Please report this issue to the {0} developers".format(_external_extension_mod))
