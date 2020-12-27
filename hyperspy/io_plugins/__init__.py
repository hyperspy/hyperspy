# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

from hyperspy.io_plugins import (
    blockfile,
    bruker,
    dens,
    digital_micrograph,
    edax,
    emd,
    empad,
    fei,
    hspy,
    image,
    mrc,
    msa,
    nexus,
    phenom,
    protochips,
    ripple,
    semper_unf,
    sur,
    tiff,
)

io_plugins = [
    blockfile,
    bruker,
    dens,
    digital_micrograph,
    edax,
    emd,
    empad,
    fei,
    hspy,
    image,
    mrc,
    msa,
    nexus,
    phenom,
    protochips,
    ripple,
    semper_unf,
    sur,
    tiff,
]


_logger = logging.getLogger(__name__)


try:
    from hyperspy.io_plugins import netcdf

    io_plugins.append(netcdf)
except ImportError:
    # NetCDF is obsolete and is only provided for users who have
    # old EELSLab files. Therefore, we silently ignore if missing.
    pass

try:
    from hyperspy.io_plugins import usid_hdf5

    io_plugins.append(usid_hdf5)
except ImportError:
    _logger.info(
        "The USID IO plugin is not available because "
        "the pyUSID or sidpy packages are not installed."
    )

try:
    from hyperspy.io_plugins import mrcz

    io_plugins.append(mrcz)
except ImportError:
    _logger.info(
        "The mrcz IO plugin is not available because "
        "the mrcz package is not installed."
    )


default_write_ext = set()
for plugin in io_plugins:
    if plugin.writes:
        default_write_ext.add(plugin.file_extensions[plugin.default_extension])
