# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

from hyperspy.io_plugins import (msa, digital_micrograph, fei, mrc, ripple,
                                 tiff, semper_unf, blockfile, dens, emd,
                                 protochips, edax, bruker)


io_plugins = [msa, digital_micrograph, fei, mrc, ripple, tiff, semper_unf,
              blockfile, dens, emd, protochips, edax, bruker]


_logger = logging.getLogger(__name__)

try:
    from hyperspy.io_plugins import netcdf
    io_plugins.append(netcdf)
except ImportError:
    pass
    # NetCDF is obsolate and is only provided for users who have
    # old EELSLab files. Therefore, we silenly ignore if missing.

try:
    from hyperspy.io_plugins import hspy
    io_plugins.append(hspy)
    from hyperspy.io_plugins import emd
    io_plugins.append(emd)
except ImportError:
    _logger.warning('The HDF5 IO features are not available. '
                    'It is highly reccomended to install h5py')

try:
    from hyperspy.io_plugins import image
    io_plugins.append(image)
except ImportError:
    _logger.info('The Signal2D (PIL) IO features are not available')

try:
    from hyperspy.io_plugins import mrcz
    io_plugins.append(mrcz)
except ImportError:
    _logger.info('The mrcz IO plugin is not available because '
                 'the mrcz Python package is not installed.')
default_write_ext = set()
for plugin in io_plugins:
    if plugin.writes:
        default_write_ext.add(
            plugin.file_extensions[plugin.default_extension])
