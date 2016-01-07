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


from hyperspy import messages
from hyperspy.io_plugins import (msa, digital_micrograph, fei, mrc,
                                 ripple, tiff)
io_plugins = [msa, digital_micrograph, fei, mrc, ripple, tiff]
try:
    from hyperspy.io_plugins import netcdf
    io_plugins.append(netcdf)
except ImportError:
    pass
    # NetCDF is obsolate and is only provided for users who have
    # old EELSLab files. Therefore, we print no message if it is not
    # available
    #~ messages.information('The NetCDF IO features are not available')

try:
    from hyperspy.io_plugins import hdf5
    io_plugins.append(hdf5)
except ImportError:
    messages.warning('The HDF5 IO features are not available. '
                     'It is highly reccomended to install h5py')

try:
    from hyperspy.io_plugins import image
    io_plugins.append(image)
except ImportError:
    messages.information('The Image (PIL) IO features are not available')

default_write_ext = set()
for plugin in io_plugins:
    if plugin.writes:
        default_write_ext.add(
            plugin.file_extensions[plugin.default_extension])
