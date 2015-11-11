# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

# The details of the format were taken from
# http://www.biochem.mpg.de/doc_tom/TOM_Release_2008/IOfun/tom_mrcread.html
# and http://ami.scripps.edu/software/mrctools/mrc_specification.php

import os

import numpy as np
from traits.api import Undefined

from hyperspy.misc.array_tools import sarray2dict


# Plugin characteristics
# ----------------------
format_name = 'EDAX TEAM'
description = 'Reader for EDS maps and spectra saved by the EDAX TEAM \n' \
              'software: An SPD file contains map data. The spectral \n' \
              'information is held in an SPC file with the same name, \n' \
              'while the spatial calibration is held in a related IPR file.'
full_support = False
# Recognised file extension
file_extensions = ['spd', 'SPD', 'spc', 'SPC']
default_extension = 0

# Writing capabilities
writes = False

spd_extensions = ('spd', 'SPD')
spc_extensions = ('spc', 'SPC')


def file_reader(filename, *args, **kwds):
    """

    Parameters
    ----------
    filename
    args
    kwds

    Returns
    -------

    """
    ext = os.path.splitext(filename)[1][1:]
    if ext in spd_extensions:
        return [spd_reader(filename, *args, **kwds), ]
    elif ext in spc_extensions:
        return spc_reader(filename, *args, **kwds)
