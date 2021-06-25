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

import os
import re
import logging

import numpy as np
from tifffile import FileHandle

# Plugin characteristics
# ----------------------
format_name = 'tvips'
description = ('Read support for TVIPS CMOS camera stream/movie files. Can be'
               'used for in-situ movies or 4D-STEM datasets.')
full_support = False
# Recognised file extension
file_extensions = ['blo', 'BLO']
default_extension = 0
# Writing capabilities:
writes = False


_logger = logging.getLogger(__name__)

TVIPS_RECORDER_GENERAL_HEADER = [
    ("size", "u4"),  #likely the size of generalheader in bytes
    ("version", "u4"),  # 1 or 2
    ("dimx", "u4"),  # image size width
    ("dimy", "u4"),  # image size height
    ("bitsperpixel", "u4"),  # 8 or 16
    ("offsetx", "u4"),  # generally 0
    ("offsety", "u4"),
    ("binx", "u4"),  # camera binning
    ("biny", "u4"),
    ("pixelsize", "u4"),  # physical pixel size in nm
    ("ht", "u4"),  # high tension, voltage
    ("magtotal", "u4"),  # magnification/camera length
    ("frameheaderbytes", "u4"),  # number of bytes per frame header
    ("dummy", "S204"),  # placeholder contains TVIPS TVIPS TVIPS...
]


TVIPS_RECORDER_FRAME_HEADER = [
    ("num", "u4"),  # tends to cycle
    ("timestamp", "u4"),  # seconds since 1.1.1970
    ("ms", "u4"),  # additional milliseconds to the timestamp
    ("LUTidx", "u4"),  # related to color, useless
    ("fcurrent", "f4"),  # usually 0 for all frames
    ("mag", "u4"),  # same for all frames, can be different from magtotal
    ("mode", "u4"),  # 1 -> image, 2 -> diffraction
    ("stagex", "f4"),
    ("stagey", "f4"),
    ("stagez", "f4"),
    ("stagea", "f4"),
    ("stageb", "f4"),
    ("rotidx", "u4"),  # encodes information about the scan
    ("temperature", "f4"),  # cycles between 0.0 and 9.0 with step 1.0
    ("objective", "f4"),  # kind of randomly between 0.0 and 1.0
    # TODO: for header version 2, some more data might be present - reverse engineer
]


def _is_valid_first_tvips_file(filename):
    """Check if the provided first tvips file path is valid"""
    filpattern = re.compile(r".+\_([0-9]{3})\.(.*)")
    match = re.match(filpattern, filename)
    if match is not None:
        num, ext = match.groups()
        if ext != "tvips":
            raise ValueError(
                f"Invalid tvips file: extension {ext}, must " f"be tvips"
            )
        if int(num) != 0:
            raise ValueError(
                "Can only read video sequences starting with " "part 000"
            )
        return True
    else:
        raise ValueError("Could not recognize as a valid tvips file")


def file_reader(filename,
                tvips_scan_shape=None,
                tvips_start_frame="auto",
                tvips_stop_frame=None,
                tvips_hysteresis=0,
                tvips_winding_scan=False,
                **kwds):
    # check whether we start at the first tvips file
    _is_valid_first_tvips_file(filename)
    # get all filenames
    filenames = []
    basename = filename[:-9]
    while True:

    with open(filename, "rb") as f:
        f.seek(0)
        # read the main header in file 0
        header = np.fromfile(f, dtype=TVIPS_RECORDER_GENERAL_HEADER, count=1)
        dtype = np.uint8 if header["bitsperpixel"][0] == 8 else np.uint16
        dimx = header["dimx"][0]
        dimy = header["dimy"][0])
        # the size 
        if header.version == 1:
            increment = 12
        elif header.version == 2:
            increment = header.frameheaderbytes
        else:
            raise NotImplementedError(
                f"This version {header.version} is not yet supported"
                " in HyperSpy. Please report this as an issue at "
                "https://github.com/hyperspy/hyperspy/issues."
            )
        frame_header_dt = np.dtype(TVIPS_RECORDER_FRAME_HEADER)
        # the record must consume less bytes than reported in the main header
        if increment < frame_header_dt.itemsize:
            raise ValueError("The record consumes more bytes than stated in the main header")
        original_metadata = {'tvips_header': sarray2dict(general_header)}

        
        
        # Create the axis objects for each axis
        dim = data.ndim
        axes = [
            {
                'size': data.shape[i],
                'index_in_array': i,
                'name': names[i],
                'scale': scales[i],
                'offset': 0.0,
                'units': units[i], }
            for i in range(dim)]

        dictionary = {'data': data,
                    'axes': axes,
                    'metadata': metadata,
                    'original_metadata': original_metadata,
                    'mapping': mapping, }

    return [dictionary,]
