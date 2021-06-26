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
from datetime import datetime

import numpy as np
import dask.array as da
import dask

from hyperspy.misc.array_tools import sarray2dict

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
    # TODO: sometimes scan positions may be present in header, may require more reverse engineering
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


def _find_auto_scan_start_stop(rotidxs):
    """Find the start and stop index in a rotator index array"""
    diff = rotidxs[1:] - rotidxs[:-1]
    indx = np.where(diff > 0)[0]
    if indx.size == 0:
        return None, None
    else:
        return indx[0] + 1, indx[-1] + 1


def _guess_scan_index_grid(rotidx, start, stop):
    total_scan_frames = rotidx[stop]
    rotidx = rotidx[start : stop + 1]
    indxs = np.zeros(total_scan_frames, dtype=np.int64)
    prevw = 1
    for j in range(indxs):
        # find where the argument is j
        w = np.argwhere(rotidx == j + 1)
        if w.size > 0:
            w = w[0, 0]
            prevw = w
        else:
            # move up if the rot index stays the same, else copy
            if prevw + 1 < len(rotidx):
                if rotidx[prevw + 1] == rotidx[prevw]:
                    prevw = prevw + 1
            w = prevw
        indxs[j] = w
    return indxs + start


def file_reader(filename,
                scan_shape=None,
                scan_start_frame=0,
                winding_scan_axis=None,
                hysteresis=0,
                lazy=True,
                **kwds):
    """
    TVIPS stream file reader for in-situ and 4D STEM data

    Parameters
    ----------
    scan_shape : str or 2-tuple of int
        By default the data is loaded as an image stack (1 navigation axis).
        If it concerns a 4D-STEM dataset, the (scan_y, scan_x) dimension can
        be provided. "auto" can also be selected, in which case the rotidx 
        information in the frame headers will be used to try to reconstruct
        the scan.
    scan_start_frame : int
        First frame where the scan starts. If scan_shape = "auto" this is
        ignored.
    winding_scan_axis : str
        "x" or "y" if the scan was performed with winding scan along an axis
        as opposed to flyback scan.
    hysteresis:
        Only applicable if winding_scan_axis is not None. This parameter allows
        every second column or row to be shifted to correct for hysteresis that
        occurs during a winding scan.
    """
    # check whether we start at the first tvips file
    _is_valid_first_tvips_file(filename)

    # get all other filenames in case they exist
    other_files = []
    basename = filename[:-9]  # last bit: 000.tvips
    file_index = 1
    _, ext = os.path.splitext(filename)
    while True:
        fn = basename + "{:03d}{}".format(file_index, ext)
        if not os.path.exists(fn):
            break
        other_files.append(fn)
        file_index += 1

    # parse the header from the first file
    with open(filename, "rb") as f:
        f.seek(0)
        # read the main header in file 0
        header = np.fromfile(f, dtype=TVIPS_RECORDER_GENERAL_HEADER, count=1)
        dtype = np.uint8 if header["bitsperpixel"][0] == 8 else np.uint16
        dimx = header["dimx"][0]
        dimy = header["dimy"][0]
        # the size of the frame header varies with version
        if header["version"][0] == 1:
            increment = 12
        elif header["version"][0] == 2:
            increment = header["frameheaderbytes"][0]
        else:
            raise NotImplementedError(
                f"This version {header.version} is not yet supported"
                " in HyperSpy. Please report this as an issue at "
                "https://github.com/hyperspy/hyperspy/issues."
            )
        frame_header_dt = np.dtype(TVIPS_RECORDER_FRAME_HEADER)
        # the record must consume less bytes than reported in the main header
        if increment < frame_header_dt.itemsize:
            raise ValueError("The frame header record consumes more bytes than stated in the main header")
        # save metadata
        original_metadata = {'tvips_header': sarray2dict(header)}
        # create custom dtype for memmap padding the frame_header as required
        extra_bytes = increment - frame_header_dt.itemsize
        record_dtype = TVIPS_RECORDER_FRAME_HEADER.copy()
        record_dtype.append(("extra", bytes, extra_bytes))
        record_dtype.append(("data", dtype, dimx*dimy))

    # memmap the data
    records_000 = np.memmap(filename, mode="r", dtype=record_dtype, offset=header["size"][0])
    # the array data
    all_array_data = [records_000["data"].reshape(-1, dimx, dimy)]
    # in case we also want the frame header metadata later
    metadata_keys = np.array(TVIPS_RECORDER_FRAME_HEADER)[:,0]
    metadata_000 = records_000[metadata_keys]
    all_metadata = [metadata_000]
    # also load data from other files
    for i in other_files:
        # no offset on the other files
        records = np.memmap(i, mode="r", dtype=record_dtype)
        all_metadata.append(records[metadata_keys])
        all_array_data.append(records["data"].reshape(-1, dimx, dimy))
    if lazy:
        data_stack = da.concatenate(all_array_data, axis=0)
    else:
        data_stack = np.concatenate(all_array_data, axis=0)
    
    # extracting some units/scales/offsets of the DP's or images
    mode = all_metadata[0]["mode"][0]
    DPU = "1/m" if mode == 2 else "m"
    SDP = header["pixelsize"][0]
    offsetx = header["offsetx"][0]
    offsety = header["offsety"][0]
    # modify the data if there is scan information
    # we construct a 2D array of indices to slice the data_stack
    if scan_shape is not None:
        # try to deduce start and stop of the scan based on rotator index
        if scan_shape == "auto":
            record_idxs = np.concatenate([i["rotidx"] for i in all_metadata])
            scan_start_frame, scan_stop_frame = _find_auto_scan_start_stop(record_idxs)
            if scan_start_frame is None or scan_stop_frame is None:
                raise ValueError("Scan start and stop information could not be automatically "
                        "determined. Please supply a scan_shape and scan_start_frame.")
            total_scan_frames = record_idxs[scan_stop_frame]  # last rotator
            scan_dim = int(np.sqrt(total_scan_frames))
            if not np.allclose(scan_dim, np.sqrt(total_scan_frames)):
                raise ValueError("Scan was not square, please supply a scan_shape and start_frame.")
            scan_shape = (scan_dim, scan_dim)
            # there may be discontinuities which must be filled up
            indices = _guess_scan_index_grid(record_idxs, scan_start_frame, scan_stop_frame)
        # scan shape and start are provided
        else:
            total_scan_frames = scan_shape[0] * scan_shape[1]
            indices = np.arange(scan_start_frame, scan_start_frame+total_scan_frames).reshape(scan_shape[0], scan_shape[1])

        # with winding scan, every second column or row must be inverted
        # due to hysteresis there is also a predictable offset
        if winding_scan_axis is not None:
            if winding_scan_axis in ["x", 0]:
                indices[::2] = indices[::2][:, ::-1]
                indices[::2] = np.roll(indices[::2], hysteresis, axis=1)
            elif winding_scan_axis in ["y", 1]:
                indices[:, ::2] = indices[:, ::2][::-1, :]
                indices[:, ::2] = np.roll(indices[:, ::2], hysteresis, axis=0)
            else:
                raise ValueError("Invalid winding scan axis")
        
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            data_stack = data_stack[indices.ravel()]
        data_stack = data_stack.reshape(*indices.shape, dimy, dimx)
        units = ['nm', 'nm', DPU, DPU]
        names = ['y', 'x', 'dy', 'dx']
        # no scale information stored in the scan!
        scales = [1, 1, SDP, SDP]
        offsets = [0, 0, offsety, offsetx]
        # Create the axis objects for each axis
        dim = data_stack.ndim
        axes = [
            {
                'size': data_stack.shape[i],
                'index_in_array': i,
                'name': names[i],
                'scale': scales[i],
                'offset': offsets[i],
                'units': units[i], }
            for i in range(dim)]
        if lazy:
            data_stack = data_stack.rechunk({0: "auto", 1: "auto", 2: None, 3: None})
    else:
        # we load as a regular image stack
        units = ['s', DPU, DPU]
        names = ['time', 'dy', 'dx']
        times = all_metadata[0]["timestamp"] + all_metadata[0]["ms"]/1000
        timescale = times[1] - times[0]
        scales = [timescale, SDP, SDP]
        offsets = [times[0], offsety, offsetx]
        # Create the axis objects for each axis
        dim = data_stack.ndim
        axes = [
            {
                'size': data_stack.shape[i],
                'index_in_array': i,
                'name': names[i],
                'scale': scales[i],
                'offset': offsets[i],
                'units': units[i], }
            for i in range(dim)]
        if lazy:
            data_stack = data_stack.rechunk({0: "auto", 1: None, 2: None})
    dtobj = datetime.fromtimestamp(all_metadata[0]["timestamp"][0])
    date = dtobj.date().isoformat()
    time = dtobj.time().isoformat()
    current = all_metadata[0]["fcurrent"][0]
    stagex = all_metadata[0]["stagex"][0]
    stagey = all_metadata[0]["stagey"][0]
    stagez = all_metadata[0]["stagez"][0]
    stagealpha = all_metadata[0]["stagea"][0]
    stagebeta = all_metadata[0]["stageb"][0]
    mag = all_metadata[0]["mag"][0]
    focus = all_metadata[0]["objective"][0]
    sigtype = "diffraction" if mode == 2 else "imaging"
    metadata = {'General': {'original_filename': os.path.split(filename)[1],
                            'date': date,
                            'time': time,
                "Signal": {'signal_type': sigtype},
                "Acquisition_instrument": {
                    "TEM": {
                        "magnification": header["magtotal"][0],
                        "beam_energy": header["ht"][0],
                        "beam_current": current,
                        "defocus": focus,
                        "Stage": {
                            "tilt_alpha": stagealpha,
                            "tilt_beta": stagebeta,
                            "x": stagex,
                            "y": stagey,
                            "z": stagez,
                            }
                    }
                }}}

    dictionary = {'data': data_stack,
                'axes': axes,
                'metadata': metadata,
                'original_metadata': original_metadata,
                'mapping': {}, }

    return [dictionary,]
