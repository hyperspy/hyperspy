# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

import os
from collections.abc import Iterable
from datetime import datetime, timedelta
import logging

import numpy as np
import numba


_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = "JEOL"
description = "Read JEOL files output by Analysis Station software"
full_support = False  # Whether all the Hyperspy features are supported
# Recognised file extension
file_extensions = ("ASW", "asw", "img", "map", "pts", "eds")
default_extension = 0  # Index of the extension that will be used by default
# Reading capabilities
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing capabilities
writes = False
non_uniform_axis = False
# ----------------------


jTYPE = {
    1: "B",
    2: "H",
    3: "i",
    4: "f",
    5: "d",
    6: "B",
    7: "H",
    8: "i",
    9: "f",
    10: "d",
    11: "?",
    12: "c",
    13: "c",
    14: "H",
    20: "c",
    65553: "?",
    65552: "?",
}


def file_reader(filename, **kwds):
    """
    File reader for JEOL format
    """
    dictionary = []
    file_ext = os.path.splitext(filename)[-1][1:].lower()
    if file_ext == "asw":
        fd = open(filename, "br")
        file_magic = np.fromfile(fd, "<I", 1)[0]
        if file_magic == 0:
            fd.seek(12)
            filetree = _parsejeol(fd)
            fd.close()
            filepath, filen = os.path.split(os.path.abspath(filename))
            if "SampleInfo" in filetree.keys():
                for i in filetree["SampleInfo"].keys():
                    if "ViewInfo" in filetree["SampleInfo"][i].keys():
                        for j in filetree["SampleInfo"][i]["ViewInfo"].keys():
                            node = filetree["SampleInfo"][i]["ViewInfo"][j]
                            if "ViewData" in node.keys():
                                scale = node["PositionMM"] * 1000
                                for k in node["ViewData"].keys():
                                    # path tuple contains:
                                    # root, sample_folder, view_folder, data_file
                                    path = node["ViewData"][k]["Filename"].split("\\")
                                    subfile = os.path.join(*path)
                                    sub_ext = os.path.splitext(subfile)[-1][1:]
                                    file_path = os.path.join(filepath, subfile)
                                    if sub_ext in extension_to_reader_mapping.keys():
                                        reader_function = extension_to_reader_mapping[sub_ext]
                                        d = reader_function(file_path, scale, **kwds)
                                        if isinstance(d, list):
                                            dictionary.extend(d)
                                        else:
                                            dictionary.append(d)
        else:
            _logger.warning("Not a valid JEOL asw format")
            fd.close()
    else:
        d = extension_to_reader_mapping[file_ext](filename, **kwds)
        if isinstance(d, list):
            dictionary.extend(d)
        else:
            dictionary.append(d)
    return dictionary


def _read_img(filename, scale=None, **kwargs):
    fd = open(filename, "br")
    file_magic = np.fromfile(fd, "<I", 1)[0]
    if file_magic == 52:
        # fileformat
        _ = _decode(fd.read(32).rstrip(b"\x00"))
        head_pos, head_len, data_pos = np.fromfile(fd, "<I", 3)
        fd.seek(data_pos + 12)
        header_long = _parsejeol(fd)
        width, height = header_long["Image"]["Size"]
        header_long["Image"]["Bits"].resize((height, width))
        data = header_long["Image"]["Bits"]
        if scale is not None:
            xscale = -scale[2] / width
            yscale = scale[3] / height
            units = "µm"
        else:
            scale = header_long["Instrument"]["ScanSize"] / header_long["Instrument"]["Mag"] * 1.0E3
            xscale = scale / width
            yscale = scale / height
            units = "µm"

        axes = [
            {
                "name": "y",
                "size": height,
                "offset": 0,
                "scale": yscale,
                "units": units,
            },
            {
                "name": "x",
                "size": width,
                "offset": 0,
                "scale": xscale,
                "units": units,
            },
        ]

        datefile = datetime(1899, 12, 30) + timedelta(
            days=header_long["Image"]["Created"]
        )
        hv = header_long["Instrument"]["AccV"]
        if hv <= 30.0:
            mode = "SEM"
        else:
            mode = "TEM"

        metadata = {
            "Acquisition_instrument": {
                mode: {
                    "beam_energy": hv,
                    "magnification": header_long["Instrument"]["Mag"],
                },
            },
            "General": {
                "original_filename": os.path.basename(filename),
                "date": datefile.date().isoformat(),
                "time": datefile.time().isoformat(),
                "title": header_long["Image"]["Title"],
            },
            "Signal": {
                "record_by": "image",
            },
        }

        dictionary = {
            "data": data,
            "axes": axes,
            "metadata": metadata,
            "original_metadata": header_long,
        }
    else:
        _logger.warning("Not a valid JEOL img format")

    fd.close()

    return dictionary


def _read_pts(filename, scale=None, rebin_energy=1, sum_frames=True,
             SI_dtype=np.uint8, cutoff_at_kV=None, downsample=1,
             only_valid_data=True, read_em_image=False,
             frame_list=None, frame_start_index=None, frame_shifts=None, 
             lazy=False,
             **kwargs):
    """
    Parameters
    ----------
    rawdata : numpy.ndarray of uint16
    	spectrum image part of pts file
    scale : list of float
    	-scale[2], scale[3] are the positional scale from asw data, 
    	default is None, calc from pts internal data
    rebin_energy : int
    	Binning parameter along energy axis. Must be 2^n.
    sum_frames : bool
        If False, returns each frame.
    SI_dtype : dtype
        data type for spectrum image. default is uint8
    cutoff_at_kV : float
        The maximum energy. Useful to reduce memory size of spectrum image.
        Default is None (no cutoff)
    downsample : int or (int, int)
    	Downsample along spatial axes to reduce memory size of spectrum image.
        Value must be 2^n. Default is 1 (no downsampling).
    only_valid_data : bool, default True
    	Read incomplete frame if only_valid_data == False
        (usually interrupting mesurement makes incomplete frame)
    read_em_image : bool, default False
        Read SEM/STEM image from pts file if read_em_image == True
    frame_list : list of int, default None
    	List of frame numbers to be read (None for all data)
    frame_shifts : list of [int, int] or list of [int, int, int], default None
    	Each frame will be loaded with offset of dy, dx, (and optional energy
        axis). Units are pixels/channels.
        This is useful for express drift correction. Not suitable for accurate
        analysis.
        Like the result of estimate_shift2D(), the first parameter is for y-axis
    frame_start_index: list
        The list of offset pointers of each frame in the raw data.
        The pointer for frame0 is 0.
    lazy : bool, default False
    	Read spectrum image into sparse array if lazy == True
    	SEM/STEM image is always read into dense array (numpy.ndarray)

    Returns
    -------
    dictionary : dict or list of dict
    	The dictionary used to create the signals, list of dictionaries of
        spectrum image and SEM/STEM image if read_em_image == True
    """
    fd = open(filename, "br")
    file_magic = np.fromfile(fd, "<I", 1)[0]

    def check_multiple(factor, number, string):
        if factor > 1 and number % factor != 0:
            fd.close()
            raise ValueError(f'`{string}` must be a multiple of {number}.')

    check_multiple(rebin_energy, 4096, 'rebin_energy')
    rebin_energy = int(rebin_energy)

    if file_magic == 304:
        # fileformat
        _ = _decode(fd.read(8).rstrip(b"\x00"))
        a, b, head_pos, head_len, data_pos, data_len = np.fromfile(fd, "<I", 6)
        # groupname
        _ = _decode(fd.read(128).rstrip(b"\x00"))
        # memo
        _ = _decode(fd.read(132).rstrip(b"\x00"))
        datefile = datetime(1899, 12, 30) + timedelta(days=np.fromfile(fd, "d", 1)[0])
        fd.seek(head_pos + 12)
        header = _parsejeol(fd)
        meas_data_header = header["PTTD Data"]["AnalyzableMap MeasData"]
        width, height = meas_data_header["Meas Cond"]["Pixels"].split("x")
        width = int(width)
        height = int(height)

        if isinstance(downsample, Iterable):
            if len(downsample) > 2:
                raise ValueError("`downsample` can't be an iterable of length "
                                 "different from 2.")
            downsample_width = downsample[0]
            downsample_height = downsample[1]
            check_multiple(downsample_width, width, 'downsample[0]')
            check_multiple(downsample_height, height, 'downsample[1]')
        else:
            downsample_width = downsample_height = int(downsample)
            check_multiple(downsample_width, width, 'downsample')
            check_multiple(downsample_height, height, 'downsample')

        check_multiple(downsample_width, width, 'downsample[0]')
        check_multiple(downsample_height, height, 'downsample[1]')

        # Normalisation factor for the x and y position in the stream; depends
        # on the downsampling and the size of the navigation space
        width_norm = int(4096 / width * downsample_width)
        height_norm = int(4096 / height * downsample_height)

        width = int(width / downsample_width)
        height = int(height / downsample_height)

        channel_number = int(4096 / rebin_energy)

        fd.seek(data_pos)
        # read spectrum image
        rawdata = np.fromfile(fd, dtype="u2")
        fd.close()

        if scale is not None:
            xscale = -scale[2] / width
            yscale = scale[3] / height
            units = "µm"
        else:
            scale = header["PTTD Param"]["Params"]["PARAMPAGE0_SEM"]["ScanSize"] / meas_data_header["MeasCond"]["Mag"] * 1.0E3
            xscale = scale / width
            yscale = scale / height
            units = "µm"

        ch_mod = meas_data_header["Meas Cond"]["Tpl"]
        ch_res = meas_data_header["Doc"]["CoefA"]
        ch_ini = meas_data_header["Doc"]["CoefB"]
        ch_pos = header["PTTD Param"]["Params"]["PARAMPAGE1_EDXRF"]["Tpl"][ch_mod][
            "DigZ"
        ]

        energy_offset = ch_ini - ch_res * ch_pos
        energy_scale = ch_res * rebin_energy

        if cutoff_at_kV is not None:
            channel_number = int(
                np.round((cutoff_at_kV - energy_offset) / energy_scale)
                )
        # pixel time in milliseconds
        pixel_time = meas_data_header["Doc"]["DwellTime(msec)"]

        # Sweep value is not reliable, so +1 frame is needed if sum_frames = False
        # priority of the length of frame_start_index is higher than "sweep" in header
        sweep = meas_data_header["Doc"]["Sweep"]
        if frame_start_index:            
            sweep = len(frame_start_index)

        auto_frame_list = False
        if frame_list:
            frame_list = np.asarray(frame_list)
        else:
            auto_frame_list = True
            frame_list = np.arange(sweep + 1)
    
        # Remove frame numbers outside the data range.
        # The frame with number == sweep is accepted in this stage
        #    for incomplete frame
        # If "frame_shifts" option is used, frame number must be in range of frame_shifts
        if frame_shifts is not None:
            nsf = len(frame_shifts)
            wrong_frames_list = frame_list[
                np.where((frame_list<0) | (frame_list > sweep)
                         | (frame_list > nsf)
                         | ((frame_list == nsf) & (not auto_frame_list)))]
            frame_list = frame_list[
                np.where((0 <= frame_list) & (frame_list <= sweep)
                         & (frame_list < nsf))]
        else:
            wrong_frames_list = frame_list[
                np.where((frame_list<0) | (frame_list > sweep))]
            frame_list = frame_list[
                np.where((0 <= frame_list) & (frame_list <= sweep))]
    
        if len(wrong_frames_list) > 0:
            wrong_frames = wrong_frames_list.flatten().tolist()
            _logger.info(f"Invalid frame number is specified. The frame {wrong_frames} is not found in pts data.")

        # + 1 for incomplete frame
        max_frame = frame_list.max() + 1

        if frame_start_index is None:
            frame_start_index = np.full(max_frame, -1, dtype = np.int32)
        else:
            frame_start_index = np.asarray(frame_start_index)

        # fill with -1 as invaid index (not loaded)
        if (frame_start_index.size < max_frame):
            fi = np.full(max_frame, -1, dtype = np.int32)
            fi[0:frame_start_index.size] = frame_start_index
            frame_start_index = fi

        if frame_shifts is None:
            frame_shifts = np.zeros((max_frame,3), dtype = np.int16)
        if (len(frame_shifts) < max_frame):
            fs =np.zeros((max_frame,3), dtype = np.int16)
            if len(frame_shifts) > 0:
                fs[0:len(frame_shifts),0:len(frame_shifts[0])] = frame_shifts
            frame_shifts = fs
        if len(frame_shifts[0])==2: # fill z with 0
            fr = np.zeros((max_frame,3), dtype = np.int16)
            fr[:len(frame_shifts), 0:2] = np.asarray(frame_shifts)
            frame_shifts = fr

        data, em_data, has_em_image, sweep, frame_start_index, last_valid, origin, frame_shifts_1 = _readcube(
            rawdata, frame_start_index, frame_list,
            width, height, channel_number,
            width_norm, height_norm, rebin_energy,
            SI_dtype, sweep, frame_shifts,
            sum_frames, read_em_image, only_valid_data, lazy)
        header["jeol_pts_frame_origin"] = origin
        header["jeol_pts_frame_shifts"] = frame_shifts_1
        header["jeol_pts_frame_start_index"] = frame_start_index
        # axes_em for SEM/STEM image  intensity[(frame,) y, x]
        # axes for spectrum image  count[(frame,) y, x, energy]
        if sum_frames:
            axes_em = []
            width = data.shape[1]
            height = data.shape[0]
        else:
            axes_em = [{
                "index_in_array": 0,
                "name": "Frame",
                "size": sweep,
                "offset": 0,
                "scale": pixel_time*height*width/1E3,
                "units": 's',
            }]
            width = data.shape[2]
            height = data.shape[1]
        axes_em.extend( [
            {
                "name": "y",
                "size": height,
                "offset": origin[1],
                "scale": yscale,
                "units": units,
            },
            {
                "name": "x",
                "size": width,
                "offset": origin[0],
                "scale": xscale,
                "units": units,
            }
        ] )
        axes = axes_em.copy()
        axes.append(
            {
                "name": "Energy",
                "size": channel_number,
                "offset": energy_offset,
                "scale": energy_scale,
                "units": "keV",
            },
        )
        if (not last_valid) and only_valid_data:
            _logger.info("The last frame (sweep) is incomplete because the acquisition stopped during this frame. The partially acquired frame is ignored. Use 'sum_frames=False, only_valid_data=False' to read all frames individually, including the last partially completed frame.")

        hv = meas_data_header["MeasCond"]["AccKV"]
        if hv <= 30.0:
            mode = "SEM"
        else:
            mode = "TEM"

        detector_hearder = header["PTTD Param"]["Params"]["PARAMPAGE0_SEM"]
        metadata = {
            "Acquisition_instrument": {
                mode: {
                    "beam_energy": hv,
                    "magnification": meas_data_header["MeasCond"]["Mag"],
                    "Detector": {
                        "EDS": {
                            "azimuth_angle": detector_hearder["DirAng"],
                            "detector_type": detector_hearder["DetT"],
                            "elevation_angle": detector_hearder["ElevAng"],
                            "energy_resolution_MnKa": detector_hearder["MnKaRES"],
                            "real_time": meas_data_header["Doc"]["RealTime"],
                        },
                    },
                },
            },
            "General": {
                "original_filename": os.path.basename(filename),
                "date": datefile.date().isoformat(),
                "time": datefile.time().isoformat(),
                "title": "EDS extracted from " + os.path.basename(filename),
            },
            "Signal": {
                "record_by": "spectrum",
                "quantity": "X-rays (Counts)",
                "signal_type": "EDS_" + mode,
            },
        }
        metadata_em = {
            "Acquisition_instrument": {
                mode: {
                    "beam_energy": hv,
                    "magnification": meas_data_header["MeasCond"]["Mag"],
                },
            },
            "General": {
                "original_filename": os.path.basename(filename),
                "date": datefile.date().isoformat(),
                "time": datefile.time().isoformat(),
                "title": "S(T)EM Image extracted from " + os.path.basename(filename)
            },
            "Signal": {
                "record_by": "image",
            },
        }

        dictionary = {
            "data": data,
            "axes": axes,
            "metadata": metadata,
            "original_metadata": header,
        }
        if read_em_image and has_em_image:
            dictionary = [dictionary,
                          {
                              "data": em_data,
                              "axes": axes_em,
                              "metadata": metadata_em,
                              "original_metadata": header
                          }]
    else:
        _logger.warning("Not a valid JEOL pts format")
        fd.close()

    return dictionary


def _parsejeol(fd):
    final_dict = {}
    tmp_list = []
    tmp_dict = final_dict

    mark = 1
    while abs(mark) == 1:
        mark = np.fromfile(fd, "b", 1)[0]
        if mark == 1:
            str_len = np.fromfile(fd, "<i", 1)[0]
            kwrd = fd.read(str_len).rstrip(b"\x00")

            if (
                kwrd == b"\xce\xdf\xb0\xc4"
            ):  # correct variable name which might be 'Port'
                kwrd = "Port"
            elif (
                kwrd[-1] == 222
            ):  # remove undecodable byte at the end of first ScanSize variable
                kwrd = _decode(kwrd[:-1])
            else:
                kwrd = _decode(kwrd)
            val_type, val_len = np.fromfile(fd, "<i", 2)
            tmp_list.append(kwrd)
            if val_type == 0:
                tmp_dict[kwrd] = {}
            else:
                c_type = jTYPE[val_type]
                arr_len = val_len // np.dtype(c_type).itemsize
                if c_type == "c":
                    value = fd.read(val_len).rstrip(b"\x00")
                    value = _decode(value).split("\x00")
                    # value = os.path.normpath(value.replace('\\','/')).split('\x00')
                else:
                    value = np.fromfile(fd, c_type, arr_len)
                if len(value) == 1:
                    value = value[0]
                if kwrd[-5:-1] == "PAGE":
                    kwrd = kwrd + "_" + value
                    tmp_dict[kwrd] = {}
                    tmp_list[-1] = kwrd
                elif kwrd == "CountRate" or kwrd == "DeadTime":
                    tmp_dict[kwrd] = {}
                    tmp_dict[kwrd]["value"] = value
                elif kwrd == "Limits":
                    pass
                    # see https://github.com/hyperspy/hyperspy/pull/2488
                    # first 16 bytes are encode in float32 and looks like limit values ([20. , 1., 2000, 1.] or [1., 0., 1000., 0.001])
                    # next 4 bytes are ascii character and looks like number format (%.0f or %.3f)
                    # next 12 bytes are unclear
                    # next 4 bytes are ascii character and are units (kV or nA)
                    # last 12 byes are unclear
                elif val_type == 14:
                    tmp_dict[kwrd] = {}
                    tmp_dict[kwrd]["index"] = value
                else:
                    tmp_dict[kwrd] = value
            if kwrd == "Limits":
                pass
                # see https://github.com/hyperspy/hyperspy/pull/2488
                # first 16 bytes are encode in int32 and looks like limit values (10, 1, 100000000, 1)
                # next 4 bytes are ascii character and looks like number format (%d)
                # next 12 bytes are unclear
                # next 4 bytes are ascii character and are units (mag)
                # last 12 byes are again unclear
            else:
                tmp_dict = tmp_dict[kwrd]
        else:
            if len(tmp_list) != 0:
                del tmp_list[-1]
                tmp_dict = final_dict
                for k in tmp_list:
                    tmp_dict = tmp_dict[k]
            else:
                mark = 0

    return final_dict


def _readcube(rawdata, frame_start_index, frame_list,
             width, height, channel_number,
             width_norm, height_norm, rebin_energy,
             SI_dtype, sweep, frame_shifts,
             sum_frames, read_em_image, only_valid_data, lazy): # pragma: no cover
    """
    Read spectrum image (and SEM/STEM image) from pts file

    Parameters
    ----------
    rawdata : numpy.ndarray
        Spectrum image part of pts file.
    frame_start_index : np.ndarray of shape (sweep+1, ) or (0, ) 
        The indices of each frame start. If length is zero, the indices will be
        determined from rawdata.
    frame_list : list
        List of frames to be read.
    width, height : int
    	The navigation dimension.
    channel_number : int
        The number of channels.
    width_norm, height_norm : int
        Rebin factor of the navigation dimension.
    rebin_energy : int
        Rebin factor of the energy dimension.
    sweep : int
        Number of sweep
    frame_shifts : list
        The list of image positions [[x0,y0,z0], ...]. The x, y, z values can
        be negative. The data points outside data cube are ignored.

    Returns
    -------
    data : numpy.ndarray or dask.array
        The spectrum image with shape (frame, x, y, energy) if sum_frames is
        False, otherwise (x, y, energy).
        If lazy is True, the dask array is a COO sparse array.
    em_data : numpy.ndarray or dask.array
        The SEM/STEM image with shape (frame, x, y) if sum_frames is False,
        otherwise (x, y).
    has_em_image : bool
        True if the stream contains SEM/STEM images.
    sweep : int
	    The number of loaded frames.
    frame_start_index : list
        The indices of each frame start. 
    max_shift : numpy.ndarray
        The maximum shifts of the origin in the navigation dimension.
```
    frame_shifts : numpy.ndarray
        The shifts of the origin in the navigation dimension for each frame.
    """

    import dask.array as da

    # In case of sum_frames, spectrum image and SEM/STEM image are summing up to the same frame number.
    # To avoid overflow on integration of SEM/STEM image, data type of np.uint32 is selected
    # for 16 frames and over. (range of image intensity in each frame is 0-4095 (0-0xfff))
    EM_dtype = np.uint16
    frame_step = 1
    if sum_frames:
        frame_step = 0
        if sweep >= 16:
            EM_dtype = np.uint32
        n_frames = 1  
    else:
        n_frames = sweep + 1
        
    if lazy:
        hypermap = np.zeros((n_frames), dtype=EM_dtype)  # dummy variable, not used
        data_list = []
    else:
        hypermap = np.zeros((n_frames, height, width, channel_number),
                            dtype=SI_dtype)

    em_image = np.zeros((n_frames, width, height), dtype=EM_dtype)

    max_value =  np.iinfo(SI_dtype).max

    frame_shifts = np.asarray(frame_shifts)
    frame_list = np.asarray(frame_list)
    max_shift = frame_shifts[frame_list].max(axis=0)
    min_shift = frame_shifts[frame_list].min(axis=0)
    #    sxyz = np.array([min_shift[0]-max_shift[0], min_shift[1]-max_shift[1],0])
    min_shift[2]=0
    max_shift[2]=0
    sxyz = min_shift-max_shift
    frame_shifts -= max_shift
    width += sxyz[1]
    height += sxyz[0]

    if lazy:
        readframe = _readframe_lazy
    else:
        readframe = _readframe_dense
        
    frame_num = 0
    p_start = 0
    target_frame_num = 0
    eof = rawdata.size
    countup = 1
    has_em_image = False
    for frame_idx in frame_list:
        if frame_idx < 0:
            continue
        elif frame_start_index[frame_idx] >= 0:
            # if frame_idx is already indexed
            p_start = frame_start_index[frame_idx]
        elif frame_num < frame_idx and frame_start_index[frame_num] < 0:
            # record start point of frame and skip frame
            frame_start_index[frame_num] = p_start
            p_start += _readframe_dummy(rawdata[p_start:])
            frame_num += 1
            continue
        else:
            frame_start_index[frame_idx] = p_start  # = end of last frame

        if frame_idx < frame_shifts.size:
            fs = frame_shifts[frame_idx]
        else:
            fs = np.zeros(3, np.uint16)
            _logger.info(f"Size of frame_shift array is too small. The frame {frame_idx} is not moved.")
        length, frame_data, has_em, valid, max_valid = readframe(
            rawdata[p_start:], 1,
            hypermap[target_frame_num], em_image[target_frame_num],
            width, height, channel_number,
            width_norm, height_norm, rebin_energy,
            fs[1], fs[0], fs[2], max_value)
        has_em_image = has_em_image or has_em
        if length == 0: # no data
            break
        if valid or not only_valid_data:
            # accept last frame
            if lazy:
                data_list.append(frame_data)
            frame_num += 1
            target_frame_num += frame_step
        else:
            # incomplete data, not accepted
            if sum_frames:
                # subtract signal counts of last frame
                _ = readframe(rawdata[p_start:], -1,
                              hypermap[target_frame_num], em_image[target_frame_num],
                              width, height, channel_number,
                              width_norm, height_norm, rebin_energy,
                              fs[1], fs[0],fs[2],  max_value)
                _logger.info("The last frame (sweep) is incomplete because the acquisition stopped during this frame. The partially acquired frame is ignored. Use 'sum_frames=False, only_valid_data=False' to read all frames individually, including the last partially completed frame.")
            break
            # else:
            #    pass

        p_start += length
    if not lazy:
        if sum_frames:
            # the first frame has integrated intensity
            return hypermap[0,:height,:width], em_image[0,:height,:width], has_em_image, frame_num, frame_start_index, valid, max_shift, frame_shifts
        else:
            return hypermap[:target_frame_num,:height,:width], em_image[:target_frame_num,:height,:width], has_em_image, frame_num, frame_start_index, valid, max_shift, frame_shifts
        
    # for lazy loading
    from hyperspy.misc.io.fei_stream_readers import DenseSliceCOO
    length = np.sum([len(d) for d in data_list])
    # length = number of data points


    # v : [[frame_no, y, x, energy_channel, 1], ....]
    v = np.zeros(shape=(5, length), dtype=np.uint16)
    ptr = 0
    frame_count = 0
    for d in data_list:
        # d : data points in one frame
        d = np.asarray(d)
        # check if the pixels are in the valid data cube
        # (frame_shifts make partially integrated area at the rim)
        valid_cube = np.where((0<=d[:,0]) & (d[:,0]<height) & (0<=d[:,1]) & (d[:,1]<width) & (0<=d[:,2]) & (d[:,2]<channel_number))
        d = d[valid_cube]
        flen = len(d)
        pv = v[:,ptr:ptr+flen]
        pv[1:4, :] = np.array(d).transpose()
        pv[0,:] = frame_count
        pv[4,:] = 1
        ptr += flen
        frame_count += 1
    if sum_frames:
        data_shape = [height, width, channel_number]
        ar_s = DenseSliceCOO(v[1:4], v[4], shape=data_shape)
    else:
        data_shape = [frame_count, height, width, channel_number]
        ar_s = DenseSliceCOO(v[0:4], v[4], shape=data_shape)
    if sum_frames:
        em_image = em_image[0]
        
    return da.from_array(ar_s, asarray=False), em_image, has_em_image, sweep, frame_start_index, valid, max_shift, frame_shifts


@numba.njit(cache=True)
def _readframe_dense(rawdata, countup, hypermap, em_image, width, height, channel_number,
                     width_norm, height_norm, rebin_energy, dx, dy, dz, max_value): # pragma: nocover
    """
    Read one frame from pts file. Used in a inner loop of _readcube function.
    This function always read SEM/STEM image even if read_em_image == False
    hypermap and em_image array will be modified

    Parameters
    ----------
    rawdata : numpy.ndarray of uint16
    	slice of one frame raw data from whole raw data 
    countup : 1 for summing up the X-ray events, -1 to cancel selected frame
    hypermap : numpy.ndarray(width, height, channel_number)
    	numpy.ndarray to store decoded spectrum image.
    em_image : numpy.ndarray(width, height), dtype = np.uint16 or np.uint32
    	numpy.ndarray to store decoded SEM/TEM image.
    width : int
    height : int
    channel_number : int
    	Limit of channel to reduce data size
    width_norm : int
    	scanning step
    height_norm : int
    	scanning step
    rebin_energy : int
    	Binning parameter along energy axis. Must be 2^n.
    dx, dy, dz : int
    	information of frame shift for drift correction. 
    max_value : int
    	limit of the data type used in hypermap array

    Returns
    -------
    raw_length : int
    	frame length based on raw data array
    _ : int
	dummy value (used for lazy loading)
    has_em_image : bool
    	True if pts file have SEM/STEM image
    valid : bool
    	True if current frame is completely swept
    	False for incomplete frame such as interrupt of measurement
    max_valid : int
    	maximum number of scan lines to be accepted as valid 
        in case of incomplete frame
    """

    count = 0
    has_em_image = False
    valid = False
    MAX_VAL = 4096
    previous_y = -1
    x = 0
    y = 0
    for value in rawdata:
        value_type = value & 0xf000
        value &= 0xfff
        if value_type == 0x8000:
            x = value // width_norm + dx
            if x >= width:
                x = -1
            previous_x = value
        elif value_type == 0x9000:
            y = value // height_norm + dy
            if value < previous_y:
                break
            previous_y = value
            if y >= height:
                y = -1
        elif value_type == 0xa000 and x >= 0 and y >= 0 :
            em_image[y, x] += value * countup
            has_em_image = True
        elif value_type == 0xb000:
            z = value // rebin_energy + dz
            if z < channel_number and x >= 0 and y >= 0 and z >= 0:
                hypermap[y, x, z] += countup
                if hypermap[y, x, z] == max_value:
                    raise ValueError("The range of the dtype is too small, "
                                     "use `SI_dtype` to set a dtype with "
                                     "higher range.")
        count += 1


    if previous_y >= MAX_VAL - height_norm and (previous_x >= MAX_VAL - width_norm or not has_em_image):
        # if the STEM-ADF/SEM-BF image is not included in pts file,
        # maximum value of x is usually smaller than max_x,
        # (sometimes no x-ray events occur in a last few pixels.)
        # So, only the y value is checked
        #
        # > is need when downsampling is specified
        valid = True
    return count, 0, has_em_image, valid, previous_y // height_norm


@numba.njit(cache=True)
def _readframe_lazy(rawdata, _1, _2, em_image, width, height, channel_number,
                    width_norm, height_norm, rebin_energy, dx, dy, dz, _3):  # pragma: no cover
    """
    Read one frame from pts file. Used in a inner loop of _readcube function.
    This function always read SEM/STEM image even if read_em_image == False
    hypermap and em_image array will be modified

    Parameters
    ----------
    rawdata : numpy.ndarray of uint16
    	slice of one frame raw data from whole raw data 
    _1 : dummy parameter, not used
    _2 : dummy parameter, not used
    em_image : numpy.ndarray(width, height), dtype = np.uint16 or np.uint32
    	numpy.ndarray to store decoded SEM/TEM image.
    width : int
    height : int
    channel_number : int
    	Limit of channel to reduce data size
    width_norm : int
    	scanning step
    height_norm : int
    	scanning step
    rebin_energy : int
    	Binning parameter along energy axis. Must be 2^n.
    dx, dy, dz : int
    	information of frame shift for drift correction. 
    _3 : dummy parameter, not used


    Returns
    -------
    raw_length : int
    	frame length based on raw data array
    data : list of [int, int, int]  
	list of X-ray events as an array of [x, y, energy_ch]
    has_em_image : bool
    	True if pts file have SEM/STEM image
    valid : bool
    	True if current frame is completely swept
    	False for incomplete frame such as interrupt of measurement
    max_valid : int
    	maximum number of scan lines to be accepted as valid 
        in case of incomplete frame
    """
    data = []
    previous_x = 0
    previous_y = 0
    MAX_VAL = 4096
    count = 0
    valid = False
    has_em_image = False
    for value in rawdata:
        count += 1
        value_type = value & 0xf000
        value &= 0xfff
        if value_type == 0x8000:
            previous_x = value
            x = value // width_norm + dx
            if x >= width:
                x = -1
        elif value_type == 0x9000:
            if (value < previous_y):
                break
            previous_y = value
            y = value // height_norm + dy
            if y >= width:
                y = -1
        elif value_type == 0xa000 and x >= 0 and y >= 0:
            em_image[y, x] += value
            has_em_image = True
        elif value_type == 0xb000:    # spectrum image
            z = value // rebin_energy + dz
            if 0 <= z and z < channel_number:
                data.append([y, x, z])
    if previous_y == MAX_VAL - height_norm and (previous_x == MAX_VAL - width_norm or not has_em_image):
        # if the STEM-ADF/SEM-BF image is not included in pts file,
        # maximum value of x is usually smaller than max_x,
        # (sometimes no x-ray events occur in a last few pixels.)
        # So, only the y value is checked
        valid = True
    return count, data, has_em_image, valid, previous_y // height_norm


def _readframe_dummy(rawdata):
    count = 0
    previous_y = 0
    for value in rawdata:
        value_type = value & 0xf000
        value &= 0xfff
        if (value_type == 0x9000):
            y = value
            if y < previous_y:
                break
            previous_y = y
        count += 1
    return count


def _read_eds(filename, **kwargs):
    header = {}
    fd = open(filename, "br")
    # file_magic
    _ = np.fromfile(fd, "<I", 1)[0]
    np.fromfile(fd, "<b", 6)
    header["filedate"] = datetime(1899, 12, 30) + timedelta(
        days=np.fromfile(fd, "<d", 1)[0]
    )
    header["sp_name"] = _decode(fd.read(80).rstrip(b"\x00"))
    header["username"] = _decode(fd.read(32).rstrip(b"\x00"))

    np.fromfile(fd, "<i", 1)  # 1
    header["arr"] = np.fromfile(fd, "<d", 10)

    np.fromfile(fd, "<i", 1)  # 7
    np.fromfile(fd, "<d", 1)[0]
    header["Esc"] = np.fromfile(fd, "<d", 1)[0]
    header["Fnano F"] = np.fromfile(fd, "<d", 1)[0]
    header["E Noise"] = np.fromfile(fd, "<d", 1)[0]
    header["CH Res"] = np.fromfile(fd, "<d", 1)[0]
    header["live time"] = np.fromfile(fd, "<d", 1)[0]
    header["real time"] = np.fromfile(fd, "<d", 1)[0]
    header["DeadTime"] = np.fromfile(fd, "<d", 1)[0]
    header["CountRate"] = np.fromfile(fd, "<d", 1)[0]
    header["CountRate n"] = np.fromfile(fd, "<i", 1)[0]
    header["CountRate sum"] = np.fromfile(fd, "<d", 2)
    header["CountRate value"] = np.fromfile(fd, "<d", 1)[0]
    np.fromfile(fd, "<d", 1)[0]
    header["DeadTime n"] = np.fromfile(fd, "<i", 1)[0]
    header["DeadTime sum"] = np.fromfile(fd, "<d", 2)
    header["DeadTime value"] = np.fromfile(fd, "<d", 1)[0]
    np.fromfile(fd, "<d", 1)[0]
    header["CoefA"] = np.fromfile(fd, "<d", 1)[0]
    header["CoefB"] = np.fromfile(fd, "<d", 1)[0]
    header["State"] = _decode(fd.read(32).rstrip(b"\x00"))
    np.fromfile(fd, "<i", 1)[0]
    np.fromfile(fd, "<d", 1)[0]
    header["Tpl"] = _decode(fd.read(32).rstrip(b"\x00"))
    header["NumCH"] = np.fromfile(fd, "<i", 1)[0]
    data = np.fromfile(fd, "<i", header["NumCH"])

    footer = {}
    _ = np.fromfile(fd, "<i", 1)

    n_fbd_elem = np.fromfile(fd, "<i", 1)[0]
    if n_fbd_elem != 0:
        list_fbd_elem = np.fromfile(fd, "<H", n_fbd_elem)
        footer["Excluded elements"] = list_fbd_elem

    n_elem = np.fromfile(fd, "<i", 1)[0]
    if n_elem != 0:
        elems = {}
        for i in range(n_elem):
            # mark elem
            _ = np.fromfile(fd, "<i", 1)[0]  # = 2
            # Z
            _ = np.fromfile(fd, "<H", 1)[0]
            mark1, mark2 = np.fromfile(fd, "<i", 2)  # = 1, 0
            roi_min, roi_max = np.fromfile(fd, "<H", 2)
            # unknown
            _ = np.fromfile(fd, "<b", 14)
            energy, unknow1, unknow2, unknow3 = np.fromfile(fd, "<d", 4)
            elem_name = _decode(fd.read(32).rstrip(b"\x00"))
            # mark3?
            _ = np.fromfile(fd, "<i", 1)[0]
            n_line = np.fromfile(fd, "<i", 1)[0]
            lines = {}
            for j in range(n_line):
                # mark_line?
                _ = np.fromfile(fd, "<i", 1)[0]  # = 1
                e_line = np.fromfile(fd, "<d", 1)[0]
                z = np.fromfile(fd, "<H", 1)[0]
                e_length = np.fromfile(fd, "<b", 1)[0]
                e_name = _decode(fd.read(e_length).rstrip(b"\x00"))
                l_length = np.fromfile(fd, "<b", 1)[0]
                l_name = _decode(fd.read(l_length).rstrip(b"\x00"))
                detect = np.fromfile(fd, "<i", 1)[0]
                lines[e_name + "_" + l_name] = {
                    "energy": e_line,
                    "Z": z,
                    "detection": detect,
                }
            elems[elem_name] = {
                "Z": z,
                "Roi_min": roi_min,
                "Roi_max": roi_max,
                "Energy": energy,
                "Lines": lines,
            }
        footer["Selected elements"] = elems

    n_quanti = np.fromfile(fd, "<i", 1)[0]
    if n_quanti != 0:
        # all unknown
        _ = np.fromfile(fd, "<i", 1)[0]
        _ = np.fromfile(fd, "<i", 1)[0]
        _ = np.fromfile(fd, "<i", 1)[0]
        _ = np.fromfile(fd, "<d", 1)[0]
        _ = np.fromfile(fd, "<i", 1)[0]
        _ = np.fromfile(fd, "<i", 1)[0]
        quanti = {}
        for i in range(n_quanti):
            # mark elem
            _ = np.fromfile(fd, "<i", 1)[0]  # = 2
            z = np.fromfile(fd, "<H", 1)[0]
            mark1, mark2 = np.fromfile(fd, "<i", 2)  # = 1, 0
            energy, unkn6 = np.fromfile(fd, "<d", 2)
            mass1 = np.fromfile(fd, "<d", 1)[0]
            error = np.fromfile(fd, "<d", 1)[0]
            atom = np.fromfile(fd, "<d", 1)[0]
            ox_name = _decode(fd.read(16).rstrip(b"\x00"))
            mass2 = np.fromfile(fd, "<d", 1)[0]
            # K
            _ = np.fromfile(fd, "<d", 1)[0]
            counts = np.fromfile(fd, "<d", 1)[0]
            # all unknown
            _ = np.fromfile(fd, "<d", 1)[0]
            _ = np.fromfile(fd, "<d", 1)[0]
            _ = np.fromfile(fd, "<i", 1)[0]
            _ = np.fromfile(fd, "<i", 1)[0]
            _ = np.fromfile(fd, "<d", 1)[0]
            quanti[ox_name] = {
                "Z": z,
                "Mass1 (%)": mass1,
                "Error": error,
                "Atom (%)": atom,
                "Mass2 (%)": mass2,
                "Counts": counts,
                "Energy": energy,
            }
        footer["Quanti"] = quanti

    e = np.fromfile(fd, "<i", 1)
    if e == 5:
        footer["Parameters"] = {
            "DetT": _decode(fd.read(16).rstrip(b"\x00")),
            "SEM": _decode(fd.read(16).rstrip(b"\x00")),
            "Port": _decode(fd.read(16).rstrip(b"\x00")),
            "AccKV": np.fromfile(fd, "<d", 1)[0],
            "AccNA": np.fromfile(fd, "<d", 1)[0],
            "skip": np.fromfile(fd, "<b", 38),
            "MnKaRES": np.fromfile(fd, "d", 1)[0],
            "WorkD": np.fromfile(fd, "d", 1)[0],
            "InsD": np.fromfile(fd, "d", 1)[0],
            "XtiltAng": np.fromfile(fd, "d", 1)[0],
            "TakeAng": np.fromfile(fd, "d", 1)[0],
            "IncAng": np.fromfile(fd, "d", 1)[0],
            "skip2": np.fromfile(fd, "<i", 1)[0],
            "ScanSize": np.fromfile(fd, "d", 1)[0],
            "DT_64": np.fromfile(fd, "<H", 1)[0],
            "DT_128": np.fromfile(fd, "<H", 1)[0],
            "DT_256": np.fromfile(fd, "<H", 1)[0],
            "DT_512": np.fromfile(fd, "<H", 1)[0],
            "DT_1K": np.fromfile(fd, "<H", 1)[0],
            "DetH": np.fromfile(fd, "d", 1)[0],
            "DirAng": np.fromfile(fd, "d", 1)[0],
            "XtalAng": np.fromfile(fd, "d", 1)[0],
            "ElevAng": np.fromfile(fd, "d", 1)[0],
            "ValidSize": np.fromfile(fd, "d", 1)[0],
            "WinCMat": _decode(fd.read(4).rstrip(b"\x00")),
            "WinCZ": np.fromfile(fd, "<H", 1)[0],
            "WinCThic": np.fromfile(fd, "d", 1)[0],
            "WinChem": _decode(fd.read(16).rstrip(b"\x00")),
            "WinChem_nelem": np.fromfile(fd, "<H", 1)[0],
            "WinChem_Z1": np.fromfile(fd, "<H", 1)[0],
            "WinChem_Z2": np.fromfile(fd, "<H", 1)[0],
            "WinChem_Z3": np.fromfile(fd, "<H", 1)[0],
            "WinChem_Z4": np.fromfile(fd, "<H", 1)[0],
            "WinChem_Z5": np.fromfile(fd, "<H", 1)[0],
            "WinChem_m1": np.fromfile(fd, "d", 1)[0],
            "WinChem_m2": np.fromfile(fd, "d", 1)[0],
            "WinChem_m3": np.fromfile(fd, "d", 1)[0],
            "WinChem_m4": np.fromfile(fd, "d", 1)[0],
            "WinChem_m5": np.fromfile(fd, "d", 1)[0],
            "WinThic": np.fromfile(fd, "d", 1)[0],
            "WinDens": np.fromfile(fd, "d", 1)[0],
            "SpatMat": _decode(fd.read(4).rstrip(b"\x00")),
            "SpatZ": np.fromfile(fd, "<H", 1)[0],
            "SpatThic": np.fromfile(fd, "d", 1)[0],
            "SiDead": np.fromfile(fd, "d", 1)[0],
            "SiThic": np.fromfile(fd, "d", 1)[0],
        }

    hv = footer["Parameters"]["AccKV"]
    if hv <= 30.0:
        mode = "SEM"
    else:
        mode = "TEM"

    axes = [
        {
            "name": "Energy",
            "size": header["NumCH"],
            "offset": header["CoefB"],
            "scale": header["CoefA"],
            "units": "keV",
        }
    ]

    metadata = {
        "Acquisition_instrument": {
            mode: {
                "beam_energy": hv,
                "Detector": {
                    "EDS": {
                        "azimuth_angle": footer["Parameters"]["DirAng"],
                        "detector_type": footer["Parameters"]["DetT"],
                        "elevation_angle": footer["Parameters"]["ElevAng"],
                        "energy_resolution_MnKa": footer["Parameters"]["MnKaRES"],
                        "live_time": header["live time"],
                    },
                },
            },
        },
        "General": {
            "original_filename": os.path.basename(filename),
            "date": header["filedate"].date().isoformat(),
            "time": header["filedate"].time().isoformat(),
            "title": "EDX",
        },
        "Signal": {
            "record_by": "spectrum",
            "quantity": "X-rays (Counts)",
            "signal_type": "EDS_" + mode,
        },
    }

    dictionary = {
        "data": data,
        "axes": axes,
        "metadata": metadata,
        "original_metadata": {"Header": header, "Footer": footer},
    }

    return dictionary


extension_to_reader_mapping = {"img": _read_img,
                               "map": _read_img,
                               "pts": _read_pts,
                               "eds": _read_eds}


def _decode(bytes_string):
    try:
        string = bytes_string.decode("utf-8")
    except:
        # See https://github.com/hyperspy/hyperspy/issues/2812
        string = bytes_string.decode("shift_jis")

    return string
