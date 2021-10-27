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
    dictionary = []
    file_ext = os.path.splitext(filename)[-1][1:].lower()
    if file_ext == "asw":
        fd = open(filename, "br")
        file_magic = np.fromfile(fd, "<I", 1)[0]
        if file_magic == 0:
            fd.seek(12)
            filetree = parsejeol(fd)
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


def read_img(filename, scale=None, **kwargs):
    fd = open(filename, "br")
    file_magic = np.fromfile(fd, "<I", 1)[0]
    if file_magic == 52:
        # fileformat
        _ = decode(fd.read(32).rstrip(b"\x00"))
        head_pos, head_len, data_pos = np.fromfile(fd, "<I", 3)
        fd.seek(data_pos + 12)
        header_long = parsejeol(fd)
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


def read_pts(filename, scale=None, rebin_energy=1, sum_frames=True,
             SI_dtype=np.uint8, cutoff_at_kV=None, downsample=1,
             only_valid_data=True, read_em_image=False, frame_list=None,
             si_lazy=False, lazy=False,
             **kwargs):
    """
    rawdata : ndarray of uint16
    	spectrum image part of pts file
    scale : list of float
    	-scale[2], scale[3] are the positional scale from asw data, 
    	default is None, calc from pts internal data
    rebin_energy : int
    	Binning parameter along energy axis. Must be 2^n.
    sum_frames : bool
	integrate along frame axis if sum_frames == True
    SI_dtype : dtype
        data type for spectrum image. default is uint8
    cutoff_at_kV : float
	cutoff energy to reduce memory size of spectrum image, default: None (do not cutoff)
    downsample : int or [int, int]
    	downsample along spacial axes to reduce memory size of spectrum image
	value must be 2^n.
	default: 1 (do not downsample)
    only_valid_data : bool, default True
    	read incomplete frame if only_valid_data == False
        (usually interrupting mesurement makes incomplete frame)
    read_em_image : bool, default False
        read SEM/STEM image from pts file if read_em_image == True
    frame_list : list of int, default None
    	list of frames to be read (None for all data)
    si_lazy : bool, default False
    	read spectrum image into sparse array if si_lazy == True
    	SEM/STEM image is always read into dense array (np.ndarray)
    lazy : bool, default False
	set lazy flag not only spectrum image but also other data,
	if lazy == True. This also set si_lazy = True.
	Only the spectrum image data is read as a sparse array,
	The others are read as a dense array even if the lazy flag is set.
	

    Returns
    -------
    dictionary :
    	dictionary of spectrum image, axes and metadata
    	(list of dictionaries of spectrum image and SEM/STEM image if read_em_image == True)
    """
    if lazy:
        si_lazy = True
    fd = open(filename, "br")
    file_magic = np.fromfile(fd, "<I", 1)[0]

    def check_multiple(factor, number, string):
        if factor > 1 and number % factor != 0:
            raise ValueError(f'`{string}` must be a multiple of {number}.')

    check_multiple(rebin_energy, 4096, 'rebin_energy')
    rebin_energy = int(rebin_energy)

    if file_magic == 304:
        # fileformat
        _ = decode(fd.read(8).rstrip(b"\x00"))
        a, b, head_pos, head_len, data_pos, data_len = np.fromfile(fd, "<I", 6)
        # groupname
        _ = decode(fd.read(128).rstrip(b"\x00"))
        # memo
        _ = decode(fd.read(132).rstrip(b"\x00"))
        datefile = datetime(1899, 12, 30) + timedelta(days=np.fromfile(fd, "d", 1)[0])
        fd.seek(head_pos + 12)
        header = parsejeol(fd)
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

        data_shape = [height, width, channel_number]
        # Sweep value is not reliable
        # sweep = meas_data_header["Doc"]["Sweep"]
        frame_ptr_list, last_valid, has_em_image = pts_prescan(rawdata, width, height)
        if last_valid or not only_valid_data:
            sweep = len(frame_ptr_list) - 1
        else:
            sweep = len(frame_ptr_list) - 2
        if not frame_list:
            frame_list = range(sweep)
        frame_list = list(frame_list)
        frame_list2 = []
        for frame_idx in frame_list:
            if frame_idx < 0 or frame_idx >= sweep:
                _logger.info(f"Ignoreing frame {frame_idx} : Selected frame is not found in pts data.")
                continue
            frame_list2.append(frame_idx)
        frame_list = np.array(frame_list2)

        data, em_data = readcube(rawdata, frame_ptr_list, frame_list,
                                 width, height, channel_number,
                                 width_norm, height_norm, rebin_energy, 
                                 sum_frames, si_lazy, SI_dtype)

        if sum_frames:
            axes_em = []
        else:
            axes_em = [{
                "index_in_array": 0,
                "name": "Frame",
                "size": frame_list.size,
                "offset": 0,
                "scale": pixel_time*height*width/1E3,
                "units": 's',
            }]
        axes_em.extend( [
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
                "title": "EDX",
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
                "title": "SEM/STEM Image"
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
            "attributes": {
                "_lazy" : si_lazy
            }
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

    return dictionary


def parsejeol(fd):
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
                kwrd = decode(kwrd[:-1])
            else:
                kwrd = decode(kwrd)
            val_type, val_len = np.fromfile(fd, "<i", 2)
            tmp_list.append(kwrd)
            if val_type == 0:
                tmp_dict[kwrd] = {}
            else:
                c_type = jTYPE[val_type]
                arr_len = val_len // np.dtype(c_type).itemsize
                if c_type == "c":
                    value = fd.read(val_len).rstrip(b"\x00")
                    value = decode(value).split("\x00")
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


@numba.njit(cache=True)
def pts_prescan(rawdata, width, height):   # pragma: no cover
    """
    Prescan data section before decoding because the sweep count in the header
    sometime shows broken value depend on the JEOL software version.

    Parameters
    ----------
    rawdata : ndarray of uint16
    	spectrum image part of pts file

    width, height : int
    	image width and height


    Returns
    -------
    frame_ptr_list : np.ndarray of int
    	list of frame head address and also bottom address of rawdata
    valid_last : bool
      is the last frame valid(True) or not(False)
    has_em_image : bool
      is the STEM/SEM-BF data available(True) or not(False)
    """
    MAX_VAL = 4096
    width_norm = int(MAX_VAL / width)
    height_norm = int(MAX_VAL / width)
    max_x = MAX_VAL - width_norm
    max_y = MAX_VAL - height_norm

    previous_x = 0
    previous_y = MAX_VAL
    pointer = 0
    frame_ptr_list = []

    has_em_image = False
    last_valid = False

    for value in rawdata:
        value_type = value & 0xF000
        if value_type == 0x8000:  # pos x
            previous_x = value & 0xFFF
        elif value_type == 0x9000:  # pox y
            y = value & 0xFFF
            if y < previous_y:
                frame_ptr_list.append(pointer)
            previous_y = y
        elif value_type == 0xa000: # image
            has_em_image = True
        #elif value_type == 0xb000: #EDS
        #    pass
        pointer += 1

    if previous_y == max_y and (previous_x == max_x or not has_em_image):
        # if the STEM-ADF/SEM-BF image is not included in pts file,
        # maximum value of x is usually smaller than max_x,
        # (sometimes no x-ray events occur in a last few pixels.)
        # So, only the y value is checked
        last_valid = True

    frame_ptr_list.append(pointer)  # end of last frame
    return np.array(frame_ptr_list), last_valid, has_em_image


@numba.njit(cache=True)
def readframe_dense(rawdata, hypermap, em_image, rebin_energy, channel_number,
              width_norm, height_norm, max_value): # pragma: nocover
    """
    Read one frame from pts file. Used in a inner loop of readcube function.
    This function always read SEM/STEM image even if read_em_image == False

    Parameters
    ----------
    rawdata : ndarray of uint16
    	slice of one frame raw data from whole raw data 
    hypermap : ndarray(width, height, channel_number)
    	np.ndarray to store decoded spectrum image.
    em_image : np.ndarray(width, height), dtype = np.uint16 or np.uint32
    	np.ndarray to store decoded SEM/TEM image.
    rebin_energy : int
    	Binning parameter along energy axis. Must be 2^n.
    channel_number : int
    	Limit of channel to reduce data size
    width_norm, height_norm : int
    	scanning step
    max_value : int
    	limit of the data type used in hypermap array

    Returns
    -------
    None

    hypermap and em_image array will be modified
    """
    for value in rawdata:
        dtype = value & 0xf000
        dval = value & 0xfff
        if dtype == 0x8000:
            x = dval // width_norm
        elif dtype == 0x9000:
            y = dval // height_norm
        elif dtype == 0xa000:
            em_image[y, x] += dval
        elif dtype == 0xb000:
            z = dval // rebin_energy
            if z < channel_number:
                hypermap[y, x, z] += 1
                if hypermap[y, x, z] == max_value:
                    raise ValueError("The range of the dtype is too small, "
                                     "use `SI_dtype` to set a dtype with "
                                     "higher range.")

def readcube(rawdata, frame_ptr_list, frame_list, 
             width, height, channel_number,
             width_norm, height_norm,
             rebin_energy, sum_frames, si_lazy, SI_dtype):  # pragma: no cover
    """
    Read spectrum image and TEM/SEM image into dense np.ndarray.
    can not apply numba to this function (variable dimenstion numpy.ndarray)

    Parameters
    ----------
    rawdata : ndarray of uint16
    	spectrum image part of pts file
    frame_ptr_list : list of int
    	list of index pointer of frame starting point
    frame_list : list of int
    	list of frames to be read
    width_norm, height_norm : int
    	scan step
    rebin_energy : int
    	Binning parameter along energy axis. Must be 2^n.
    channel_number : int
    	Limit of channel to reduce data size
    sum_frames : bool
	integrate along frame axis if sum_frames == True
    si_lazy : bool
	read spectrum image as dask.array if si_lazy == True
    SI_dtype : dtype
        data type for spectrum image.

    Returns
    -------
    hypermap : ndarray(frame, width, height, channel_number) or dask.array
    	np.ndarray(dask.array if si_lazy==True)  of  spectrum image.
    em_image : np.ndarray(width, height), dtype = np.uint16 or np.uint32
    	np.ndarray of SEM/TEM image.
    """

    n_frames = len(frame_list)
    EM_dtype = np.uint16
    frame_step = 1
    if sum_frames:
        n_frames = 1
        frame_step = 0
        if n_frames > 16:
            EM_dtype = np.uint32

    if si_lazy:
        data_list = []
    else:
        hypermap = np.zeros((n_frames, width, height, channel_number),
                            dtype=SI_dtype)
    em_image = np.zeros((n_frames, width, height), dtype=EM_dtype)
    max_value =  np.iinfo(SI_dtype).max

    frame_count = 0
    for frame_idx in frame_list:
        p_start = frame_ptr_list[frame_idx]
        p_end = frame_ptr_list[frame_idx + 1]
        if si_lazy:
            data_list.append(
                readframe_lazy(rawdata[p_start:p_end],
                               em_image[frame_count],
                               rebin_energy, channel_number,
                               width_norm, height_norm, max_value))
        else:
            readframe_dense(rawdata[p_start:p_end],
                        hypermap[frame_count], em_image[frame_count],
                        rebin_energy, channel_number,
                        width_norm, height_norm, max_value)
        frame_count += frame_step
    if not si_lazy:
        if sum_frames:
            return hypermap[0], em_image[0]
        return hypermap, em_image

    length = 0
    for d in data_list:
        length += len(d)
    v = np.zeros(shape=(5, length), dtype=np.uint16)

    if sum_frames:
        data_shape = [width, height, channel_number]
    else:
        data_shape = [frame_count, width, height, channel_number]


    ptr = 0
    frame_count = 0
    for d in data_list:
        flen = len(d)
        pv = v[:,ptr:ptr+flen]
        pv[1:4, :] = np.array(d).transpose()
        pv[0,:] = frame_count
        pv[4,:] = 1
        ptr += flen
        frame_count += 1
    import sparse
    import dask.array as da
    if sum_frames:
        ar_s = sparse.COO(v[1:4], v[4], shape=data_shape)
    else:
        ar_s = sparse.COO(v[0:4], v[4], shape=data_shape)
    return da.from_array(ar_s, asarray=False), em_image



@numba.njit(cache=True)
def readframe_lazy(rawdata, em_image, rebin_energy, channel_number,
                   width_norm, height_norm, read_em_image):  # pragma: no cover
    data = []
    for value in rawdata:
        dtype = value & 0xf000
        dval = value & 0xfff
        if dtype == 0x8000:
            x = dval // width_norm
        elif dtype == 0x9000:
            y = dval // height_norm
        elif dtype == 0xa000:
            em_image[y, x] = dval
        elif dtype == 0xb000:    # energy axis
            z = dval // rebin_energy
            if z < channel_number:
                data.append([y, x, z])
    return data





def read_eds(filename, **kwargs):
    header = {}
    fd = open(filename, "br")
    # file_magic
    _ = np.fromfile(fd, "<I", 1)[0]
    np.fromfile(fd, "<b", 6)
    header["filedate"] = datetime(1899, 12, 30) + timedelta(
        days=np.fromfile(fd, "<d", 1)[0]
    )
    header["sp_name"] = decode(fd.read(80).rstrip(b"\x00"))
    header["username"] = decode(fd.read(32).rstrip(b"\x00"))

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
    header["State"] = decode(fd.read(32).rstrip(b"\x00"))
    np.fromfile(fd, "<i", 1)[0]
    np.fromfile(fd, "<d", 1)[0]
    header["Tpl"] = decode(fd.read(32).rstrip(b"\x00"))
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
            elem_name = decode(fd.read(32).rstrip(b"\x00"))
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
                e_name = decode(fd.read(e_length).rstrip(b"\x00"))
                l_length = np.fromfile(fd, "<b", 1)[0]
                l_name = decode(fd.read(l_length).rstrip(b"\x00"))
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
            ox_name = decode(fd.read(16).rstrip(b"\x00"))
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
            "DetT": decode(fd.read(16).rstrip(b"\x00")),
            "SEM": decode(fd.read(16).rstrip(b"\x00")),
            "Port": decode(fd.read(16).rstrip(b"\x00")),
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
            "WinCMat": decode(fd.read(4).rstrip(b"\x00")),
            "WinCZ": np.fromfile(fd, "<H", 1)[0],
            "WinCThic": np.fromfile(fd, "d", 1)[0],
            "WinChem": decode(fd.read(16).rstrip(b"\x00")),
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
            "SpatMat": decode(fd.read(4).rstrip(b"\x00")),
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


extension_to_reader_mapping = {"img": read_img,
                               "map": read_img,
                               "pts": read_pts,
                               "eds": read_eds}


def decode(bytes_string):
    try:
        string = bytes_string.decode("utf-8")
    except:
        # See https://github.com/hyperspy/hyperspy/issues/2812
        string = bytes_string.decode("shift_jis")

    return string
