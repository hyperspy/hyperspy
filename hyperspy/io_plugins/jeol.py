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
                                        dictionary.append(d)
        else:
            _logger.warning("Not a valid JEOL asw format")
    else:
        d = extension_to_reader_mapping[file_ext](filename, **kwds)
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
            xscale = 1
            yscale = 1
            units = "px"

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
             only_valid_data=True, **kwargs):
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

        if scale is not None:
            xscale = -scale[2] / width
            yscale = scale[3] / height
            units = "µm"
        else:
            scale = header["PTTD Param"]["Params"]["PARAMPAGE0_SEM"]["ScanSize"] / meas_data_header["MeasCond"]["Mag"] * 1.0E6
            xscale = scale / width
            yscale = scale / height
            units = "nm"

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
            {
                "name": "Energy",
                "size": channel_number,
                "offset": energy_offset,
                "scale": energy_scale,
                "units": "keV",
            },
        ]
        # pixel time in milliseconds
        pixel_time = meas_data_header["Doc"]["DwellTime(msec)"]

        data_shape = [height, width, channel_number]
        sweep = meas_data_header["Doc"]["Sweep"]
        if not sum_frames:
            data_shape.insert(0, sweep)
            if not only_valid_data:
                data_shape[0] += 1 # for partly swept frame
            axes.insert(0, {
                "index_in_array": 0,
                "name": "Frame",
                "size": sweep,
                "offset": 0,
                "scale": pixel_time*height*width/1E3,
                "units": 's',
            })

        data = np.zeros(data_shape, dtype=SI_dtype)
        datacube_reader = readcube if sum_frames else readcube_frames
        data, swept = datacube_reader(rawdata, data, sweep, only_valid_data,
                                      rebin_energy, channel_number,
                               width_norm, height_norm, np.iinfo(SI_dtype).max)

        if not sum_frames and not only_valid_data:
            if  (sweep == swept):
                data = data[:sweep]
            else:
                axes[0]["size"] = swept

        if sweep < swept and only_valid_data:
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

        dictionary = {
            "data": data,
            "axes": axes,
            "metadata": metadata,
            "original_metadata": header,
        }
    else:
        _logger.warning("Not a valid JEOL pts format")

    fd.close()

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
def readcube(rawdata, hypermap, sweep, only_valid_data, rebin_energy, 
            channel_number, width_norm, height_norm, max_value):  # pragma: no cover
    frame_idx = 0
    previous_y = 0
    for value in rawdata:
        if value >= 32768 and value < 36864:
            x = int((value - 32768) / width_norm)
        elif value >= 36864 and value < 40960:
            y = int((value - 36864) / height_norm)
            if y < previous_y:
                frame_idx += 1
                if frame_idx >= sweep:
                    # skip incomplete_frame
                    break 
            previous_y = y
        elif value >= 45056 and value < 49152:
            z = int((value - 45056) / rebin_energy)
            if z < channel_number:
                hypermap[y, x, z] += 1
                if hypermap[y, x, z] == max_value:
                    raise ValueError("The range of the dtype is too small, "
                                     "use `SI_dtype` to set a dtype with "
                                     "higher range.")
    return hypermap, frame_idx + 1


@numba.njit(cache=True)
def readcube_frames(rawdata, hypermap, sweep, only_valid_data, rebin_energy, channel_number,
             width_norm, height_norm, max_value):  # pragma: no cover
    """
    We need to create a separate function, because numba.njit doesn't play well
    with an array having its shape depending on something else
    """
    frame_idx = 0
    previous_y = 0
    for value in rawdata:
        if value >= 32768 and value < 36864:
            x = int((value - 32768) / width_norm)
        elif value >= 36864 and value < 40960:
            y = int((value - 36864) / height_norm)
            if y < previous_y:
                frame_idx += 1
                if frame_idx == sweep: # incomplete frame exist
                    if only_valid_data:
                        break  # ignore
                if frame_idx > sweep:
                    raise ValueError("The frame number is too large")
            previous_y = y
        elif value >= 45056 and value < 49152:
            z = int((value - 45056) / rebin_energy)
            if z < channel_number:
                hypermap[frame_idx, y, x, z] += 1
                if hypermap[frame_idx, y, x, z] == max_value:
                    raise ValueError("The range of the dtype is too small, "
                                     "use `SI_dtype` to set a dtype with "
                                     "higher range.")
    return hypermap, frame_idx + 1


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
