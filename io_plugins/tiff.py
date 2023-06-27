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
import re
import csv
import warnings
import logging
from datetime import datetime, timedelta
from dateutil import parser
from tifffile import imwrite, TiffFile, TIFF
import tifffile
import traits.api as t
import numpy as np
from packaging.version import Version

from hyperspy.api_nogui import _ureg
from hyperspy.misc import rgb_tools
from hyperspy.misc.date_time_tools import get_date_time_from_metadata

_logger = logging.getLogger(__name__)

# Plugin characteristics
# ----------------------
format_name = 'TIFF'
description = ('Import/Export standard image formats Christoph Gohlke\'s '
               'tifffile library')
full_support = False
file_extensions = ['tif', 'tiff']
default_extension = 0  # tif
# Writing capabilities
writes = [(2, 0), (2, 1)]
non_uniform_axis = False
# ----------------------


axes_label_codes = {
    'X': "width",
    'Y': "height",
    'S': "sample",
    'P': "plane",
    'I': "image series",
    'Z': "depth",
    'C': "color|em-wavelength|channel",
    'E': "ex-wavelength|lambda",
    'T': "time",
    'R': "region|tile",
    'A': "angle",
    'F': "phase",
    'H': "lifetime",
    'L': "exposure",
    'V': "event",
    'Q': t.Undefined,
    '_': t.Undefined}


def file_writer(filename, signal, export_scale=True, extratags=[], **kwds):
    """Writes data to tif using Christoph Gohlke's tifffile library

    Parameters
    ----------
    filename: str
    signal: a BaseSignal instance
    export_scale: bool
        default: True
        Export the scale and the units (compatible with DM and ImageJ) to
        appropriate tags.
    """

    data = signal.data
    if signal.is_rgbx is True:
        data = rgb_tools.rgbx2regular_array(data)
        photometric = "RGB"
    else:
        photometric = "MINISBLACK"
    if 'description' in kwds.keys() and export_scale:
        kwds.pop('description')
        _logger.warning(
            "Description and export scale cannot be used at the same time, "
            "because it is incompability with the 'ImageJ' tiff format")
    if export_scale:
        kwds.update(_get_tags_dict(signal, extratags=extratags))
        _logger.debug(f"kwargs passed to tifffile.py imsave: {kwds}")

        if 'metadata' not in kwds.keys():
            # Because we write the calibration to the ImageDescription tag
            # for imageJ, we need to disable tiffile from also writing JSON
            # metadata if not explicitely requested
            # (https://github.com/cgohlke/tifffile/issues/21)
            kwds['metadata'] = None

    if 'date' in signal.metadata['General']:
        dt = get_date_time_from_metadata(signal.metadata,
                                         formatting='datetime')
        kwds['datetime'] = dt

    imwrite(filename,
            data,
            software="hyperspy",
            photometric=photometric,
            **kwds)


def file_reader(filename, force_read_resolution=False, lazy=False, **kwds):
    """
    Read data from tif files using Christoph Gohlke's tifffile library.
    The units and the scale of images saved with ImageJ or Digital
    Micrograph is read. There is limited support for reading the scale of
    files created with Zeiss and FEI SEMs.

    Parameters
    ----------
    filename: str
        Name of the file to read
    force_read_resolution: bool
        Force reading the x_resolution, y_resolution and the resolution_unit
        of the tiff tags.
        See http://www.awaresystems.be/imaging/tiff/tifftags/resolutionunit.html
        Default is False.
    lazy : bool
        Load the data lazily. Default is False
    **kwds, optional
    """
    tmp = kwds.pop('hamamatsu_streak_axis_type', None)


    with TiffFile(filename, **kwds) as tiff:
        if tmp is not None:
            kwds.update({'hamamatsu_streak_axis_type': tmp})
        dict_list = [_read_serie(tiff, serie, filename, force_read_resolution,
                                 lazy=lazy, **kwds) for serie in tiff.series]

    return dict_list


def _order_axes_by_name(names: list, scales: dict, offsets: dict, units: dict):
    """order axes by names in lists"""
    scales_new = [1.0] * len(names)
    offsets_new = [0.0] * len(names)
    units_new = [t.Undefined] * len(names)
    for i, name in enumerate(names):
        if name == 'height':
            scales_new[i] = scales['x']
            offsets_new[i] = offsets['x']
            units_new[i] = units['x']
        elif name == 'width':
            scales_new[i] = scales['y']
            offsets_new[i] = offsets['y']
            units_new[i] = units['y']
        elif name in ['depth', 'image series', 'time']:
            scales_new[i] = scales['z']
            offsets_new[i] = offsets['z']
            units_new[i] = units['z']
    return scales_new, offsets_new, units_new


def _build_axes_dictionaries(shape, names=None, scales=None, offsets=None,
                             units=None):
    """Build axes dictionaries from a set of lists"""
    if names is None:
        names = [""] * len(shape)
    if scales is None:
        scales = [1.0] * len(shape)
    if scales is None:
        scales = [0.0] * len(shape)
    if units is None:
        scales = [t.Undefined] * len(shape)

    axes = [{'size': size, 'name': str(name), 'scale': scale, 'offset': offset, 'units': unit}
            for size, name, scale, offset, unit in zip(shape, names, scales, offsets, units)]
    return axes


def _read_serie(tiff, serie, filename, force_read_resolution=False,
                lazy=False, memmap=None, **kwds):
    axes = serie.axes
    page = serie.pages[0]
    if hasattr(serie, 'shape'):
        shape = serie.shape
        dtype = serie.dtype
    else:
        shape = serie['shape']
        dtype = serie['dtype']

    is_rgb = page.photometric == TIFF.PHOTOMETRIC.RGB
    _logger.debug("Is RGB: %s" % is_rgb)
    if is_rgb:
        axes = axes[:-1]
        names = ['R', 'G', 'B', 'A']
        lastshape = shape[-1]
        dtype = np.dtype({'names': names[:lastshape],
                          'formats': [dtype] * lastshape})
        shape = shape[:-1]

    if Version(tifffile.__version__) >= Version("2020.2.16"):
        op = {tag.name: tag.value for tag in page.tags}
    else:
        op = {key: tag.value for key, tag in page.tags.items()}

    names = [axes_label_codes[axis] for axis in axes]

    _logger.debug('Tiff tags list: %s' % op)
    _logger.debug("Photometric: %s" % op['PhotometricInterpretation'])
    _logger.debug('is_imagej: {}'.format(page.is_imagej))

    try:
        axes = _parse_scale_unit(tiff, page, op, shape, force_read_resolution, names, **kwds)
    except BaseException:
        _logger.info("Scale and units could not be imported")
        axes = _build_axes_dictionaries(shape, names)

    md = {'General': {'original_filename': os.path.split(filename)[1]},
           'Signal': {'signal_type': "", 'record_by': "image" }}

    if 'DateTime' in op:
        dt = None
        try:
            dt = datetime.strptime(op['DateTime'], "%Y:%m:%d %H:%M:%S")
        except:
            try:
                if 'ImageDescription' in op:
                    # JEOL SightX.
                    _dt = op['ImageDescription']['DateTime']
                    md['General']['date'] = _dt[0:10]
                    # 1 extra digit for millisec should be removed
                    md['General']['time'] = _dt[11:26]
                    md['General']['time_zone'] = _dt[-6:]
                    dt = None
                else:
                    dt = datetime.strptime(op['DateTime'], "%Y/%m/%d %H:%M")
            except:
                _logger.info("Date/Time is invalid : " + op['DateTime'])
        if dt is not None:
            md['General']['date'] = dt.date().isoformat()
            md['General']['time'] = dt.time().isoformat()

    #Get the digital micrograph intensity axis
    if _is_digital_micrograph(op):
        intensity_axis = _intensity_axis_digital_micrograph(op)
    else:
        intensity_axis = {}

    if 'units' in intensity_axis:
        md['Signal']['quantity'] = intensity_axis['units']
    if 'scale' in intensity_axis and 'offset' in intensity_axis:
        dic = {'gain_factor': intensity_axis['scale'],
               'gain_offset': intensity_axis['offset']}
        md['Signal']['Noise_properties'] = {'Variance_linear_model': dic}

    data_args = serie, is_rgb
    if lazy:
        from dask import delayed
        from dask.array import from_delayed
        memmap = 'memmap'
        val = delayed(_load_data, pure=True)(*data_args, memmap=memmap, **kwds)
        dc = from_delayed(val, dtype=dtype, shape=shape)
        # TODO: maybe just pass the memmap from tiffile?
    else:
        dc = _load_data(*data_args, memmap=memmap, **kwds)

    if _is_streak_hamamatsu(op):
        op.update({'ImageDescriptionParsed': _get_hamamatsu_streak_description(tiff, op)})

    metadata_mapping = get_metadata_mapping(page, op)
    if 'SightX_Notes' in op:
        md['General']['title'] = op['SightX_Notes']
    return {'data': dc,
            'original_metadata': op,
            'axes': axes,
            'metadata': md,
            'mapping': metadata_mapping,
            }


def _load_data(serie, is_rgb, sl=None, memmap=None, **kwds):
    dc = serie.asarray(out=memmap)
    _logger.debug("data shape: {0}".format(dc.shape))
    if is_rgb:
        dc = rgb_tools.regular_array2rgbx(dc)
    if sl is not None:
        dc = dc[tuple(sl)]
    return dc


def _axes_defaults():
    """Get default axes dictionaries, with offsets and scales"""
    axes_labels = ['x', 'y', 'z']
    scales = {axis: 1.0 for axis in axes_labels}
    offsets = {axis: 0.0 for axis in axes_labels}
    units = {axis: t.Undefined for axis in axes_labels}

    return scales, offsets, units


def _is_force_readable(op, force_read_resolution) -> bool:
    return force_read_resolution and 'ResolutionUnit' in op and 'XResolution' in op


def _axes_force_read(op, shape, names):
    scales, offsets, units = _axes_defaults()
    res_unit_tag = op['ResolutionUnit']
    if res_unit_tag != TIFF.RESUNIT.NONE:
        _logger.debug("Resolution unit: %s" % res_unit_tag)
        scales['x'], scales['y'] = _get_scales_from_x_y_resolution(op)
        # conversion to µm:
        if res_unit_tag == TIFF.RESUNIT.INCH:
            for key in ['x', 'y']:
                units[key] = 'µm'
                scales[key] = scales[key] * 25400
        elif res_unit_tag == TIFF.RESUNIT.CENTIMETER:
            for key in ['x', 'y']:
                units[key] = 'µm'
                scales[key] = scales[key] * 10000

    scales, offsets, units = _order_axes_by_name(names, scales, offsets, units)

    axes = _build_axes_dictionaries(shape, names, scales, offsets, units)

    return axes


def _is_fei(tiff) -> bool:
    return 'fei' in tiff.flags


def _axes_fei(tiff, op, shape, names):
    _logger.debug("Reading FEI tif metadata")

    scales, offsets, units = _axes_defaults()

    op['fei_metadata'] = tiff.fei_metadata
    try:
        del op['FEI_HELIOS']
    except KeyError:
        del op['FEI_SFEG']
    try:
        scales['x'] = float(op['fei_metadata']['Scan']['PixelWidth'])
        scales['y'] = float(op['fei_metadata']['Scan']['PixelHeight'])
        units.update({'x': 'm', 'y': 'm'})
    except KeyError:
        _logger.debug("No 'Scan' information found in FEI metadata; attempting to get pixel size "
                        "from 'IRBeam' metadata")
        try:
            scales['x'] = float(op['fei_metadata']['IRBeam']['HFW']) / float(op['fei_metadata']['Image']['ResolutionX'])
            scales['y'] = float(op['fei_metadata']['IRBeam']['VFW']) / float(op['fei_metadata']['Image']['ResolutionY'])
            units.update({'x': 'm', 'y': 'm'})
        except KeyError:
            _logger.warning("Could not determine pixel size; resulting Signal will not be calibrated")

    scales, offsets, units = _order_axes_by_name(names, scales, offsets, units)

    axes = _build_axes_dictionaries(shape, names, scales, offsets, units)

    return axes


def _is_zeiss(tiff) -> bool:
    return 'sem' in tiff.flags


def _axes_zeiss(tiff, op, shape, names):
    _logger.debug("Reading Zeiss tif pixel_scale")
    scales, offsets, units = _axes_defaults()
    # op['CZ_SEM'][''] is containing structure of primary
    # not described SEM parameters in SI units.
    # tifffiles returns flattened version of the structure (as tuple)
    # and the scale in it is at index 3.
    # The scale is tied with physical display and needs to be multiplied
    # with factor, which is the 1024 (1k) divide by horizontal pixel n.
    # CZ_SEM tiff can contain resolution of lesser precision
    # in the described tags as 'ap_image_pixel_size' and/or
    # 'ap_pixel_size', which depending from ZEISS software version
    # can be absent and thus is not used here.
    scale_in_m = op['CZ_SEM'][''][3] * 1024 / tiff.pages[0].shape[1]
    scales.update({'x': scale_in_m, 'y': scale_in_m})
    units.update({'x': 'm', 'y': 'm'})

    scales, offsets, units = _order_axes_by_name(names, scales, offsets, units)

    axes = _build_axes_dictionaries(shape, names, scales, offsets, units)

    return axes
    # return scales, offsets, units, intensity_axis


def _is_tvips(tiff) -> bool:
    return 'tvips' in tiff.flags


def _axes_tvips(tiff, op, shape, names):
    _logger.debug("Reading TVIPS tif metadata")

    scales, offsets, units = _axes_defaults()

    if 'PixelSizeX' in op['TVIPS'] and 'PixelSizeY' in op['TVIPS']:
        _logger.debug("getting TVIPS scale from PixelSizeX")
        scales['x'] = op['TVIPS']['PixelSizeX']
        scales['y'] = op['TVIPS']['PixelSizeY']
        units.update({'x': 'nm', 'y': 'nm'})
    else:
        _logger.debug("getting TVIPS scale from XYResolution")
        scales['x'], scales['y'] = _get_scales_from_x_y_resolution(
            op, factor=1e-2)
        units.update({'x': 'm', 'y': 'm'})

    scales, offsets, units = _order_axes_by_name(names, scales, offsets, units)

    axes = _build_axes_dictionaries(shape, names, scales, offsets, units)

    return axes


def _is_olympus_sis(page) -> bool:
    return page.is_sis


def _axes_olympus_sis(page, tiff, op, shape, names):
    _logger.debug("Reading Olympus SIS tif metadata")
    scales, offsets, units = _axes_defaults()

    sis_metadata = {}
    for tag_number in [33471, 33560]:
        try:
            sis_metadata = page.tags[tag_number].value
        except Exception:
            pass
    op['Olympus_SIS_metadata'] = sis_metadata
    scales['x'] = round(float(sis_metadata['pixelsizex']), 15)
    scales['y'] = round(float(sis_metadata['pixelsizey']), 15)
    units.update({'x': 'm', 'y': 'm'})

    scales, offsets, units = _order_axes_by_name(names, scales, offsets, units)

    axes = _build_axes_dictionaries(shape, names, scales, offsets, units)

    return axes


def _is_jeol_sightx(op) -> bool:
    return op.get('Make', None) == "JEOL Ltd."


def _axes_jeol_sightx(tiff, op, shape, names):
    # convert xml text to dictionary of tiff op['ImageDescription']
    # convert_xml_to_dict need to remove white spaces before decoding XML
    scales, offsets, units = _axes_defaults()

    jeol_xml = ''.join([line.strip(" \r\n\t\x01\x00") for line in op['ImageDescription'].split('\n')])
    from hyperspy.misc.io.tools import convert_xml_to_dict
    jeol_dict = convert_xml_to_dict(jeol_xml)
    op['ImageDescription'] = jeol_dict['TemReporter']
    eos = op["ImageDescription"]["Eos"]["EosMode"]
    illumi = op["ImageDescription"]["IlluminationSystem"]
    imaging = op["ImageDescription"]["ImageFormingSystem"]

    # TEM/STEM
    is_STEM = eos == "modeASID"
    mode_strs = []
    mode_strs.append("STEM" if is_STEM else "TEM")
    mode_strs.append(illumi["ImageField"][4:])  # Bright Fiels?
    if is_STEM:
        mode_strs.append(imaging["ScanningImageFormingMode"][4:])
    else:
        mode_strs.append(imaging["ImageFormingMode"][4:])
    mode_strs.append(imaging["SelectorString"])  # Mag / Camera Length
    op["SightX_Notes"] = ", ".join(mode_strs)

    res_unit_tag = op['ResolutionUnit']
    if res_unit_tag == TIFF.RESUNIT.INCH:
        scale = 0.0254  # inch/m
    else:
        scale = 0.01  # tiff scaling, cm/m
    # TEM - MAG
    if (eos == "eosTEM") and (imaging["ModeString"] == "MAG"):
        mag = float(imaging["SelectorValue"])
        scales['x'], scales['y'] = _get_scales_from_x_y_resolution(op, factor=scale / mag * 1e9)
        units = {"x": "nm", "y": "nm", "z": "nm"}
    # TEM - DIFF
    elif (eos == "eosTEM") and (imaging["ModeString"] == "DIFF"):
        def wave_len(ht):
            import scipy.constants as constants
            momentum = 2 * constants.m_e * constants.elementary_charge * ht * \
                (1 + constants.elementary_charge * ht / (2 * constants.m_e * constants.c ** 2))
            return constants.h / np.sqrt(momentum)

        camera_len = float(imaging["SelectorValue"])
        ht = float(op["ImageDescription"]["ElectronGun"]["AccelerationVoltage"])
        if imaging["SelectorUnitString"] == "mm":  # convert to "m"
            camera_len /= 1000
        elif imaging["SelectorUnitString"] == "cm":  # convert to "m"
            camera_len /= 100
        scale /= camera_len * wave_len(ht) * 1e9  # in nm
        scales['x'], scales['y'] = _get_scales_from_x_y_resolution(op, factor=scale)
        units = {"x": "1 / nm", "y": "1 / nm", "z": t.Undefined}

    scales, offsets, units = _order_axes_by_name(names, scales, offsets, units)

    axes = _build_axes_dictionaries(shape, names, scales, offsets, units)

    return axes


def _is_streak_hamamatsu(op) -> bool:
    """Determines whether a .tiff page is likely to be a hamamatsu
    streak file based on the original op content.
    """
    is_hamatif = True

    # Check that the original op has an "Artist" field with "Copyright Hamamatsu"
    if 'Artist' not in op:
        is_hamatif = False
        return is_hamatif
    else:
        artist = op['Artist']
        if not artist.startswith("Copyright Hamamatsu"):
            is_hamatif = False
            return is_hamatif

    # Check that the original op has a "Software" corresponding to "HPD-TA"
    if 'Software' not in op:
        is_hamatif = False
        return is_hamatif
    else:
        software = op['Software']
        if not software.startswith('HPD-TA'):
            is_hamatif = False

    return is_hamatif


def _get_hamamatsu_streak_description(tiff, op):
    """Extract a dictionary recursively from the ImageDescription
    Metadata field in a Hamamatsu Streak .tiff file"""

    desc = op['ImageDescription']
    dict_meta = {}
    reader = csv.reader(desc.splitlines(), delimiter=',', quotechar='"')
    for row in reader:
        key = row[0].strip(" []")
        key_dict = {}
        for element in row[1:]:
            spl = element.split('=')
            if len(spl) == 2:
                key_dict[spl[0]] = spl[1].strip('"')
        dict_meta[key] = key_dict

    # Scaling entry
    scaling = dict_meta['Scaling']

    # Address in file where the X axis is saved
    x_scale_address = int(re.findall(r'\d+', scaling['ScalingXScalingFile'])[0])
    xlen = op['ImageWidth']

    # If focus mode is used there is no Y axis
    if scaling['ScalingYScalingFile'].startswith('Focus mode'):
        y_scale_address = None
    else:
        y_scale_address = int(re.findall(r'\d+', scaling['ScalingYScalingFile'])[0])
    ylen = op['ImageLength']

    # Accessing the file as a binary
    fh = tiff.filehandle
    # Reading the x axis
    fh.seek(x_scale_address, 0)
    xax = np.fromfile(fh, dtype='f', count=xlen)
    if y_scale_address is None:
        yax = np.arange(ylen)
    else:
        fh.seek(y_scale_address, 0)
        yax = np.fromfile(fh, dtype='f', count=ylen)

    dict_meta['Scaling']['ScalingXaxis'] = xax
    dict_meta['Scaling']['ScalingYaxis'] = yax

    return dict_meta


def _axes_hamamatsu_streak(tiff, op, shape, names, **kwds):
    _logger.debug("Reading Hamamatsu Streak Map tif metadata")

    if 'hamamatsu_streak_axis_type' in kwds:
        hamamatsu_streak_axis_type = kwds['hamamatsu_streak_axis_type']
    else:
        hamamatsu_streak_axis_type = 'uniform'
        warnings.warn(f"{tiff} contain a non linear axis. By default, "
                      f"a linearized version is initialised, which can "
                      f"induce errors. Use the `hamamatsu_streak_axis_type` keyword to load "
                      f"either a parabolic functional axis using `hamamatsu_streak_axis_type='functional'`, "
                      f"a data axis using `hamamatsu_streak_axis_type='data'`, or use `hamamatsu_streak_axis_type='uniform'`to "
                      f"linearize the axis and make this warning disappear", UserWarning)

    if hamamatsu_streak_axis_type not in ['functional', 'data', 'uniform']:
        hamamatsu_streak_axis_type = 'uniform'
        warnings.warn("The `hamamatsu_streak_axis_type`  argument only admits "
                         "the values `'data'`, `'functional'` and `'uniform'`", UserWarning)

    # Parsing the Metadata
    desc = _get_hamamatsu_streak_description(tiff, op)
    #Getting the raw axes
    xax = desc['Scaling']['ScalingXaxis']
    yax = desc['Scaling']['ScalingYaxis']

    #Axes are initialised as a list of empty dictionaries
    axes = [{}]*len(names)

    #The width axis is always linear
    [xsc, xof] = np.polyfit(np.arange(len(xax)), xax, 1)

    i = names.index('width')
    axes[i] = {'size': shape[i],
               'name': 'width',
               'units': desc['Scaling']['ScalingXUnit'],
               'scale': xsc,
               'offset': xof}

    #The height axis is changing
    i = names.index('height')
    axes[i] = {'name': 'height',
               'units': desc['Scaling']['ScalingYUnit']}
    if hamamatsu_streak_axis_type == 'uniform':
        #Uniform axis initialisation
        [ysc, yof] = np.polyfit(np.arange(len(yax)), yax, 1)
        axes[i].update({'scale': ysc,
                        'offset': yof,
                        'size': shape[i],
                        })
    elif hamamatsu_streak_axis_type == 'data':
        #Data axis initialisation
        axes[i].update({'axis': yax})
    elif hamamatsu_streak_axis_type == 'functional':
        #Functional axis initialisation
        xaxis = {'scale': 1, 'offset': 0, 'size': len(yax)}
        poly = np.polyfit(np.arange(len(yax)), yax, 3)
        axes[i].update({'size': len(yax), 'x': xaxis,
                   'expression': "a*x**3+b*x**2+c*x+d",
                   'a': poly[0], 'b': poly[1], 'c': poly[2], 'd': poly[3]})

    return axes


def _is_imagej(tiff) -> bool:
    return 'imagej' in tiff.flags


def _add_axes_imagej(tiff, op, scales, offsets, units ):
    imagej_metadata = tiff.imagej_metadata
    if 'ImageJ' in imagej_metadata:
        _logger.debug("Reading ImageJ tif metadata")
        # ImageJ write the unit in the image description
        if 'unit' in imagej_metadata:
            if imagej_metadata['unit'] == 'micron':
                units.update({'x': 'µm', 'y': 'µm'})
            scales['x'], scales['y'] = _get_scales_from_x_y_resolution(op)
        if 'spacing' in imagej_metadata:
            scales['z'] = imagej_metadata['spacing']
    return scales, offsets, units


def _is_digital_micrograph(op) -> bool:
    # for files containing DM metadata
    tags = ['65003', '65004', '65005', '65009', '65010', '65011',
            '65006', '65007', '65008', '65022', '65024', '65025']
    search_result = [tag in op for tag in tags]
    return any(search_result)


def _intensity_axis_digital_micrograph(op, intensity_axis=None):
    if intensity_axis is None:
        intensity_axis = {}
    if '65022' in op:
        intensity_axis['units'] = op['65022']  # intensity units
    if '65024' in op:
        intensity_axis['offset'] = op['65024']  # intensity offset
    if '65025' in op:
        intensity_axis['scale'] = op['65025']  # intensity scale
    return intensity_axis


def _add_axes_digital_micrograph(op, scales, offsets, units):
    if '65003' in op:
        _logger.debug("Reading Gatan DigitalMicrograph tif metadata")
        units['y'] = op['65003']  # x units
    if '65004' in op:
        units['x'] = op['65004']  # y units
    if '65005' in op:
        units['z'] = op['65005']  # z units
    if '65009' in op:
        scales['y'] = op['65009']  # x scales
    if '65010' in op:
        scales['x'] = op['65010']  # y scales
    if '65011' in op:
        scales['z'] = op['65011']  # z scales
    if '65006' in op:
        offsets['y'] = op['65006']  # x offset
    if '65007' in op:
        offsets['x'] = op['65007']  # y offset
    if '65008' in op:
        offsets['z'] = op['65008']  # z offset

    return scales, offsets, units


def _parse_scale_unit(tiff, page, op, shape, force_read_resolution, names, **kwds):
    # Force reading always has priority
    if _is_force_readable(op, force_read_resolution):
        axes = _axes_force_read(op, shape, names)
        return axes
    # Other axes readers can change position if you need to do it
    elif _is_fei(tiff):
        axes = _axes_fei(tiff, op, shape, names)
        return axes
    elif _is_zeiss(tiff):
        axes = _axes_zeiss(tiff, op, shape, names)
        return axes
    elif _is_tvips(tiff):
        axes = _axes_tvips(tiff, op, shape, names)
        return axes
    elif _is_olympus_sis(page):
        axes = _axes_olympus_sis(page, tiff, op, shape, names)
        return axes
    elif _is_jeol_sightx(op):
        axes = _axes_jeol_sightx(tiff, op, shape, names)
        return axes
    elif _is_streak_hamamatsu(op):
        axes = _axes_hamamatsu_streak(tiff, op, shape, names, **kwds)
        return axes
    # Axes are otherwise set to defaults
    else:
        scales, offsets, units = _axes_defaults()
        # Axes descriptors can be additionally parsed from digital micrograph or imagej-style files
        if _is_digital_micrograph(op):
            scales, offsets, units = _add_axes_digital_micrograph(op, scales, offsets, units)
        if _is_imagej(tiff):
            scales, offsets, units = _add_axes_imagej(tiff, op, scales, offsets, units)

        scales, offsets, units = _order_axes_by_name(names, scales, offsets, units)

        axes = _build_axes_dictionaries(shape, names, scales, offsets, units)

        return axes


def _get_scales_from_x_y_resolution(op, factor=1.0):
    scales = (op["YResolution"][1] / op["YResolution"][0] * factor,
              op["XResolution"][1] / op["XResolution"][0] * factor)
    return scales


def _get_tags_dict(signal, extratags=[], factor=int(1E8)):
    """ Get the tags to export the scale and the unit to be used in
        Digital Micrograph and ImageJ.
    """
    scales, units, offsets = _get_scale_unit(signal, encoding=None)
    _logger.debug("{0}".format(units))
    tags_dict = _get_imagej_kwargs(signal, scales, units, factor=factor)
    scales, units, offsets = _get_scale_unit(signal, encoding='latin-1')

    tags_dict["extratags"].extend(
        _get_dm_kwargs_extratag(
            signal,
            scales,
            units,
            offsets))
    tags_dict["extratags"].extend(extratags)
    return tags_dict


def _get_imagej_kwargs(signal, scales, units, factor=int(1E8)):
    resolution = ((factor, int(scales[-1] * factor)),
                  (factor, int(scales[-2] * factor)))
    if len(signal.axes_manager.navigation_axes) == 1:  # For stacks
        spacing = '%s' % scales[0]
    else:
        spacing = None
    description_string = _imagej_description(unit=units[1], spacing=spacing)
    _logger.debug("Description tag: %s" % description_string)
    extratag = [(270, 's', 1, description_string, False)]
    return {"resolution": resolution, "extratags": extratag}


def _get_dm_kwargs_extratag(signal, scales, units, offsets):
    extratags = [(65003, 's', 3, units[-1], False),  # x unit
                 (65004, 's', 3, units[-2], False),  # y unit
                 (65006, 'd', 1, offsets[-1], False),  # x origin
                 (65007, 'd', 1, offsets[-2], False),  # y origin
                 (65009, 'd', 1, float(scales[-1]), False),  # x scale
                 (65010, 'd', 1, float(scales[-2]), False)]  # y scale
    #                 (65012, 's', 3, units[-1], False),  # x unit full name
    #                 (65013, 's', 3, units[-2], False)]  # y unit full name
    #                 (65015, 'i', 1, 1, False), # don't know
    #                 (65016, 'i', 1, 1, False), # don't know
    #                 (65026, 'i', 1, 1, False)] # don't know
    md = signal.metadata
    if md.has_item('Signal.quantity'):
        try:
            intensity_units = md.Signal.quantity
        except BaseException:
            _logger.info("The units of the 'intensity axes' couldn't be"
                         "retrieved, please report the bug.")
            intensity_units = ""
        extratags.extend([(65022, 's', 3, intensity_units, False),
                          (65023, 's', 3, intensity_units, False)])
    if md.has_item('Signal.Noise_properties.Variance_linear_model'):
        try:
            dic = md.Signal.Noise_properties.Variance_linear_model
            intensity_offset = dic.gain_offset
            intensity_scale = dic.gain_factor
        except BaseException:
            _logger.info("The scale or the offset of the 'intensity axes'"
                         "couldn't be retrieved, please report the bug.")
            intensity_offset = 0.0
            intensity_scale = 1.0
        extratags.extend([(65024, 'd', 1, intensity_offset, False),
                          (65025, 'd', 1, intensity_scale, False)])
    if signal.axes_manager.navigation_dimension > 0:
        extratags.extend([(65005, 's', 3, units[0], False),  # z unit
                          (65008, 'd', 1, offsets[0], False),  # z origin
                          (65011, 'd', 1, float(scales[0]), False),  # z scale
                          # (65014, 's', 3, units[0], False), # z unit full name
                          (65017, 'i', 1, 1, False)])
    return extratags


def _get_scale_unit(signal, encoding=None):
    """ Return a list of scales and units, the length of the list is equal to
        the signal dimension. """
    signal_axes = signal.axes_manager._axes
    scales = [signal_axis.scale for signal_axis in signal_axes]
    units = [signal_axis.units for signal_axis in signal_axes]
    offsets = [signal_axis.offset for signal_axis in signal_axes]
    for i, unit in enumerate(units):
        if unit == t.Undefined:
            units[i] = ''
        if encoding is not None:
            units[i] = units[i].encode(encoding)
    return scales, units, offsets


def _imagej_description(version='1.11a', **kwargs):
    """ Return a string that will be used by ImageJ to read the unit when
        appropriate arguments are provided """
    result = ['ImageJ=%s' % version]

    append = []
    if kwargs['spacing'] is None:
        kwargs.pop('spacing')
    for key, value in list(kwargs.items()):
        if value == 'µm':
            value = 'micron'
        if value == 'Å':
            value = 'angstrom'
        append.append(f'{key.lower()}={value}')

    return '\n'.join(result + append + [''])


def _parse_beam_current_FEI(value):
    try:
        return float(value) * 1e9
    except ValueError:
        return None


def _parse_beam_energy_FEI(value):
    try:
        return float(value) * 1e-3
    except ValueError:
        return None


def _parse_working_distance_FEI(value):
    try:
        return float(value) * 1e3
    except ValueError:
        return None


def _parse_tuple_Zeiss(tup):
    value = tup[1]
    try:
        return float(value)
    except ValueError:
        return value


def _parse_tuple_Zeiss_with_units(tup, to_units=None):
    (value, parse_units) = tup[1:]
    if to_units is not None:
        v = value * _ureg(parse_units)
        value = float("%.6e" % v.to(to_units).magnitude)
    return value


def _parse_tvips_time(value):
    # assuming this is the time in second
    return str(timedelta(seconds=int(value)))


def _parse_tvips_date(value):
    # get a number, such as 132122901, no idea, what it is... this is not
    # an excel serial, nor an unix time...
    return None


def _parse_string(value):
    if value == '':
        return None
    return value


mapping_fei = {
    'fei_metadata.Beam.HV':
        ("Acquisition_instrument.SEM.beam_energy", _parse_beam_energy_FEI),
    'fei_metadata.Stage.StageX':
        ("Acquisition_instrument.SEM.Stage.x", None),
    'fei_metadata.Stage.StageY':
        ("Acquisition_instrument.SEM.Stage.y", None),
    'fei_metadata.Stage.StageZ':
        ("Acquisition_instrument.SEM.Stage.z", None),
    'fei_metadata.Stage.StageR':
        ("Acquisition_instrument.SEM.Stage.rotation", None),
    'fei_metadata.Stage.StageT':
        ("Acquisition_instrument.SEM.Stage.tilt", None),
    'fei_metadata.Stage.WorkingDistance':
        ("Acquisition_instrument.SEM.working_distance", _parse_working_distance_FEI),
    'fei_metadata.Scan.Dwelltime':
        ("Acquisition_instrument.SEM.dwell_time", None),
    'fei_metadata.EBeam.BeamCurrent':
        ("Acquisition_instrument.SEM.beam_current", _parse_beam_current_FEI),
    'fei_metadata.System.SystemType':
        ("Acquisition_instrument.SEM.microscope", None),
    'fei_metadata.User.Date':
        ("General.date", lambda x: parser.parse(x).date().isoformat()),
    'fei_metadata.User.Time':
        ("General.time", lambda x: parser.parse(x).time().isoformat()),
    'fei_metadata.User.User':
        ("General.authors", None)
}


def get_jeol_sightx_mapping(op):
    mapping = {
        'ImageDescription.ElectronGun.AccelerationVoltage':
            ("Acquisition_instrument.TEM.beam_energy", lambda x: float(x) * 0.001),  # keV
        'ImageDescription.ElectronGun.BeamCurrent':
            ("Acquisition_instrument.TEM.beam_current", lambda x: float(x) * 0.001),  # nA
        'ImageDescription.Instruments':
            ("Acquisition_instrument.TEM.microscope", None),

        # Gonio Stage
        # depends on sample holder
        #    ("Acquisition_instrument.TEM.Stage.rotation", None),  #deg
        'ImageDescription.GonioStage.StagePosition.TX':
            ("Acquisition_instrument.TEM.Stage.tilt_alpha", None),  # deg
        'ImageDescription.GonioStage.StagePosition.TY':
            ("Acquisition_instrument.TEM.Stage.tilt_beta", None),  # deg
        # ToDo: MX(Motor)+PX(Piezo), MY+PY should be used
        #    'ImageDescription.GonioStage.StagePosition.MX':
        #    ("Acquisition_instrument.TEM.Stage.x", lambda x: float(x)*1E-6), # mm
        #    'ImageDescription.GonioStage.StagePosition.MY':
        #    ("Acquisition_instrument.TEM.Stage.y", lambda x: float(x)*1E-6), # mm
        'ImageDescription.GonioStage.MZ':
            ("Acquisition_instrument.TEM.Stage.z", lambda x: float(x) * 1E-6),  # mm

        #    ("General.notes", None),
        #    ("General.title", None),
        'ImageDescription.Eos.EosMode':
            ("Acquisition_instrument.TEM.acquisition_mode",
             lambda x: "STEM" if x == "eosASID" else "TEM"),

        "ImageDescription.ImageFormingSystem.SelectorValue": None,
    }
    if op["ImageDescription"]["ImageFormingSystem"]["ModeString"] == "DIFF":
        mapping["ImageDescription.ImageFormingSystem.SelectorValue"] = (
            "Acquisition_instrument.TEM.camera_length", None)
    else:  # Mag Mode
        mapping['ImageDescription.ImageFormingSystem.SelectorValue'] = (
            "Acquisition_instrument.TEM.magnification", None)
    return mapping


mapping_cz_sem = {
    'CZ_SEM.ap_actualkv':
        ("Acquisition_instrument.SEM.beam_energy", _parse_tuple_Zeiss),
    'CZ_SEM.ap_mag':
        ("Acquisition_instrument.SEM.magnification", _parse_tuple_Zeiss),
    'CZ_SEM.ap_stage_at_x':
        ("Acquisition_instrument.SEM.Stage.x",
         lambda tup: _parse_tuple_Zeiss_with_units(tup, to_units='mm')),
    'CZ_SEM.ap_stage_at_y':
        ("Acquisition_instrument.SEM.Stage.y",
         lambda tup: _parse_tuple_Zeiss_with_units(tup, to_units='mm')),
    'CZ_SEM.ap_stage_at_z':
        ("Acquisition_instrument.SEM.Stage.z",
         lambda tup: _parse_tuple_Zeiss_with_units(tup, to_units='mm')),
    'CZ_SEM.ap_stage_at_r':
        ("Acquisition_instrument.SEM.Stage.rotation", _parse_tuple_Zeiss),
    'CZ_SEM.ap_stage_at_t':
        ("Acquisition_instrument.SEM.Stage.tilt", _parse_tuple_Zeiss),
    'CZ_SEM.ap_wd':
        ("Acquisition_instrument.SEM.working_distance",
         lambda tup: _parse_tuple_Zeiss_with_units(tup, to_units='mm')),
    'CZ_SEM.dp_dwell_time':
        ("Acquisition_instrument.SEM.dwell_time",
         lambda tup: _parse_tuple_Zeiss_with_units(tup, to_units='s')),
    'CZ_SEM.ap_iprobe':
        ("Acquisition_instrument.SEM.beam_current",
         lambda tup: _parse_tuple_Zeiss_with_units(tup, to_units='nA')),
    'CZ_SEM.dp_detector_type':
        ("Acquisition_instrument.SEM.Detector.detector_type",
         lambda tup: _parse_tuple_Zeiss(tup)),
    'CZ_SEM.sv_serial_number':
        ("Acquisition_instrument.SEM.microscope", _parse_tuple_Zeiss),
    'CZ_SEM.ap_date':
        ("General.date", lambda tup: parser.parse(tup[1]).date().isoformat()),
    'CZ_SEM.ap_time':
        ("General.time", lambda tup: parser.parse(tup[1]).time().isoformat()),
    'CZ_SEM.sv_user_name':
        ("General.authors", _parse_tuple_Zeiss),
}


def get_tvips_mapping(mapped_magnification):
    mapping_tvips = {
        'TVIPS.TemMagnification':
            ("Acquisition_instrument.TEM.%s" % mapped_magnification, None),
        'TVIPS.CameraType':
            ("Acquisition_instrument.TEM.Detector.Camera.name", None),
        'TVIPS.ExposureTime':
            ("Acquisition_instrument.TEM.Detector.Camera.exposure",
             lambda x: float(x) * 1e-3),
        'TVIPS.TemHighTension':
            ("Acquisition_instrument.TEM.beam_energy", lambda x: float(x) * 1e-3),
        'TVIPS.Comment':
            ("General.notes", _parse_string),
        'TVIPS.Date':
            ("General.date", _parse_tvips_date),
        'TVIPS.Time':
            ("General.time", _parse_tvips_time),
        'TVIPS.TemStagePosition':
            ("Acquisition_instrument.TEM.Stage", lambda stage: {
                'x': stage[0] * 1E3,
                'y': stage[1] * 1E3,
                'z': stage[2] * 1E3,
                'tilt_alpha': stage[3],
                'tilt_beta': stage[4]
            }
             )
    }
    return mapping_tvips


mapping_olympus_sis = {
    'Olympus_SIS_metadata.magnification':
        ("Acquisition_instrument.TEM.magnification", None),
    'Olympus_SIS_metadata.cameraname':
        ("Acquisition_instrument.TEM.Detector.Camera.name", None),
}


def get_metadata_mapping(tiff_page, op):
    if tiff_page.is_fei:
        return mapping_fei

    elif tiff_page.is_sem:
        return mapping_cz_sem

    elif tiff_page.is_tvips:
        try:
            if op['TVIPS']['TemMode'] == 3:
                mapped_magnification = 'camera_length'
            else:
                mapped_magnification = 'magnification'
        except KeyError:
            mapped_magnification = 'magnification'
        return get_tvips_mapping(mapped_magnification)
    elif tiff_page.is_sis:
        return mapping_olympus_sis
    elif op.get('Make', None) == "JEOL Ltd.":
        return get_jeol_sightx_mapping(op)
    else:
        return {}
