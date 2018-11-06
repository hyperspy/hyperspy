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
import logging
from datetime import datetime, timedelta
from dateutil import parser
import pint
from hyperspy.external.tifffile import imsave, TiffFile, TIFF
import traits.api as t
import numpy as np

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
# Writing features
writes = [(2, 0), (2, 1)]
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


ureg = pint.UnitRegistry()


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
    _logger.debug('************* Saving *************')
    data = signal.data
    if signal.is_rgbx is True:
        data = rgb_tools.rgbx2regular_array(data)
        photometric = "RGB"
    else:
        photometric = "MINISBLACK"
    if 'description' in kwds and export_scale:
        kwds.pop('description')
        _logger.warning(
            "Description and export scale cannot be used at the same time, "
            "because it is incompability with the 'ImageJ' tiff format")
    if export_scale:
        kwds.update(_get_tags_dict(signal, extratags=extratags))
        _logger.debug("kwargs passed to tifffile.py imsave: {0}".format(kwds))

    if 'date' in signal.metadata['General']:
        dt = get_date_time_from_metadata(signal.metadata,
                                         formatting='datetime')
        kwds['datetime'] = dt

    imsave(filename, data,
           software="hyperspy",
           photometric=photometric,
           **kwds)


def file_reader(filename, record_by='image', force_read_resolution=False,
                **kwds):
    """
    Read data from tif files using Christoph Gohlke's tifffile library.
    The units and the scale of images saved with ImageJ or Digital
    Micrograph is read. There is limited support for reading the scale of
    files created with Zeiss and FEI SEMs.

    Parameters
    ----------
    filename: str
    record_by: {'image'}
        Has no effect because this format only supports recording by
        image.
    force_read_resolution: Bool
        Default: False.
        Force reading the x_resolution, y_resolution and the resolution_unit
        of the tiff tags.
        See http://www.awaresystems.be/imaging/tiff/tifftags/resolutionunit.html
    **kwds, optional
    """

    _logger.debug('************* Loading *************')
    # For testing the use of local and skimage tifffile library

    lazy = kwds.pop('lazy', False)
    memmap = kwds.pop('memmap', None)
    with TiffFile(filename, **kwds) as tiff:

        # change in the Tifffiles API
        if hasattr(tiff.series[0], 'axes'):
            # in newer version the axes is an attribute
            axes = tiff.series[0].axes
        else:
            # old version
            axes = tiff.series[0]['axes']
        is_rgb = tiff.pages[0].photometric == TIFF.PHOTOMETRIC.RGB
        _logger.debug("Is RGB: %s" % is_rgb)
        series = tiff.series[0]
        if hasattr(series, 'shape'):
            shape = series.shape
            dtype = series.dtype
        else:
            shape = series['shape']
            dtype = series['dtype']
        if is_rgb:
            axes = axes[:-1]
            names = ['R', 'G', 'B', 'A']
            lastshape = shape[-1]
            dtype = np.dtype({'names': names[:lastshape],
                              'formats': [dtype] * lastshape})
            shape = shape[:-1]
        op = {key: tag.value for key, tag in tiff.pages[0].tags.items()}
        names = [axes_label_codes[axis] for axis in axes]

        _logger.debug('Tiff tags list: %s' % op)
        _logger.debug("Photometric: %s" % op['PhotometricInterpretation'])
        _logger.debug('is_imagej: {}'.format(tiff.pages[0].is_imagej))

        scales = [1.0] * len(names)
        offsets = [0.0] * len(names)
        units = [t.Undefined] * len(names)
        intensity_axis = {}
        try:
            scales_d, units_d, offsets_d, intensity_axis = _parse_scale_unit(
                tiff, op, shape, force_read_resolution)
            for i, name in enumerate(names):
                if name == 'height':
                    scales[i], units[i] = scales_d['x'], units_d['x']
                    offsets[i] = offsets_d['x']
                elif name == 'width':
                    scales[i], units[i] = scales_d['y'], units_d['y']
                    offsets[i] = offsets_d['y']
                elif name in ['depth', 'image series', 'time']:
                    scales[i], units[i] = scales_d['z'], units_d['z']
                    offsets[i] = offsets_d['z']
        except BaseException:
            _logger.info("Scale and units could not be imported")

        axes = [{'size': size,
                 'name': str(name),
                 'scale': scale,
                 'offset': offset,
                 'units': unit,
                 }
                for size, name, scale, offset, unit in zip(shape, names,
                                                           scales, offsets,
                                                           units)]

        md = {'General': {'original_filename': os.path.split(filename)[1]},
              'Signal': {'signal_type': "",
                         'record_by': "image",
                         },
              }

        if 'DateTime' in op:
            dt = datetime.strptime(op['DateTime'], "%Y:%m:%d %H:%M:%S")
            md['General']['date'] = dt.date().isoformat()
            md['General']['time'] = dt.time().isoformat()
        if 'units' in intensity_axis:
            md['Signal']['quantity'] = intensity_axis['units']
        if 'scale' in intensity_axis and 'offset' in intensity_axis:
            dic = {'gain_factor': intensity_axis['scale'],
                   'gain_offset': intensity_axis['offset']}
            md['Signal']['Noise_properties'] = {'Variance_linear_model': dic}

    data_args = TiffFile, filename, is_rgb
    if lazy:
        from dask import delayed
        from dask.array import from_delayed
        memmap = 'memmap'
        val = delayed(_load_data, pure=True)(*data_args, memmap=memmap, **kwds)
        dc = from_delayed(val, dtype=dtype, shape=shape)
        # TODO: maybe just pass the memmap from tiffile?
    else:
        dc = _load_data(*data_args, memmap=memmap, **kwds)

    metadata_mapping = get_metadata_mapping(tiff.pages[0], op)

    return [{'data': dc,
             'original_metadata': op,
             'axes': axes,
             'metadata': md,
             'mapping': metadata_mapping,
             }]


def _load_data(TF, filename, is_rgb, sl=None, memmap=None, **kwds):
    with TF(filename, **kwds) as tiff:
        dc = tiff.asarray(out=memmap)
        _logger.debug("data shape: {0}".format(dc.shape))
        if is_rgb:
            dc = rgb_tools.regular_array2rgbx(dc)
        if sl is not None:
            dc = dc[tuple(sl)]
        return dc


def _parse_scale_unit(tiff, op, shape, force_read_resolution):
    axes_l = ['x', 'y', 'z']
    scales = {axis: 1.0 for axis in axes_l}
    offsets = {axis: 0.0 for axis in axes_l}
    units = {axis: t.Undefined for axis in axes_l}
    intensity_axis = {}
    
    if force_read_resolution and 'ResolutionUnit' in op \
            and 'XResolution' in op:
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
    
        return scales, units, offsets, intensity_axis
    
    # for files created with FEI, ZEISS or TVIPS (no DM or ImageJ metadata)
    if 'fei' in tiff.flags:
        _logger.debug("Reading FEI tif metadata")
        op['fei_metadata'] = tiff.fei_metadata
        try:
            del op['FEI_HELIOS']
        except KeyError:
            del op['FEI_SFEG']
        scales['x'] = float(op['fei_metadata']['Scan']['PixelWidth'])
        scales['y'] = float(op['fei_metadata']['Scan']['PixelHeight'])
        units.update({'x': 'm', 'y': 'm'})
        return scales, units, offsets, intensity_axis

    elif 'sem' in tiff.flags:
        _logger.debug("Reading Zeiss tif pixel_scale")
        # op['CZ_SEM'][''] is containing structure of primary
        # not described SEM parameters in SI units.
        # tifffiles returns flattened version of the structure (as tuple)
        # and the scale in it is at index 3.
        # The scale is tied with physical display and needs to be multiplied
        # with factor, which is the 1024 (1k) divide by horizontal pixel n.
        # CZ_SEM tiff can contain reslution of lesser precission
        # in the described tags as 'ap_image_pixel_size' and/or
        # 'ap_pixel_size', which depending from ZEISS software version
        # can be absent and thus is not used here.
        scale_in_m = op['CZ_SEM'][''][3] * 1024 / tiff.pages[0].shape[1]
        scales.update({'x': scale_in_m, 'y': scale_in_m})
        units.update({'x': 'm', 'y': 'm'})
        return scales, units, offsets, intensity_axis

    elif 'tvips' in tiff.flags:
        _logger.debug("Reading TVIPS tif metadata")
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
        return scales, units, offsets, intensity_axis
    
    # for files containing DM metadata
    if '65003' in op:
        _logger.debug("Reading Gatan DigitalMicrograph tif metadata")
        units['y'] = op['65003']  # x units
    if '65004' in op:
        units['x'] = op['65004']  # y units
    if '65005' in op:
        units['z'] = op['65005']  # z units
    if '65009' in op:
        scales['y'] = op['65009']   # x scales
    if '65010' in op:
        scales['x'] = op['65010']   # y scales
    if '65011' in op:
        scales['z'] = op['65011']   # z scales
    if '65006' in op:
        offsets['y'] = op['65006']   # x offset
    if '65007' in op:
        offsets['x'] = op['65007']   # y offset
    if '65008' in op:
        offsets['z'] = op['65008']   # z offset
    if '65022' in op:
        intensity_axis['units'] = op['65022']   # intensity units
    if '65024' in op:
        intensity_axis['offset'] = op['65024']   # intensity offset
    if '65025' in op:
        intensity_axis['scale'] = op['65025']   # intensity scale
    
    # for files containing ImageJ metadata
    if 'imagej' in tiff.flags:
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
    
    return scales, units, offsets, intensity_axis


def _get_scales_from_x_y_resolution(op, factor=1):
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
                          #(65014, 's', 3, units[0], False), # z unit full name
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
        append.append('%s=%s' % (key.lower(), value))

    return '\n'.join(result + append + [''])


def _parse_beam_current_FEI(value):
    try:
        return float(value) * 1e9
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
        v = value * ureg(parse_units)
        value = float("%.3e" % v.to(to_units).magnitude)
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
    ("Acquisition_instrument.SEM.beam_energy", lambda x: float(x) * 1e-3),
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
    ("Acquisition_instrument.SEM.working_distance", lambda x: float(x) * 1e3),
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


mapping_cz_sem = {
    'CZ_SEM.ap_actualkv':
    ("Acquisition_instrument.SEM.beam_energy", _parse_tuple_Zeiss),
    'CZ_SEM.ap_mag':
    ("Acquisition_instrument.SEM.magnification", _parse_tuple_Zeiss),
    'CZ_SEM.ap_stage_at_x':
    ("Acquisition_instrument.SEM.Stage.x", _parse_tuple_Zeiss),
    'CZ_SEM.ap_stage_at_y':
    ("Acquisition_instrument.SEM.Stage.y", _parse_tuple_Zeiss),
    'CZ_SEM.ap_stage_at_z':
    ("Acquisition_instrument.SEM.Stage.z", _parse_tuple_Zeiss),
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
    'CZ_SEM.ap_beam_current':
    ("Acquisition_instrument.SEM.beam_current",
     lambda tup: _parse_tuple_Zeiss_with_units(tup, to_units='nA')),
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
    else:
        return {}
