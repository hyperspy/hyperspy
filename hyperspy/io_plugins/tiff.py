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
from skimage.external.tifffile import imsave, TiffFile

import numpy as np
import traits.api as t
from hyperspy.misc import rgb_tools
from hyperspy.misc.date_time_tools import get_date_time_from_metadata
from hyperspy.misc.utils import DictionaryTreeBrowser

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
        If the scikit-image version is too old, use the hyperspy embedded
        tifffile library to allow exporting the scale and the unit.
    """
    _logger.debug('************* Saving *************')
    data = signal.data
    if signal.is_rgbx is True:
        data = rgb_tools.rgbx2regular_array(data)
        photometric = "rgb"
    else:
        photometric = "minisblack"
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
    memmap = kwds.pop('memmap', False)
    with TiffFile(filename, **kwds) as tiff:

        # change in the Tifffiles API
        if hasattr(tiff.series[0], 'axes'):
            # in newer version the axes is an attribute
            axes = tiff.series[0].axes
        else:
            # old version
            axes = tiff.series[0]['axes']
        is_rgb = tiff.is_rgb
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
        op = {}
        for key, tag in tiff[0].tags.items():
            op[key] = tag.value
        names = [axes_label_codes[axis] for axis in axes]

        _logger.debug('Tiff tags list: %s' % op)
        _logger.debug("Photometric: %s" % op['photometric'])
        _logger.debug('is_imagej: {}'.format(tiff[0].is_imagej))

        # workaround for 'palette' photometric, keep only 'X' and 'Y' axes
        sl = None
        if op['photometric'] == 3:
            sl = [0] * len(shape)
            names = []
            for i, axis in enumerate(axes):
                if axis == 'X' or axis == 'Y':
                    sl[i] = slice(None)
                    names.append(axes_label_codes[axis])
                else:
                    axes.replace(axis, '')
            shape = tuple(_sh for _s, _sh in zip(sl, shape)
                          if isinstance(_s, slice))
        _logger.debug("names: {0}".format(names))

        scales = [1.0] * len(names)
        offsets = [0.0] * len(names)
        units = [t.Undefined] * len(names)
        intensity_axis = {}
        try:
            scales_d, units_d, offsets_d, intensity_axis, op = \
                _parse_scale_unit(tiff, op, shape,
                                  force_read_resolution)
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
        except:
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

        if 'datetime' in op:
            dt = datetime.strptime(_decode_string(
                op['datetime']), "%Y:%m:%d %H:%M:%S")
            md['General']['date'] = dt.date().isoformat()
            md['General']['time'] = dt.time().isoformat()
        if 'units' in intensity_axis:
            md['Signal']['quantity'] = intensity_axis['units']
        if 'scale' in intensity_axis and 'offset' in intensity_axis:
            dic = {'gain_factor': intensity_axis['scale'],
                   'gain_offset': intensity_axis['offset']}
            md['Signal']['Noise_properties'] = {'Variance_linear_model': dic}

    data_args = TiffFile, filename, is_rgb, sl
    if lazy:
        from dask import delayed
        from dask.array import from_delayed
        memmap = True
        val = delayed(_load_data, pure=True)(*data_args, memmap=memmap, **kwds)
        dc = from_delayed(val, dtype=dtype, shape=shape)
        # TODO: maybe just pass the memmap from tiffile?
    else:
        dc = _load_data(*data_args, memmap=memmap, **kwds)

    metadata = Metadata(op)
    md.update(metadata.get_additional_metadata())

    return [{'data': dc,
             'original_metadata': op,
             'axes': axes,
             'metadata': md,
             'mapping': metadata.mapping,
             }]


def _load_data(TF, filename, is_rgb, sl=None, memmap=False, **kwds):
    with TF(filename, **kwds) as tiff:
        dc = tiff.asarray(memmap=memmap)
        _logger.debug("data shape: {0}".format(dc.shape))
        if is_rgb:
            dc = rgb_tools.regular_array2rgbx(dc)
        if sl is not None:
            dc = dc[sl]
        return dc


def _parse_scale_unit(tiff, op, shape, force_read_resolution):
    axes_l = ['x', 'y', 'z']
    scales = {axis: 1.0 for axis in axes_l}
    offsets = {axis: 0.0 for axis in axes_l}
    units = {axis: t.Undefined for axis in axes_l}
    intensity_axis = {}

    # for files created with DM
    if '65003' in op:
        _logger.debug("Reading Gatan DigitalMicrograph tif metadata")
        units['y'] = _decode_string(op['65003'])  # x units
    if '65004' in op:
        units['x'] = _decode_string(op['65004'])  # y units
    if '65005' in op:
        units['z'] = _decode_string(op['65005'])  # z units
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
        intensity_axis['units'] = _decode_string(
            op['65022'])   # intensity units
    if '65024' in op:
        intensity_axis['offset'] = op['65024']   # intensity offset
    if '65025' in op:
        intensity_axis['scale'] = op['65025']   # intensity scale

    # for files created with imageJ
    if tiff[0].is_imagej:
        image_description = _decode_string(op["image_description"])
        if "image_description_1" in op:
            image_description = _decode_string(op["image_description_1"])
        _logger.debug(
            "Image_description tag: {0}".format(image_description))
        if 'ImageJ' in image_description:
            _logger.debug("Reading ImageJ tif metadata")
            # ImageJ write the unit in the image description
            if 'unit' in image_description:
                unit = image_description.split('unit=')[1].splitlines()[0]
                for key in ['x', 'y']:
                    units[key] = unit
                scales['x'], scales['y'] = _get_scales_from_x_y_resolution(op)
            if 'spacing' in image_description:
                scales['z'] = float(
                    image_description.split('spacing=')[1].splitlines()[0])

    # for FEI Helios tiff files (apparently, works also for Quanta):
    elif 'helios_metadata' in op or 'sfeg_metadata' in op:
        _logger.debug("Reading FEI tif metadata")
        try:
            op['fei_metadata'] = op['helios_metadata']
            del op['helios_metadata']
        except KeyError:
            op['fei_metadata'] = op['sfeg_metadata']
            del op['sfeg_metadata']
        scales['x'] = float(op['fei_metadata']['Scan']['PixelWidth'])
        scales['y'] = float(op['fei_metadata']['Scan']['PixelHeight'])
        for key in ['x', 'y']:
            units[key] = 'm'

    # for Zeiss SEM tiff files:
    elif 'sem_metadata' in op:
        _logger.debug("Reading Zeiss tif metadata")
        if 'ap_pixel_size' in op['sem_metadata']:
            (ps, units0) = op['sem_metadata']['ap_pixel_size'][1:]
            for key in ['x', 'y']:
                scales[key] = ps
                units[key] = units0

    # for TVIPS tiff files:
    elif 'tvips_metadata' in op:
        _logger.debug("Reading TVIPS tif metadata")
        if 'pixel_size_x' in op[
                'tvips_metadata'] and 'pixel_size_y' in op['tvips_metadata']:
            scales['x'] = op['tvips_metadata']['pixel_size_x']
            scales['y'] = op['tvips_metadata']['pixel_size_y']

        else:
            scales['x'], scales['y'] = _get_scales_from_x_y_resolution(
                op, factor=1e-2)
        for key in ['x', 'y']:
            units[key] = 'm'

    if force_read_resolution and 'resolution_unit' in op \
            and 'x_resolution' in op:
        res_unit_tag = op['resolution_unit']
        if res_unit_tag != 1:
            _logger.debug("Resolution unit: %s" % res_unit_tag)
            scales['x'], scales['y'] = _get_scales_from_x_y_resolution(op)
            if res_unit_tag == 2:  # unit is in inch, conversion to um
                for key in ['x', 'y']:
                    units[key] = 'µm'
                    scales[key] = scales[key] * 25400
            if res_unit_tag == 3:  # unit is in cm, conversion to um
                for key in ['x', 'y']:
                    units[key] = 'µm'
                    scales[key] = scales[key] * 10000

    return scales, units, offsets, intensity_axis, op


def _get_scales_from_x_y_resolution(op, factor=1):
    scales = (op["y_resolution"][1] / op["y_resolution"][0] * factor,
              op["x_resolution"][1] / op["x_resolution"][0] * factor)
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
        except:
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
        except:
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


def _decode_string(string):
    try:
        string = string.decode('utf8')
    except:
        # Sometimes the strings are encoded in latin-1 instead of utf8
        string = string.decode('latin-1', errors='ignore')
    return string


class Metadata:

    def __init__(self, original_metadata):
        self.original_metadata = original_metadata
        self.mapping = {}
        self.get_mapping_FEI()
        self.get_mapping_Zeiss()
        if 'tvips_metadata' in self.original_metadata:
            self.get_mapping_TVIPS()

    def get_additional_metadata(self):
        self.md = DictionaryTreeBrowser()
        if 'tvips_metadata' in self.original_metadata:
            self._get_additional_metadata_TVIPS()
        return self.md.as_dictionary()

    def _parse_beam_current_FEI(self, value):
        try:
            return float(value) * 1e9
        except ValueError:
            return None

    def _parse_tuple_Zeiss(self, tup):
        value = tup[1]
        try:
            return float(value)
        except ValueError:
            return value

    def _parse_tuple_Zeiss_with_units(self, tup, to_units=None):
        (value, parse_units) = tup[1:]
        if to_units is not None:
            v = value * ureg(parse_units)
            value = float("%.3e" % v.to(to_units).magnitude)
        return value

    def _parse_tvips_time(self, value):
        # assuming this is the time in second
        return str(timedelta(seconds=int(value)))

    def _parse_tvips_date(self, value):
        # get a number, such as 132122901, no idea, what it is... this is not
        # an excel serial, nor an unix time...
        return None

    def _parse_string(self, value):
        if value == '':
            value = None
        return value

    def get_mapping_FEI(self):
        # mapping FEI metadata
        mapping_FEI = {
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
            ("Acquisition_instrument.SEM.beam_current",
             self._parse_beam_current_FEI),
            'fei_metadata.System.SystemType':
            ("Acquisition_instrument.SEM.microscope", None),
            'fei_metadata.User.Date':
            ("General.date", lambda x: parser.parse(x).date().isoformat()),
            'fei_metadata.User.Time':
            ("General.time", lambda x: parser.parse(x).time().isoformat()),
            'fei_metadata.User.User':
            ("General.authors", None),
        }
        self.mapping.update(mapping_FEI)

    def get_mapping_Zeiss(self):
        # mapping Zeiss metadata
        mapping_Zeiss = {
            'sem_metadata.ap_actualkv':
            ("Acquisition_instrument.SEM.beam_energy", self._parse_tuple_Zeiss),
            'sem_metadata.ap_mag':
            ("Acquisition_instrument.SEM.magnification", self._parse_tuple_Zeiss),
            'sem_metadata.ap_stage_at_x':
            ("Acquisition_instrument.SEM.Stage.x", self._parse_tuple_Zeiss),
            'sem_metadata.ap_stage_at_y':
            ("Acquisition_instrument.SEM.Stage.y", self._parse_tuple_Zeiss),
            'sem_metadata.ap_stage_at_z':
            ("Acquisition_instrument.SEM.Stage.z", self._parse_tuple_Zeiss),
            'sem_metadata.ap_stage_at_r':
            ("Acquisition_instrument.SEM.Stage.rotation", self._parse_tuple_Zeiss),
            'sem_metadata.ap_stage_at_t':
            ("Acquisition_instrument.SEM.Stage.tilt", self._parse_tuple_Zeiss),
            'sem_metadata.ap_free_wd':
            ("Acquisition_instrument.SEM.working_distance",
             lambda tup: self._parse_tuple_Zeiss_with_units(tup, to_units='mm')),
            'sem_metadata.dp_dwell_time':
            ("Acquisition_instrument.SEM.dwell_time",
             lambda tup: self._parse_tuple_Zeiss_with_units(tup, to_units='s')),
            'sem_metadata.ap_beam_current':
            ("Acquisition_instrument.SEM.beam_current",
             lambda tup: self._parse_tuple_Zeiss_with_units(tup, to_units='nA')),
            'sem_metadata.sv_serial_number':
            ("Acquisition_instrument.SEM.microscope", self._parse_tuple_Zeiss),
            # I have not find the corresponding metadata....
            #'sem_metadata.???':
            #("General.date", lambda tup: parser.parse(tup[1]).date().isoformat()),
            #'sem_metadata.???':
            #("General.time", lambda tup: parser.parse(tup[1]).time().isoformat()),
            'sem_metadata.sv_user_name':
            ("General.authors", self._parse_tuple_Zeiss),
        }
        self.mapping.update(mapping_Zeiss)

    def get_mapping_TVIPS(self):
        try:
            if self.original_metadata['tvips_metadata']['tem_mode'] == 3:
                mapped_magnification = 'camera_length'
            else:
                mapped_magnification = 'magnification'
        except KeyError:
            mapped_magnification = 'magnification'

        # mapping TVIPSs metadata
        mapping_TVIPS = {
            'tvips_metadata.tem_magnification':
            ("Acquisition_instrument.TEM.%s" % mapped_magnification, None),
            'tvips_metadata.camera_type':
            ("Acquisition_instrument.TEM.Detector.Camera.name", None),
            'tvips_metadata.exposure_time':
            ("Acquisition_instrument.TEM.Detector.Camera.exposure",
             lambda x: float(x) * 1e-3),
            'tvips_metadata.tem_high_tension':
            ("Acquisition_instrument.TEM.beam_energy", lambda x: float(x) * 1e-3),
            'tvips_metadata.comment':
            ("General.notes", self._parse_string),
            'tvips_metadata.date':
            ("General.date", self._parse_tvips_date),
            'tvips_metadata.time':
            ("General.time", self._parse_tvips_time),
        }
        self.mapping.update(mapping_TVIPS)

    def _get_additional_metadata_TVIPS(self):
        if 'tem_stage_position' in self.original_metadata['tvips_metadata']:
            stage = self.original_metadata[
                'tvips_metadata']['tem_stage_position']
            # Guess on what is x, y, z, tilt_alpha and tilt_beta...
            self.md.set_item(
                "Acquisition_instrument.TEM.Stage.x", stage[0] * 1E3)
            self.md.set_item(
                "Acquisition_instrument.TEM.Stage.y", stage[1] * 1E3)
            self.md.set_item(
                "Acquisition_instrument.TEM.Stage.z", stage[2] * 1E3)
            self.md.set_item(
                "Acquisition_instrument.TEM.Stage.tilt_alpha", stage[3])
            self.md.set_item(
                "Acquisition_instrument.TEM.Stage.tilt_beta", stage[4])
