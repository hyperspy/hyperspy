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
import warnings
from distutils.version import LooseVersion

import numpy as np
import traits.api as t
from hyperspy.misc import rgb_tools

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


def _import_tifffile_library(import_local_tifffile_if_necessary=False,
                             loading=False):
    def import_local_tifffile(loading=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from hyperspy.external.tifffile import imsave, TiffFile
            if loading:
                # when we don't use skimage tifffile
                warnings.warn(
                    "Loading of some compressed images will be slow.\n")
        return imsave, TiffFile

    try:  # in case skimage is not available, import local tifffile.py
        import skimage
    except ImportError:
        return import_local_tifffile(loading=loading)

    # import local tifffile.py only if the skimage version too old
    skimage_version = LooseVersion(skimage.__version__)
    if import_local_tifffile_if_necessary and skimage_version <= LooseVersion(
            '0.12.3'):
        return import_local_tifffile(loading=loading)
    else:
        from skimage.external.tifffile import imsave, TiffFile
        return imsave, TiffFile


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
    imsave, TiffFile = _import_tifffile_library(export_scale)
    data = signal.data
    if signal.is_rgbx is True:
        data = rgb_tools.rgbx2regular_array(data)
        photometric = "rgb"
    else:
        photometric = "minisblack"
    if 'description' in kwds and export_scale:
        kwds.pop('description')
        # Comment this warning, since it was not passing the test online...
#        warnings.warn(
#            "Description and export scale cannot be used at the same time, "
#            "because of incompability with the 'ImageJ' format")
    if export_scale:
        kwds.update(_get_tags_dict(signal, extratags=extratags))
        _logger.debug("kwargs passed to tifffile.py imsave: {0}".format(kwds))

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
    import_local_tifffile = False
    if 'import_local_tifffile' in kwds.keys():
        import_local_tifffile = kwds.pop('import_local_tifffile')

    imsave, TiffFile = _import_tifffile_library(import_local_tifffile)
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

        _logger.debug('Tiff tags list: %s' % op.keys())
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
        try:
            scales_d, units_d, offsets_d, intensity_axis = \
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

        if 'units' in intensity_axis.keys():
            md['Signal']['quantity'] = intensity_axis['units']
        if 'scale' in intensity_axis.keys() and 'offset' in intensity_axis.keys():
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

    return [{'data': dc,
             'original_metadata': op,
             'axes': axes,
             'metadata': md,
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
    if '65003' in op.keys():
        _logger.debug("Reading Gatan DigitalMicrograph tif metadata")
        units['y'] = _decode_string(op['65003'])  # x units
    if '65004' in op.keys():
        units['x'] = _decode_string(op['65004'])  # y units
    if '65005' in op.keys():
        units['z'] = _decode_string(op['65005'])  # z units
    if '65009' in op.keys():
        scales['y'] = op['65009']   # x scales
    if '65010' in op.keys():
        scales['x'] = op['65010']   # y scales
    if '65011' in op.keys():
        scales['z'] = op['65011']   # z scales
    if '65006' in op.keys():
        offsets['y'] = op['65006']   # x offset
    if '65007' in op.keys():
        offsets['x'] = op['65007']   # y offset
    if '65008' in op.keys():
        offsets['z'] = op['65008']   # z offset
    if '65022' in op.keys():
        intensity_axis['units'] = _decode_string(
            op['65022'])   # intensity units
    if '65024' in op.keys():
        intensity_axis['offset'] = op['65024']   # intensity offset
    if '65025' in op.keys():
        intensity_axis['scale'] = op['65025']   # intensity scale

    # for files created with imageJ
    if tiff[0].is_imagej:
        image_description = _decode_string(op["image_description"])
        if "image_description_1" in op.keys():
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

    # for FEI SEM tiff files:
    elif '34682' in op.keys():
        _logger.debug("Reading FEI tif metadata")
        op = _read_original_metadata_FEI(op)
        scales['x'], scales['y'] = _get_scale_FEI(op)
        for key in ['x', 'y']:
            units[key] = 'm'

    # for Zeiss SEM tiff files:
    elif '34118' in op.keys():
        _logger.debug("Reading Zeiss tif metadata")
        op = _read_original_metadata_Zeiss(op)
        # It seems that Zeiss software doesn't store/compute correctly the
        # scale in the metadata... it needs to be corrected by the image
        # resolution.
        corr = 1024 / max(size for size in shape)
        scales['x'], scales['y'] = _get_scale_Zeiss(op, corr)
        for key in ['x', 'y']:
            units[key] = 'm'

    if force_read_resolution and 'resolution_unit' in op.keys() \
            and 'x_resolution' in op.keys():
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

    return scales, units, offsets, intensity_axis


def _get_scales_from_x_y_resolution(op):
    scales = op["y_resolution"][1] / op["y_resolution"][0], \
        op["x_resolution"][1] / op["x_resolution"][0]
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
                          #                          (65014, 's', 3, units[0], False), # z unit full name
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


def _read_original_metadata_FEI(original_metadata):
    """ information saved in tag '34682' """
    metadata_string = _decode_string(original_metadata['34682'])
    import configparser
    metadata = configparser.ConfigParser(allow_no_value=True)
    metadata.read_string(metadata_string)
    d = {section: dict(metadata.items(section))
         for section in metadata.sections()}
    original_metadata['FEI_metadata'] = d
    return original_metadata


def _get_scale_FEI(original_metadata):
    return float(original_metadata['FEI_metadata']['Scan']['pixelwidth']),\
        float(original_metadata['FEI_metadata']['Scan']['pixelheight'])


def _read_original_metadata_Zeiss(original_metadata):
    """ information saved in tag '34118' """
    metadata_list = _decode_string(original_metadata['34118']).splitlines()
    original_metadata['Zeiss_metadata'] = metadata_list
    return original_metadata


def _get_scale_Zeiss(original_metadata, corr=1.0):
    metadata_list = original_metadata['Zeiss_metadata']
    return float(metadata_list[3]) * corr, float(metadata_list[11]) * corr


def _decode_string(string):
    try:
        string = string.decode('utf8')
    except:
        # Sometimes the strings are encoded in latin-1 instead of utf8
        string = string.decode('latin-1', errors='ignore')
    return string
