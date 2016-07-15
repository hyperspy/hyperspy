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
    if import_local_tifffile_if_necessary and skimage_version <= LooseVersion('0.12.3'):
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
        _logger.info("kwargs passed to tifffile.py imsave: {0}".format(kwds))

    imsave(filename, data,
           software="hyperspy",
           photometric=photometric,
           **kwds)


def file_reader(filename, record_by='image', **kwds):
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
    """
    force_read_resolution = False
    if 'force_read_resolution' in kwds.keys():
        force_read_resolution = kwds.pop('force_read_resolution')

    # For testing the use of local and skimage tifffile library
    import_local_tifffile = False
    if 'import_local_tifffile' in kwds.keys():
        import_local_tifffile = kwds.pop('import_local_tifffile')

    imsave, TiffFile = _import_tifffile_library(import_local_tifffile)
    with TiffFile(filename, **kwds) as tiff:
        dc = tiff.asarray()
        # change in the Tifffiles API
        if hasattr(tiff.series[0], 'axes'):
            # in newer version the axes is an attribute
            axes = tiff.series[0].axes
        else:
            # old version
            axes = tiff.series[0]['axes']
        _logger.info("Is RGB: %s" % tiff.is_rgb)
        if tiff.is_rgb:
            dc = rgb_tools.regular_array2rgbx(dc)
            axes = axes[:-1]
        op = {}
        for key, tag in tiff[0].tags.items():
            op[key] = tag.value
        names = [axes_label_codes[axis] for axis in axes]
        units = t.Undefined
        scales = []

        _logger.info('Tiff tags list: %s' % op.keys())
        _logger.info("Photometric: %s" % op['photometric'])

        # for files created with imageJ
        if 'image_description' in op.keys():
            image_description = _decode_string(op["image_description"])
            _logger.info(
                "Image_description tag: {0}".format(image_description))
            if 'ImageJ' in image_description:
                _logger.info("Reading ImageJ tif metadata")
                # ImageJ write the unit in the image description
                units = image_description.split('unit=')[1].split('\n')[0]
                scales = _get_scales_from_x_y_resolution(op)

        # for files created with DM
        if '65003' in op.keys():
            _logger.info("Reading DM tif metadata")
            units = []
            units.extend([_decode_string(op['65003']),  # x unit
                          _decode_string(op['65004'])])  # y unit
            scales = []
            scales.extend([op['65009'],  # x scale
                           op['65010']])  # y scale

        # for FEI SEM tiff files:
        if '34682' in op.keys():
            _logger.info("Reading FEI tif metadata")
            op = _read_original_metadata_FEI(op)
            scales = _get_scale_FEI(op)
            units = 'm'

        # for Zeiss SEM tiff files:
        if '34118' in op.keys():
            _logger.info("Reading Zeiss tif metadata")
            op = _read_original_metadata_Zeiss(op)
            # It seems that Zeiss software doesn't store/compute correctly the
            # scale in the metadata... it needs to be corrected by the image
            # resolution.
            corr = 1024 / max(size for size in dc.shape)
            scales = _get_scale_Zeiss(op, corr)
            units = 'm'

        if force_read_resolution and 'resolution_unit' in op.keys() \
                and 'x_resolution' in op.keys():
            res_unit_tag = op['resolution_unit']
            if res_unit_tag != 1 and len(scales) == 0:
                _logger.info("Resolution unit: %s" % res_unit_tag)
                scales = _get_scales_from_x_y_resolution(op)
                if res_unit_tag == 2:  # unit is in inch, conversion to um
                    scales = [scale * 25400 for scale in scales]
                    units = 'µm'
                if res_unit_tag == 3:  # unit is in cm, conversion to um
                    scales = [scale * 10000 for scale in scales]
                    units = 'µm'

        _logger.info("data shape: {0}".format(dc.shape))

        # workaround for 'palette' photometric, keep only 'X' and 'Y' axes
        if op['photometric'] == 3:
            sl = [0] * dc.ndim
            names = []
            for i, axis in enumerate(axes):
                if axis == 'X' or axis == 'Y':
                    sl[i] = slice(None)
                    names.append(axes_label_codes[axis])
                else:
                    axes.replace(axis, '')
            dc = dc[sl]
        _logger.info("names: {0}".format(names))

        # add the scale for the missing axes when necessary
        for i in dc.shape[len(scales):]:
            if op['photometric'] == 0 or op['photometric'] == 1:
                scales.append(1.0)
            elif op['photometric'] == 2:
                scales.insert(0, 1.0)

        if len(scales) == 0:
            scales = [1.0] * dc.ndim

        if isinstance(units, str) or units == t.Undefined:
            units = [units for i in dc.shape]

        if len(dc.shape) == 3:
            units[0] = t.Undefined

        axes = [{'size': size,
                 'name': str(name),
                 'scale': scale,
                 #'offset' : origins[i],
                 'units': unit,
                 }
                for size, name, scale, unit in zip(dc.shape, names, scales, units)]

    return [{'data': dc,
             'original_metadata': op,
             'axes': axes,
             'metadata': {'General': {'original_filename': os.path.split(filename)[1]},
                          'Signal': {'signal_type': "",
                                     'record_by': "image", },
                          },
             }]


def _get_scales_from_x_y_resolution(op):
    scales = []
    scales.append(op["x_resolution"][1] / op["x_resolution"][0])
    scales.append(op["y_resolution"][1] / op["y_resolution"][0])
    return scales


def _get_tags_dict(signal, extratags=[], factor=int(1E8)):
    """ Get the tags to export the scale and the unit to be used in
        Digital Micrograph and ImageJ.
    """
    scales, units = _get_scale_unit(signal, encoding=None)
    _logger.info("{0}".format(units))
    tags_dict = _get_imagej_kwargs(signal, scales, units, factor=factor)
    scales, units = _get_scale_unit(signal, encoding='latin-1')
    tags_dict["extratags"].extend(
        _get_dm_kwargs_extratag(
            signal,
            scales,
            units))
    tags_dict["extratags"].extend(extratags)
    return tags_dict


def _get_imagej_kwargs(signal, scales, units, factor=int(1E8)):
    resolution = ((factor, int(scales[0] * factor)),
                  (factor, int(scales[1] * factor)))
    description_string = _imagej_description(unit=units[0])
    _logger.info("Description tag: %s" % description_string)
    extratag = [(270, 's', 1, description_string, False)]
    return {"resolution": resolution, "extratags": extratag}


def _get_dm_kwargs_extratag(signal, scales, units):
    extratags = [(65003, 's', 3, units[0], False),  # x unit
                 (65004, 's', 3, units[1], False),  # y unit
                 # (65006, 'd', 1, 0.0, False), # x origin in pixel
                 # (65007, 'd', 1, 0.0, False), # y origin in pixel
                 (65009, 'd', 1, float(scales[0]), False),  # x scale
                 (65010, 'd', 1, float(scales[1]), False),  # y scale
                 (65012, 's', 3, units[0], False),  # x unit
                 (65013, 's', 3, units[1], False)]  # y unit
#                 (65015, 'i', 1, 1, False),
#                 (65016, 'i', 1, 1, False),
#                 (65024, 'd', 1, 0.0, False),
#                 (65025, 'd', 1, 0.0, False),
#                 (65026, 'i', 1, 1, False)]
    if signal.axes_manager.navigation_dimension > 0:
        extratags.extend([(65005, 's', 3, units[2], False),  # z unit
                          (65008, 'd', 1, 3.0, False),  # z origin in pixel
                          (65011, 'd', 1, float(scales[2]), False),  # z scale
                          (65014, 's', 3, units[2], False),  # z unit
                          (65017, 'i', 1, 1, False)])
    return extratags


def _get_scale_unit(signal, encoding=None):
    """ Return a list of scales and units, the length of the list is equal to
        the signal dimension. """
    signal_axes = signal.axes_manager.navigation_axes + \
        signal.axes_manager.signal_axes
    scales = [signal_axis.scale for signal_axis in signal_axes]
    units = [signal_axis.units for signal_axis in signal_axes]
    for i, unit in enumerate(units):
        if unit == t.Undefined:
            units[i] = ''
        if encoding is not None:
            units[i] = units[i].encode(encoding)
    return scales, units


def _imagej_description(version='1.11a', **kwargs):
    """ Return a string that will be used by ImageJ to read the unit when
        appropriate arguments are provided """
    result = ['ImageJ=%s' % version]
    append = []
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
    return [float(original_metadata['FEI_metadata']['Scan']['pixelwidth']),
            float(original_metadata['FEI_metadata']['Scan']['pixelheight'])]


def _read_original_metadata_Zeiss(original_metadata):
    """ information saved in tag '34118' """
    metadata_list = _decode_string(original_metadata['34118']).split('\r\n')
    original_metadata['Zeiss_metadata'] = metadata_list
    return original_metadata


def _get_scale_Zeiss(original_metadata, corr=1.0):
    metadata_list = original_metadata['Zeiss_metadata']
    return [float(metadata_list[3]) * corr, float(metadata_list[11]) * corr]


def _decode_string(string):
    try:
        string = string.decode('utf8')
    except:
        # Sometimes the strings are encoded in latin-1 instead of utf8
        string = string.decode('latin-1', errors='ignore')
    return string
