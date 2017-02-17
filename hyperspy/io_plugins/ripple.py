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

#  for more information on the RPL/RAW format, see
#  http://www.nist.gov/lispix/
#  and
#  http://www.nist.gov/lispix/doc/image-file-formats/raw-file-format.htm

import codecs
import os.path
from io import StringIO
import logging

import numpy as np

from hyperspy.misc.io.utils_readfile import *
from hyperspy import Release
from hyperspy.misc.utils import DictionaryTreeBrowser

_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'Ripple'
description = 'RPL file contains the information on how to read\n'
description += 'the RAW file with the same name.'
description += '\nThis format may not provide information on the calibration.'
description += '\nIf so, you should add that after loading the file.'
full_support = False  # but maybe True
# Recognised file extension
file_extensions = ['rpl', 'RPL']
default_extension = 0
# Writing capabilities
writes = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), ]
# ----------------------

# The format only support the followng data types
newline = ('\n', '\r\n')
comment = ';'
sep = '\t'

dtype2keys = {
    'float64': ('float', 8),
    'float32': ('float', 4),
    'uint8': ('unsigned', 1),
    'uint16': ('unsigned', 2),
    'int32': ('signed', 4),
    'int64': ('signed', 8), }

endianess2rpl = {
    '=': 'dont-care',
    '<': 'little-endian',
    '>': 'big-endian'}

# Warning: for selection lists use tuples not lists.
rpl_keys = {
    # spectrum/image keys
    'width': int,
    'height': int,
    'depth': int,
    'offset': int,
    'data-length': ('1', '2', '4', '8'),
    'data-type': ('signed', 'unsigned', 'float'),
    'byte-order': ('little-endian', 'big-endian', 'dont-care'),
    'record-by': ('image', 'vector', 'dont-care'),
    # X-ray keys
    'ev-per-chan': float,    # usually 5 or 10 eV
    'detector-peak-width-ev': float,  # usually 150 eV
    # HyperSpy-specific keys
    'depth-origin': float,
    'depth-scale': float,
    'depth-units': str,
    'width-origin': float,
    'width-scale': float,
    'width-units': str,
    'height-origin': float,
    'height-scale': float,
    'height-units': str,
    'signal': str,
    # EELS HyperSpy keys
    'collection-angle': float,
    # TEM HyperSpy keys
    'convergence-angle': float,
    'beam-energy': float,
    # EDS HyperSpy keys
    'elevation-angle': float,
    'azimuth-angle': float,
    'live-time': float,
    # From 0.8.5 energy-resolution is deprecated as it is a duplicate of
    # detector-peak-width-ev of the ripple standard format. We keep it here
    # to keep compatibility with rpl file written by HyperSpy < 0.8.4
    'energy-resolution': float,
    'tilt-stage': float,
    'date': str,
    'time': str,
    'title': str,
}


def correct_INCA_format(fp):
    fp_list = list()
    fp.seek(0)
    if '(' in fp.readline():
        for line in fp:
            line = line.replace(
                "(MLX::",
                "").replace(
                " : ",
                "\t").replace(
                " :",
                "\t").replace(
                " ",
                "\t").lower().strip().replace(
                ")",
                "\n")
            if "record-by" in line:
                if "image" in line:
                    line = "record-by\timage"
                if "vector" in line:
                    line = "record-by\tvector"
                if "dont-care" in line:
                    line = "record-by\tdont-care"
            fp_list.append(line)
        fp = StringIO()
        fp.writelines(fp_list)
    fp.seek(0)
    return fp


def parse_ripple(fp):
    """Parse information from ripple (.rpl) file.
    Accepts file object 'fp. Returns dictionary rpl_info.
    """

    fp = correct_INCA_format(fp)

    rpl_info = {}
    for line in fp.readlines():
        # correct_brucker_format
        line = line.replace('data-Length', 'data-length')
        if line[:2] not in newline and line[0] != comment:
            line = line.strip('\r\n')
            if comment in line:
                line = line[:line.find(comment)]
            if sep not in line:
                err = 'Separator in line "%s" is wrong, ' % line
                err += 'it should be a <TAB> ("\\t")'
                raise IOError(err)
            line = line.split(sep)  # now it's a list
            if (line[0] in rpl_keys) is True:
                value_type = rpl_keys[line[0]]
                if isinstance(value_type, tuple):  # is selection list
                    if line[1] not in value_type:
                        err = \
                            'Wrong value for key %s.\n' \
                            'Value read is %s'  \
                            ' but it should be one of %s' % \
                            (line[0], line[1], str(value_type))
                        raise IOError(err)
                else:
                    # rpl_keys[line[0]] must then be a type
                    line[1] = value_type(line[1])

            rpl_info[line[0]] = line[1]

    if rpl_info['depth'] == 1 and rpl_info['record-by'] != 'dont-care':
        err = '"depth" and "record-by" keys mismatch.\n'
        err += '"depth" cannot be "1" if "record-by" is "dont-care" '
        err += 'and vice versa.'
        err += 'Check %s' % fp.name
        raise IOError(err)
    if rpl_info['data-type'] == 'float' and int(rpl_info['data-length']) < 4:
        err = '"data-length" for float "data-type" must be "4" or "8".\n'
        err += 'Check %s' % fp.name
        raise IOError(err)
    if (rpl_info['data-length'] == '1' and
            rpl_info['byte-order'] != 'dont-care'):
        err = '"data-length" and "byte-order" mismatch.\n'
        err += '"data-length" cannot be "1" if "byte-order" is not "dont-care"'
        err += ' and vice versa.'
        err += 'Check %s' % fp.name
        raise IOError(err)
    return rpl_info


def read_raw(rpl_info, fp, mmap_mode='c'):
    """Read the raw file object 'fp' based on the information given in the
    'rpl_info' dictionary.

    Parameters
    ----------
    rpl_info: dict
        A dictionary containing the keywords as parsed by read_rpl
    fp:
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
    If not None, then memory-map the file, using the given mode
    (see `numpy.memmap`).  The mode has no effect for pickled or
    zipped files.
    A memory-mapped array is stored on disk, and not directly loaded
    into memory.  However, it can be accessed and sliced like any
    ndarray.  Memory mapping is especially useful for accessing
    small fragments of large files without reading the entire file
    into memory.


    """
    width = rpl_info['width']
    height = rpl_info['height']
    depth = rpl_info['depth']
    offset = rpl_info['offset']
    data_length = rpl_info['data-length']
    data_type = rpl_info['data-type']
    endian = rpl_info['byte-order']
    record_by = rpl_info['record-by']

    if data_type == 'signed':
        data_type = 'int'
    elif data_type == 'unsigned':
        data_type = 'uint'
    elif data_type == 'float':
        pass
    else:
        raise TypeError('Unknown "data-type" string.')

    if endian == 'big-endian':
        endian = '>'
    elif endian == 'little-endian':
        endian = '<'
    else:
        endian = '='

    data_type += str(int(data_length) * 8)
    data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)

    data = np.memmap(fp,
                     offset=offset,
                     dtype=data_type,
                     mode=mmap_mode)

    if record_by == 'vector':   # spectral image
        size = (height, width, depth)
        data = data.reshape(size)
    elif record_by == 'image':  # stack of images
        size = (depth, height, width)
        data = data.reshape(size)
    elif record_by == 'dont-care':  # stack of images
        size = (height, width)
        data = data.reshape(size)
    return data


def file_reader(filename, rpl_info=None, encoding="latin-1",
                mmap_mode='c', *args, **kwds):
    """Parses a Lispix (http://www.nist.gov/lispix/) ripple (.rpl) file
    and reads the data from the corresponding raw (.raw) file;
    or, read a raw file if the dictionary rpl_info is provided.

    This format is often uses in EDS/EDX experiments.

    Images and spectral images or data cubes that are written in the
    (Lispix) raw file format are just a continuous string of numbers.

    Data cubes can be stored image by image, or spectrum by spectrum.
    Single images are stored row by row, vector cubes are stored row by row
    (each row spectrum by spectrum), image cubes are stored image by image.

    All of the numbers are in the same format, such as 16 bit signed integer,
    IEEE 8-byte real, 8-bit unsigned byte, etc.

    The "raw" file should be accompanied by text file with the same name and
    ".rpl" extension. This file lists the characteristics of the raw file so
    that it can be loaded without human intervention.

    Alternatively, dictionary 'rpl_info' containing the information can
    be given.

    Some keys are specific to HyperSpy and will be ignored by other software.

    RPL stands for "Raw Parameter List", an ASCII text, tab delimited file in
    which HyperSpy reads the image parameters for a raw file.

                    TABLE OF RPL PARAMETERS
        key             type     description
      ----------   ------------ --------------------
      # Mandatory      keys:
      width            int      # pixels per row
      height           int      # number of rows
      depth            int      # number of images or spectral pts
      offset           int      # bytes to skip
      data-type        str      # 'signed', 'unsigned', or 'float'
      data-length      str      # bytes per pixel  '1', '2', '4', or '8'
      byte-order       str      # 'big-endian', 'little-endian', or 'dont-care'
      record-by        str      # 'image', 'vector', or 'dont-care'
      # X-ray keys:
      ev-per-chan      int      # optional, eV per channel
      detector-peak-width-ev  int   # optional, FWHM for the Mn K-alpha line
      # HyperSpy-specific keys
      depth-origin    int      # energy offset in pixels
      depth-scale     float    # energy scaling (units per pixel)
      depth-units     str      # energy units, usually eV
      depth-name      str      # Name of the magnitude stored as depth
      width-origin         int      # column offset in pixels
      width-scale          float    # column scaling (units per pixel)
      width-units          str      # column units, usually nm
      width-name      str           # Name of the magnitude stored as width
      height-origin         int      # row offset in pixels
      height-scale          float    # row scaling (units per pixel)
      height-units          str      # row units, usually nm
      height-name      str           # Name of the magnitude stored as height
      signal            str        # Type of the signal stored, e.g. EDS_SEM
      convergence-angle float   # TEM convergence angle in mrad
      collection-angle  float   # EELS spectrometer collection semi-angle in mrad
      beam-energy       float   # TEM beam energy in keV
      elevation-angle   float   # Elevation angle of the EDS detector
      azimuth-angle     float   # Elevation angle of the EDS detector
      live-time         float   # Live time per spectrum
      energy-resolution float   # Resolution of the EDS (FHWM of MnKa)
      tilt-stage        float   # The tilt of the stage
      date              str     # date in ISO 8601
      time              str     # time in ISO 8601
      title              str    # title of the signal to be stored

    NOTES

    When 'data-length' is 1, the 'byte order' is not relevant as there is only
    one byte per datum, and 'byte-order' should be 'dont-care'.

    When 'depth' is 1, the file has one image, 'record-by' is not relevant and
    should be 'dont-care'. For spectral images, 'record-by' is 'vector'.
    For stacks of images, 'record-by' is 'image'.

    Floating point numbers can be IEEE 4-byte, or IEEE 8-byte. Therefore if
    data-type is float, data-length MUST be 4 or 8.

    The rpl file is read in a case-insensitive manner. However, when providing
    a dictionary as input, the keys MUST be lowercase.

    Comment lines, beginning with a semi-colon ';' are allowed anywhere.

    The first non-comment in the rpl file line MUST have two column names:
    'name_1'<TAB>'name_2'; any name would do e.g. 'key'<TAB>'value'.

    Parameters can be in ANY order.

    In the rpl file, the parameter name is followed by ONE tab (spaces are
    ignored) e.g.: 'data-length'<TAB>'2'

    In the rpl file, other data and more tabs can follow the two items on
    each row, and are ignored.

    Other keys and values can be included and are ignored.

    Any number of spaces can go along with each tab.

    """

    if not rpl_info:
        if filename[-3:] in file_extensions:
            with codecs.open(filename, encoding=encoding,
                             errors='replace') as f:
                rpl_info = parse_ripple(f)
        else:
            raise IOError('File has wrong extension: "%s"' % filename[-3:])
    for ext in ['raw', 'RAW']:
        rawfname = filename[:-3] + ext
        if os.path.exists(rawfname):
            break
        else:
            rawfname = ''
    if not rawfname:
        raise IOError('RAW file "%s" does not exists' % rawfname)
    else:
        lazy = kwds.pop('lazy', False)
        if lazy:
            mmap_mode = 'r'
        data = read_raw(rpl_info, rawfname, mmap_mode=mmap_mode)

    if rpl_info['record-by'] == 'vector':
        _logger.info('Loading as Signal1D')
        record_by = 'spectrum'
    elif rpl_info['record-by'] == 'image':
        _logger.info('Loading as Signal2D')
        record_by = 'image'
    else:
        if len(data.shape) == 1:
            _logger.info('Loading as Signal1D')
            record_by = 'spectrum'
        else:
            _logger.info('Loading as Signal2D')
            record_by = 'image'

    if rpl_info['record-by'] == 'vector':
        idepth, iheight, iwidth = 2, 0, 1
        names = ['height', 'width', 'depth', ]
    else:
        idepth, iheight, iwidth = 0, 1, 2
        names = ['depth', 'height', 'width']

    scales = [1, 1, 1]
    origins = [0, 0, 0]
    units = ['', '', '']
    sizes = [rpl_info[names[i]] for i in range(3)]

    if 'date' not in rpl_info:
        rpl_info['date'] = ""

    if 'time' not in rpl_info:
        rpl_info['time'] = ""

    if 'signal' not in rpl_info:
        rpl_info['signal'] = ""

    if 'title' not in rpl_info:
        rpl_info['title'] = ""

    if 'depth-scale' in rpl_info:
        scales[idepth] = rpl_info['depth-scale']
    # ev-per-chan is the only calibration supported by the original ripple
    # format
    elif 'ev-per-chan' in rpl_info:
        scales[idepth] = rpl_info['ev-per-chan']

    if 'depth-origin' in rpl_info:
        origins[idepth] = rpl_info['depth-origin']

    if 'depth-units' in rpl_info:
        units[idepth] = rpl_info['depth-units']

    if 'depth-name' in rpl_info:
        names[idepth] = rpl_info['depth-name']

    if 'width-origin' in rpl_info:
        origins[iwidth] = rpl_info['width-origin']

    if 'width-scale' in rpl_info:
        scales[iwidth] = rpl_info['width-scale']

    if 'width-units' in rpl_info:
        units[iwidth] = rpl_info['width-units']

    if 'width-name' in rpl_info:
        names[iwidth] = rpl_info['width-name']

    if 'height-origin' in rpl_info:
        origins[iheight] = rpl_info['height-origin']

    if 'height-scale' in rpl_info:
        scales[iheight] = rpl_info['height-scale']

    if 'height-units' in rpl_info:
        units[iheight] = rpl_info['height-units']

    if 'height-name' in rpl_info:
        names[iheight] = rpl_info['height-name']

    mp = DictionaryTreeBrowser({
        'General': {'original_filename': os.path.split(filename)[1],
                    'date': rpl_info['date'],
                    'time': rpl_info['time'],
                    'title': rpl_info['title']
                    },
        "Signal": {'signal_type': rpl_info['signal'],
                   'record_by': record_by},
    })
    if 'convergence-angle' in rpl_info:
        mp.set_item('Acquisition_instrument.TEM.convergence_angle',
                    rpl_info['convergence-angle'])
    if 'tilt-stage' in rpl_info:
        mp.set_item('Acquisition_instrument.TEM.tilt_stage',
                    rpl_info['tilt-stage'])
    if 'collection-angle' in rpl_info:
        mp.set_item('Acquisition_instrument.TEM.Detector.EELS.' +
                    'collection_angle',
                    rpl_info['collection-angle'])
    if 'beam-energy' in rpl_info:
        mp.set_item('Acquisition_instrument.TEM.beam_energy',
                    rpl_info['beam-energy'])
    if 'elevation-angle' in rpl_info:
        mp.set_item('Acquisition_instrument.TEM.Detector.EDS.elevation_angle',
                    rpl_info['elevation-angle'])
    if 'azimuth-angle' in rpl_info:
        mp.set_item('Acquisition_instrument.TEM.Detector.EDS.azimuth_angle',
                    rpl_info['azimuth-angle'])
    if 'energy-resolution' in rpl_info:
        mp.set_item('Acquisition_instrument.TEM.Detector.EDS.' +
                    'energy_resolution_MnKa',
                    rpl_info['energy-resolution'])
    if 'detector-peak-width-ev' in rpl_info:
        mp.set_item('Acquisition_instrument.TEM.Detector.EDS.' +
                    'energy_resolution_MnKa',
                    rpl_info['detector-peak-width-ev'])
    if 'live-time' in rpl_info:
        mp.set_item('Acquisition_instrument.TEM.Detector.EDS.live_time',
                    rpl_info['live-time'])

    axes = []
    index_in_array = 0
    for i in range(3):
        if sizes[i] > 1:
            axes.append({
                'size': sizes[i],
                'index_in_array': index_in_array,
                'name': names[i],
                'scale': scales[i],
                'offset': origins[i],
                'units': units[i],
            })
            index_in_array += 1

    dictionary = {
        'data': data.squeeze(),
        'axes': axes,
        'metadata': mp.as_dictionary(),
        'original_metadata': rpl_info
    }
    return [dictionary, ]


def file_writer(filename, signal, encoding='latin-1', *args, **kwds):

    # Set the optional keys to None
    ev_per_chan = None

    # Check if the dtype is supported
    dc = signal.data
    dtype_name = signal.data.dtype.name
    if dtype_name not in dtype2keys.keys():
        err = 'The ripple format does not support writting data of %s type' % (
            dtype_name)
        raise IOError(err)
    # Check if the dimensions are supported
    dimension = len(signal.data.shape)
    if dimension > 3:
        err = 'This file format does not support %i dimension data' % (
            dimension)
        raise IOError(err)

    # Gather the information to write the rpl
    data_type, data_length = dtype2keys[dc.dtype.name]
    byte_order = endianess2rpl[dc.dtype.byteorder.replace('|', '=')]
    offset = 0
    if signal.metadata.has_item("Signal.signal_type"):
        signal_type = signal.metadata.Signal.signal_type
    else:
        signal_type = ""
    if signal.metadata.has_item("General.date"):
        date = signal.metadata.General.date
    else:
        date = ""
    if signal.metadata.has_item("General.time"):
        time = signal.metadata.General.time
    else:
        time = ""
    if signal.metadata.has_item("General.title"):
        title = signal.metadata.General.title
    else:
        title = ""
    if signal.axes_manager.signal_dimension == 1:
        record_by = 'vector'
        depth_axis = signal.axes_manager.signal_axes[0]
        ev_per_chan = int(round(depth_axis.scale))
        if dimension == 3:
            width_axis = signal.axes_manager.navigation_axes[0]
            height_axis = signal.axes_manager.navigation_axes[1]
            depth, width, height = \
                depth_axis.size, width_axis.size, height_axis.size
        elif dimension == 2:
            width_axis = signal.axes_manager.navigation_axes[0]
            depth, width, height = depth_axis.size, width_axis.size, 1
        elif dimension == 1:
            record_by == 'dont-care'
            depth, width, height = depth_axis.size, 1, 1

    elif signal.axes_manager.signal_dimension == 2:
        width_axis = signal.axes_manager.signal_axes[0]
        height_axis = signal.axes_manager.signal_axes[1]
        if dimension == 3:
            depth_axis = signal.axes_manager.navigation_axes[0]
            record_by = 'image'
            depth, width, height =  \
                depth_axis.size, width_axis.size, height_axis.size
        elif dimension == 2:
            record_by = 'dont-care'
            width, height, depth = width_axis.size, height_axis.size, 1
        elif dimension == 1:
            record_by = 'dont-care'
            depth, width, height = width_axis.size, 1, 1
    else:
        _logger.info("Only Signal1D and Signal2D objects can be saved")
        return

    # Fill the keys dictionary
    keys_dictionary = {
        'width': width,
        'height': height,
        'depth': depth,
        'offset': offset,
        'data-type': data_type,
        'data-length': data_length,
        'byte-order': byte_order,
        'record-by': record_by,
        'signal': signal_type,
        'date': date,
        'time': time,
        'title': title
    }
    if ev_per_chan is not None:
        keys_dictionary['ev-per-chan'] = ev_per_chan
    keys = ['depth', 'height', 'width']
    for key in keys:
        if eval(key) > 1:
            keys_dictionary['%s-scale' % key] = eval(
                '%s_axis.scale' % key)
            keys_dictionary['%s-origin' % key] = eval(
                '%s_axis.offset' % key)
            keys_dictionary['%s-units' % key] = eval(
                '%s_axis.units' % key)
            keys_dictionary['%s-name' % key] = eval(
                '%s_axis.name' % key)
    if signal.metadata.Signal.signal_type == "EELS":
        if "Acquisition_instrument.TEM" in signal.metadata:
            mp = signal.metadata.Acquisition_instrument.TEM
            if mp.has_item('beam_energy'):
                keys_dictionary['beam-energy'] = mp.beam_energy
            if mp.has_item('convergence_angle'):
                keys_dictionary['convergence-angle'] = mp.convergence_angle
            if mp.has_item('Detector.EELS.collection_angle'):
                keys_dictionary[
                    'collection-angle'] = mp.Detector.EELS.collection_angle
    if "EDS" in signal.metadata.Signal.signal_type:
        if signal.metadata.Signal.signal_type == "EDS_SEM":
            mp = signal.metadata.Acquisition_instrument.SEM
        elif signal.metadata.Signal.signal_type == "EDS_TEM":
            mp = signal.metadata.Acquisition_instrument.TEM
        if mp.has_item('beam_energy'):
            keys_dictionary['beam-energy'] = mp.beam_energy
        if mp.has_item('Detector.EDS.elevation_angle'):
            keys_dictionary[
                'elevation-angle'] = mp.Detector.EDS.elevation_angle
        if mp.has_item('tilt_stage'):
            keys_dictionary['tilt-stage'] = mp.tilt_stage
        if mp.has_item('Detector.EDS.azimuth_angle'):
            keys_dictionary['azimuth-angle'] = mp.Detector.EDS.azimuth_angle
        if mp.has_item('Detector.EDS.live_time'):
            keys_dictionary['live-time'] = mp.Detector.EDS.live_time
        if mp.has_item('Detector.EDS.energy_resolution_MnKa'):
            keys_dictionary[
                'detector-peak-width-ev'] = \
                mp.Detector.EDS.energy_resolution_MnKa

    write_rpl(filename, keys_dictionary, encoding)
    write_raw(filename, signal, record_by)


def write_rpl(filename, keys_dictionary, encoding='ascii'):
    f = codecs.open(filename, 'w', encoding=encoding,
                    errors='ignore')
    f.write(';File created by HyperSpy version %s\n' % Release.version)
    f.write('key\tvalue\n')
    # Even if it is not necessary, we sort the keywords when writing
    # to make the rpl file more human friendly
    for key, value in iter(sorted(keys_dictionary.items())):
        if not isinstance(value, str):
            value = str(value)
        f.write(key + '\t' + value + '\n')
    f.close()


def write_raw(filename, signal, record_by):
    """Writes the raw file object

    Parameters:
    -----------
    filename : string
        the filename, either with the extension or without it
    record_by : string
     'vector' or 'image'

        """
    filename = os.path.splitext(filename)[0] + '.raw'
    dshape = signal.data.shape
    data = signal.data
    if len(dshape) == 3:
        if record_by == 'vector':
            np.rollaxis(
                data, signal.axes_manager.signal_axes[0].index_in_array, 3
            ).ravel().tofile(filename)
        elif record_by == 'image':
            data = np.rollaxis(
                data, signal.axes_manager.navigation_axes[0].index_in_array, 0
            ).ravel().tofile(filename)
    elif len(dshape) == 2:
        if record_by == 'vector':
            np.rollaxis(
                data, signal.axes_manager.signal_axes[0].index_in_array, 2
            ).ravel().tofile(filename)
        elif record_by in ('image', 'dont-care'):
            data.ravel().tofile(filename)
    elif len(dshape) == 1:
        data.ravel().tofile(filename)
