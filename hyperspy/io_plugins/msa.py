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

from datetime import datetime as dt
import warnings
import locale
import codecs
import os
import logging

import numpy as np
from traits.api import Undefined

from hyperspy.misc.config_dir import os_name
from hyperspy import Release
from hyperspy.misc.utils import DictionaryTreeBrowser

_logger = logging.getLogger(__name__)

# Plugin characteristics
# ----------------------
format_name = 'MSA'
description = ''
full_support = False
file_extensions = ('msa', 'ems', 'mas', 'emsa', 'EMS', 'MAS', 'EMSA', 'MSA')
default_extension = 0

writes = [(1, 0), ]
# ----------------------

# For a description of the EMSA/MSA format, incluiding the meaning of the
# following keywords:
# http://www.amc.anl.gov/ANLSoftwareLibrary/02-MMSLib/XEDS/EMMFF/EMMFF.IBM/Emmff.Total
keywords = {
    # Required parameters
    'FORMAT': {'dtype': str, 'mapped_to': None},
    'VERSION': {'dtype': str, 'mapped_to': None},
    'TITLE': {'dtype': str, 'mapped_to': 'General.title'},
    'DATE': {'dtype': str, 'mapped_to': None},
    'TIME': {'dtype': str, 'mapped_to': None},
    'OWNER': {'dtype': str, 'mapped_to': None},
    'NPOINTS': {'dtype': float, 'mapped_to': None},
    'NCOLUMNS': {'dtype': float, 'mapped_to': None},
    'DATATYPE': {'dtype': str, 'mapped_to': None},
    'XPERCHAN': {'dtype': float, 'mapped_to': None},
    'OFFSET': {'dtype': float, 'mapped_to': None},
    # Optional parameters
    # Signal1D characteristics
    'SIGNALTYPE': {'dtype': str, 'mapped_to':
                   'Signal.signal_type'},
    'XLABEL': {'dtype': str, 'mapped_to': None},
    'YLABEL': {'dtype': str, 'mapped_to': None},
    'XUNITS': {'dtype': str, 'mapped_to': None},
    'YUNITS': {'dtype': str, 'mapped_to': None},
    'CHOFFSET': {'dtype': float, 'mapped_to': None},
    'COMMENT': {'dtype': str, 'mapped_to': None},
    # Microscope
    'BEAMKV': {'dtype': float, 'mapped_to':
               'Acquisition_instrument.TEM.beam_energy'},
    'EMISSION': {'dtype': float, 'mapped_to': None},
    'PROBECUR': {'dtype': float, 'mapped_to':
                 'Acquisition_instrument.TEM.beam_current'},
    'BEAMDIAM': {'dtype': float, 'mapped_to': None},
    'MAGCAM': {'dtype': float, 'mapped_to': None},
    'OPERMODE': {'dtype': str, 'mapped_to': None},
    'CONVANGLE': {'dtype': float, 'mapped_to':
                  'Acquisition_instrument.TEM.convergence_angle'},

    # Specimen
    'THICKNESS': {'dtype': float, 'mapped_to':
                  'Sample.thickness'},
    'XTILTSTGE': {'dtype': float, 'mapped_to':
                  'Acquisition_instrument.TEM.tilt_stage'},
    'YTILTSTGE': {'dtype': float, 'mapped_to': None},
    'XPOSITION': {'dtype': float, 'mapped_to': None},
    'YPOSITION': {'dtype': float, 'mapped_to': None},
    'ZPOSITION': {'dtype': float, 'mapped_to': None},

    # EELS
    # in ms:
    'INTEGTIME': {'dtype': float, 'mapped_to':
                  'Acquisition_instrument.TEM.Detector.EELS.exposure'},
    # in ms:
    'DWELLTIME': {'dtype': float, 'mapped_to':
                  'Acquisition_instrument.TEM.Detector.EELS.dwell_time'},
    'COLLANGLE': {'dtype': float, 'mapped_to':
                  'Acquisition_instrument.TEM.Detector.EELS.collection_angle'},
    'ELSDET': {'dtype': str, 'mapped_to': None},

    # EDS
    'ELEVANGLE': {'dtype': float, 'mapped_to':
                  'Acquisition_instrument.TEM.Detector.EDS.elevation_angle'},
    'AZIMANGLE': {'dtype': float, 'mapped_to':
                  'Acquisition_instrument.TEM.Detector.EDS.azimuth_angle'},
    'SOLIDANGLE': {'dtype': float, 'mapped_to':
                   'Acquisition_instrument.TEM.Detector.EDS.solid_angle'},
    'LIVETIME': {'dtype': float, 'mapped_to':
                 'Acquisition_instrument.TEM.Detector.EDS.live_time'},
    'REALTIME': {'dtype': float, 'mapped_to':
                 'Acquisition_instrument.TEM.Detector.EDS.real_time'},
    'FWHMMNKA': {'dtype': float, 'mapped_to':
                 'Acquisition_instrument.TEM.Detector.EDS.' +
                 'energy_resolution_MnKa'},
    'TBEWIND': {'dtype': float, 'mapped_to': None},
    'TAUWIND': {'dtype': float, 'mapped_to': None},
    'TDEADLYR': {'dtype': float, 'mapped_to': None},
    'TACTLYR': {'dtype': float, 'mapped_to': None},
    'TALWIND': {'dtype': float, 'mapped_to': None},
    'TPYWIND': {'dtype': float, 'mapped_to': None},
    'TBNWIND': {'dtype': float, 'mapped_to': None},
    'TDIWIND': {'dtype': float, 'mapped_to': None},
    'THCWIND': {'dtype': float, 'mapped_to': None},
    'EDSDET': {'dtype': str, 'mapped_to':
               'Acquisition_instrument.TEM.Detector.EDS.EDS_det'},
}


def parse_msa_string(string, filename=None):
    """Parse an EMSA/MSA file content.

    Parameters
    ----------
    string: string or file object
        It must complain with the EMSA/MSA standard.
    filename: string or None
        The filename.

    Returns:
    --------
    file_data_list: list
        The list containts a dictionary that contains the parsed
        information. It can be used to create a `:class:BaseSignal`
        using `:func:hyperspy.io.dict2signal`.

    """
    if not hasattr(string, "readlines"):
        string = string.splitlines()
    parameters = {}
    mapped = DictionaryTreeBrowser({})
    y = []
    # Read the keywords
    data_section = False
    for line in string:
        if data_section is False:
            if line[0] == "#":
                try:
                    key, value = line.split(': ')
                    value = value.strip()
                except ValueError:
                    key = line
                    value = None
                key = key.strip('#').strip()

                if key != 'SPECTRUM':
                    parameters[key] = value
                else:
                    data_section = True
        else:
            # Read the data
            if line[0] != "#" and line.strip():
                if parameters['DATATYPE'] == 'XY':
                    xy = line.replace(',', ' ').strip().split()
                    y.append(float(xy[1]))
                elif parameters['DATATYPE'] == 'Y':
                    data = [
                        float(i) for i in line.replace(',', ' ').strip().split()]
                    y.extend(data)
    # We rewrite the format value to be sure that it complies with the
    # standard, because it will be used by the writer routine
    parameters['FORMAT'] = "EMSA/MAS Spectral Data File"

    # Convert the parameters to the right type and map some
    # TODO: the msa format seems to support specifying the units of some
    # parametes. We should add this feature here
    for parameter, value in parameters.items():
        # Some parameters names can contain the units information
        # e.g. #AZIMANGLE-dg: 90.
        if '-' in parameter:
            clean_par, units = parameter.split('-')
            clean_par, units = clean_par.strip(), units.strip()
        else:
            clean_par, units = parameter, None
        if clean_par in keywords:
            try:
                parameters[parameter] = keywords[clean_par]['dtype'](value)
            except:
                # Normally the offending mispelling is a space in the scientic
                # notation, e.g. 2.0 E-06, so we try to correct for it
                try:
                    parameters[parameter] = keywords[clean_par]['dtype'](
                        value.replace(' ', ''))
                except:
                    _logger.exception(
                        "The %s keyword value, %s could not be converted to "
                        "the right type", parameter, value)

            if keywords[clean_par]['mapped_to'] is not None:
                mapped.set_item(keywords[clean_par]['mapped_to'],
                                parameters[parameter])
                if units is not None:
                    mapped.set_item(keywords[clean_par]['mapped_to'] +
                                    '_units', units)

    # The data parameter needs some extra care
    # It is necessary to change the locale to US english to read the date
    # keyword
    loc = locale.getlocale(locale.LC_TIME)
    # Setting locale can raise an exception because
    # their name depends on library versions, platform etc.
    try:
        if os_name == 'posix':
            locale.setlocale(locale.LC_TIME, ('en_US', 'utf8'))
        elif os_name == 'windows':
            locale.setlocale(locale.LC_TIME, 'english')
        try:
            time = dt.strptime(parameters['TIME'], "%H:%M")
            mapped.set_item('General.time', time.time().isoformat())
        except:
            if 'TIME' in parameters and parameters['TIME']:
                _logger.warn('The time information could not be retrieved')
        try:
            date = dt.strptime(parameters['DATE'], "%d-%b-%Y")
            mapped.set_item('General.date', date.date().isoformat())
        except:
            if 'DATE' in parameters and parameters['DATE']:
                _logger.warn('The date information could not be retrieved')
    except:
        warnings.warn("I couldn't read the date information due to"
                      "an unexpected error. Please report this error to "
                      "the developers")
    locale.setlocale(locale.LC_TIME, loc)  # restore saved locale

    axes = [{
        'size': len(y),
        'index_in_array': 0,
        'name': parameters['XLABEL'] if 'XLABEL' in parameters else '',
        'scale': parameters['XPERCHAN'] if 'XPERCHAN' in parameters else 1,
        'offset': parameters['OFFSET'] if 'OFFSET' in parameters else 0,
        'units': parameters['XUNITS'] if 'XUNITS' in parameters else '',
    }]
    if filename is not None:
        mapped.set_item('General.original_filename',
                        os.path.split(filename)[1])
    mapped.set_item('Signal.record_by', 'spectrum')
    if mapped.has_item('Signal.signal_type'):
        if mapped.Signal.signal_type == 'ELS':
            mapped.Signal.signal_type = 'EELS'
        if mapped.Signal.signal_type in ['EDX', 'XEDS']:
            mapped.Signal.signal_type = 'EDS'
    else:
        # Defaulting to EELS looks reasonable
        mapped.set_item('Signal.signal_type', 'EELS')
    if 'YUNITS' in parameters.keys():
        yunits = "(%s)" % parameters['YUNITS']
    else:
        yunits = ""
    if 'YLABEL' in parameters.keys():
        quantity = "%s" % parameters['YLABEL']
    else:
        if mapped.Signal.signal_type == 'EELS':
            quantity = 'Electrons'
            if not yunits:
                yunits = "(Counts)"
        elif 'EDS' in mapped.Signal.signal_type:
            quantity = 'X-rays'
            if not yunits:
                yunits = "(Counts)"
        else:
            quantity = ""
    if quantity or yunits:
        quantity_units = "%s %s" % (quantity, yunits)
        mapped.set_item('Signal.quantity', quantity_units.strip())

    dictionary = {
        'data': np.array(y),
        'axes': axes,
        'metadata': mapped.as_dictionary(),
        'original_metadata': parameters
    }
    file_data_list = [dictionary, ]
    return file_data_list


def file_reader(filename, encoding='latin-1', **kwds):
    with codecs.open(
            filename,
            encoding=encoding,
            errors='replace') as spectrum_file:
        return parse_msa_string(string=spectrum_file,
                                filename=filename)


def file_writer(filename, signal, format=None, separator=', ',
                encoding='latin-1'):
    loc_kwds = {}
    FORMAT = "EMSA/MAS Spectral Data File"
    md = signal.metadata
    if hasattr(signal.original_metadata, 'FORMAT') and \
            signal.original_metadata.FORMAT == FORMAT:
        loc_kwds = signal.original_metadata.as_dictionary()
        if format is not None:
            loc_kwds['DATATYPE'] = format
        else:
            if 'DATATYPE' in loc_kwds:
                format = loc_kwds['DATATYPE']
    else:
        if format is None:
            format = 'Y'
        if md.has_item("General.date"):
            # Setting locale can raise an exception because
            # their name depends on library versions, platform etc.
            loc = locale.getlocale(locale.LC_TIME)
            if os_name == 'posix':
                locale.setlocale(locale.LC_TIME, ('en_US', 'latin-1'))
            elif os_name == 'windows':
                locale.setlocale(locale.LC_TIME, 'english')
            try:
                date = dt.strptime(md.General.date, "%Y-%m-%d")
                loc_kwds['DATE'] = date.strftime("%d-%b-%Y")
                if md.has_item("General.time"):
                    time = dt.strptime(md.General.time, "%H:%M:%S")
                    loc_kwds['TIME'] = time.strftime("%H:%M")
            except:
                warnings.warn(
                    "I couldn't write the date information due to"
                    "an unexpected error. Please report this error to "
                    "the developers")
            locale.setlocale(locale.LC_TIME, loc)  # restore saved locale
    keys_from_signal = {
        # Required parameters
        'FORMAT': FORMAT,
        'VERSION': '1.0',
        # 'TITLE' : signal.title[:64] if hasattr(signal, "title") else '',
        'DATE': '',
        'TIME': '',
        'OWNER': '',
        'NPOINTS': signal.axes_manager._axes[0].size,
        'NCOLUMNS': 1,
        'DATATYPE': format,
        'SIGNALTYPE': signal.metadata.Signal.signal_type,
        'XPERCHAN': signal.axes_manager._axes[0].scale,
        'OFFSET': signal.axes_manager._axes[0].offset,
        # Signal1D characteristics

        'XLABEL': signal.axes_manager._axes[0].name
        if signal.axes_manager._axes[0].name is not Undefined
        else "",

        #        'YLABEL' : '',
        'XUNITS': signal.axes_manager._axes[0].units
        if signal.axes_manager._axes[0].units is not Undefined
        else "",
        #        'YUNITS' : '',
        'COMMENT': 'File created by HyperSpy version %s' % Release.version,
        # Microscope
        #        'BEAMKV' : ,
        #        'EMISSION' : ,
        #        'PROBECUR' : ,
        #        'BEAMDIAM' : ,
        #        'MAGCAM' : ,
        #        'OPERMODE' : ,
        #        'CONVANGLE' : ,
        # Specimen
        #        'THICKNESS' : ,
        #        'XTILTSTGE' : ,
        #        'YTILTSTGE' : ,
        #        'XPOSITION' : ,
        #        'YPOSITION' : ,
        #        'ZPOSITION' : ,
        #
        # EELS
        # 'INTEGTIME' : , # in ms
        # 'DWELLTIME' : , # in ms
        #        'COLLANGLE' : ,
        #        'ELSDET' :  ,
    }

    # Update the loc_kwds with the information retrieved from the signal class
    for key, value in keys_from_signal.items():
        if key not in loc_kwds or value != '':
            loc_kwds[key] = value

    for key, dic in keywords.items():

        if dic['mapped_to'] is not None:
            if 'SEM' in signal.metadata.Signal.signal_type:
                dic['mapped_to'] = dic['mapped_to'].replace('TEM', 'SEM')
            if signal.metadata.has_item(dic['mapped_to']):
                loc_kwds[key] = eval('signal.metadata.%s' %
                                     dic['mapped_to'])

    with codecs.open(
            filename,
            'w',
            encoding=encoding,
            errors='ignore') as f:
        # Remove the following keys from loc_kwds if they are in
        # (although they shouldn't)
        for key in ['SPECTRUM', 'ENDOFDATA']:
            if key in loc_kwds:
                del(loc_kwds[key])

        f.write('#%-12s: %s\u000D\u000A' % ('FORMAT', loc_kwds.pop('FORMAT')))
        f.write(
            '#%-12s: %s\u000D\u000A' %
            ('VERSION', loc_kwds.pop('VERSION')))
        for keyword, value in loc_kwds.items():
            f.write('#%-12s: %s\u000D\u000A' % (keyword, value))

        f.write('#%-12s: Spectral Data Starts Here\u000D\u000A' % 'SPECTRUM')

        if format == 'XY':
            for x, y in zip(signal.axes_manager._axes[0].axis, signal.data):
                f.write("%g%s%g" % (x, separator, y))
                f.write('\u000D\u000A')
        elif format == 'Y':
            for y in signal.data:
                f.write('%f%s' % (y, separator))
                f.write('\u000D\u000A')
        else:
            raise ValueError('format must be one of: None, \'XY\' or \'Y\'')

        f.write('#%-12s: End Of Data and File' % 'ENDOFDATA')
