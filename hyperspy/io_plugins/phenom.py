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
#
#
# Install this as hyperspy/io_plugins/phenom.py
#
# If you get the error "cannot decompress LZW", run
# pip install imagecodecs
#
# Edit hyperspy/io_plugins/__init__.py and add phenom to both the
# list of imports and io_plugins
#
# You should now be able to load Phenom EID .elid files.
#
# This reader supports the ELID file format used in Phenom ProSuite
# Element Identification version 3.8.0 and later. You can convert
# older ELID files by loading the file into a recent Element
# Identification release and then save the ELID file into the newer
# file format.

import bz2
import math
import numpy as np
import copy
import os
import struct
import io
from datetime import datetime
from dateutil import tz
import tifffile
from hyperspy.misc import rgb_tools
import xml.etree.ElementTree as ET

# Plugin characteristics
# ----------------------
format_name = 'Phenom Element Identification (ELID)'
description = 'Read data from Phenom ELID files.'
full_support = False
# Recognised file extension
file_extensions = ('elid', 'ELID')
default_extension = 0
# Reading capabilities
reads_images = False
reads_spectrum = True
reads_spectrum_image = True
# Writing features
writes = False
# ----------------------

def element_symbol(z):
    elements = ['',
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
        'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
        'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
        'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
        'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    if z < 1 or z >= len(elements):
        raise Exception('Invalid atomic number')
    return elements[z]

def family_symbol(i):
    families = ['',	'K', 'L', 'M', 'N', 'O', 'P']
    if i < 1 or i >= len(families):
        raise Exception('Invalid atomic number')
    return families[i]

def IsGZip(pathname):
    with open(pathname, 'rb') as f:
        (magic,) = struct.unpack('2s', f.read(2))
    return magic == b'\x1f\x8b'

def IsBZip2(pathname):
    with open(pathname, 'rb') as f:
        (magic, _, bytes) = struct.unpack('2s2s6s', f.read(10))
    return magic == b'BZ' and bytes == b'\x31\x41\x59\x26\x53\x59'


class ElidReader:

    def __init__(self, pathname, block_size=1024*1024):
        if IsGZip(pathname):
            raise Exception('pre EID 3.8 files are not supported')
        if not IsBZip2(pathname):
            raise Exception('not an ELID file')

        self._pathname = pathname
        self._file = open(pathname, 'rb')
        self._decompressor = bz2.BZ2Decompressor()
        self._block_size = block_size
        (id, version) = struct.unpack('<4si', self._read(8))
        if id != b'EID2':
            raise Exception('not an ELID file')
        if version > 2:
            raise Exception('unsupported ELID format')
        self._version = version
        self.dictionaries = self._read_Project()
        self._file.close()

    def _read(self, size=1):
        data = self._decompressor.decompress(b'', size)
        while self._decompressor.needs_input:
            data += self._decompressor.decompress(self._file.read(self._block_size), size - len(data))
        return data

    def _read_bool(self):
        return struct.unpack('?', self._read(1))[0]

    def _read_uint8(self):
        return struct.unpack('B', self._read(1))[0]

    def _read_int32(self):
        return struct.unpack('<i', self._read(4))[0]

    def _read_uint32(self):
        return struct.unpack('<I', self._read(4))[0]

    def _read_string(self):
        n = self._read_uint32()
        return self._read(n).decode("utf-8")

    def _get_unit_factor(self, unit):
        if len(unit) < 2:
            return 1
        elif unit[0] == 'M':
            return 1e+6
        elif unit[0] == 'k':
            return 1e+3
        elif unit[0] == 'm':
            return 1e-3
        elif unit[0] == 'u' or unit[0] == 'µ':
            return 1e-6
        elif unit[0] == 'n':
            return 1e-9
        elif unit[0] == 'p':
            return 1e-12
        else:
            raise Exception('Unknown unit: ' + unit)

    def _get_value_with_unit(self, item):
        if isinstance(item, dict):
            return float(item['value']) * self._get_unit_factor(item['unit'])
        else:
            return float(item)

    def _read_tiff(self):

        def xml_element_to_dict(element):
            dict = {}
            if len(element) == 0:
                if len(element.items()) > 0:
                    dict[element.tag] = { 'value': element.text }
                    for attrib, value in element.items():
                        dict[element.tag].update({ attrib: value })
                else:
                    dict[element.tag] = element.text
            else:
                dict[element.tag] = {}
                for child in element:
                    dict[element.tag].update(xml_element_to_dict(child))
            return dict

        def make_metadata_dict(xml):
            dict = xml_element_to_dict(ET.fromstring(xml))
            return dict['FeiImage'] if dict else {}

        n = self._read_uint32()
        if n == 0:
            return (None, None)
        bytes = io.BytesIO(self._read(n))
        with tifffile.TiffFile(bytes) as tiff:
            data = tiff.asarray()
            if len(data.shape) > 2:
               data = rgb_tools.regular_array2rgbx(data)
            tags = tiff.pages[0].tags
            if 'FEI_TITAN' in tags:
                metadata = make_metadata_dict(tags['FEI_TITAN'].value)
                metadata['acquisition']['scan']['fieldSize'] = max(self._get_value_with_unit(metadata['pixelHeight']) * data.shape[0], self._get_value_with_unit(metadata['pixelWidth']) * data.shape[1])
            else:
                metadata = {}
        return (metadata, data)

    def _read_int32s(self):
        n = self._read_uint32()
        return [self._read_int32() for _ in range(n)]

    def _read_float64(self):
        return struct.unpack('<d', self._read(8))[0]

    def _read_float64s(self):
        n = self._read_uint32()
        return [self._read_float64() for _ in range(n)]

    def _read_varuint32(self):
        value = 0
        shift = 0
        while True:
            b = self._read_uint8()
            value = value | ((b & 127) << shift)
            if (b & 128) == 0:
                break
            shift += 7
        return value

    def _read_spectrum(self):
        offset = self._read_float64()
        dispersion = self._read_float64()
        n = self._read_uint32()
        return (offset, dispersion, [self._read_varuint32() for _ in range(n)])

    def _read_uint8s(self):
        n = self._read_uint32()
        return [self._read_uint8() for _ in range(n)]

    def _read_oxide(self):
        element = self._read_uint8()
        num_element = self._read_uint8()
        num_oxygen = self._read_uint8()
        oxide = element_symbol(element)
        if num_element > 1:
            oxide += str(num_element)
        oxide += 'O'
        if num_oxygen > 1:
            oxide += str(num_oxygen)
        return oxide

    def _read_oxides(self):
        n = self._read_uint32()
        return [self._read_oxide() for _ in range(n)]

    def _read_element_family(self):
        element = element_symbol(self._read_uint8())
        family = family_symbol(self._read_uint8())
        return (element, family)

    def _read_element_families(self):
        n = self._read_uint32()
        return [self._read_element_family() for _ in range(n)]

    def _read_drift_correction(self):
        dc = self._read_uint8()
        if dc == 1:
            return 'on'
        elif dc == 2:
            return 'off'
        else:
            return 'unknown'

    def _read_eds_metadata(self, om):
        metadata = {}
        metadata['high_tension'] = self._read_float64()
        detector_elevation = self._read_float64()
        if self._version == 0:
            detector_elevation = math.radians(detector_elevation)
        metadata['detector_elevation'] = detector_elevation
        metadata['detector_azimuth'] = self._read_float64()
        metadata['live_time'] = self._read_float64()
        metadata['real_time'] = self._read_float64()
        metadata['slow_peaking_time'] = self._read_float64()
        metadata['fast_peaking_time'] = self._read_float64()
        metadata['detector_resolution'] = self._read_float64()
        metadata['instrument_id'] = self._read_string()
        if self._version == 0 and 'workingDistance' in om:
            metadata['working_distance'] = self._get_value_with_unit(om['workingDistance'])
            metadata['slow_peaking_time'] = 11.2e-6 if float(om['acquisition']['scan']['spotSize']) < 4.5 else 2e-6
            metadata['fast_peaking_time'] = 100e-9;
            metadata['detector_surface_area'] = 25e-6;
        elif self._version > 0:
            metadata['optical_working_distance'] = self._read_float64()
            metadata['working_distance'] = self._read_float64()
            metadata['detector_surface_area'] = self._read_float64()
            metadata['detector_distance'] = self._read_float64()
            metadata['sample_tilt_angle'] = self._read_float64()
        return metadata

    def _read_CommonAnalysis(self, am):
        (metadata, cutout) = self._read_tiff()
        sum_spectrum = self._read_spectrum()
        eds_metadata = self._read_eds_metadata(am)
        eds_metadata['offset'] = sum_spectrum[0]
        eds_metadata['dispersion'] = sum_spectrum[1]
        data = sum_spectrum[2]
        eds_metadata['included_elements'] = [element_symbol(z) for z in self._read_uint8s()]
        eds_metadata['excluded_elements'] = [element_symbol(z) for z in self._read_uint8s()]
        eds_metadata['background_fit_bins'] = self._read_int32s()
        eds_metadata['selected_oxides'] = self._read_oxides()
        eds_metadata['auto_id'] = self._read_bool()
        eds_metadata['order_nr'] = self._read_int32()
        eds_metadata['family_overrides'] = self._read_element_families()
        if self._version >= 2:
            eds_metadata['drift_correction'] = self._read_drift_correction()
        else:
            eds_metadata['drift_correction'] = 'unknown'
        if metadata:
            metadata['acquisition']['scan']['detectors']['EDS'] = eds_metadata
        else:
            metadata = {}
            metadata.update({'acquisition': {'scan': {'detectors': {}}}})
            metadata['acquisition']['scan']['detectors']['EDS'] = eds_metadata
        return (metadata, data)

    def _make_metadata_dict(self, signal_type=None, title="", datetime=None):
        metadata_dict = {
            'General': {
                'original_filename': os.path.split(self._pathname)[1],
                'title': title
            }
        }
        if signal_type:
            metadata_dict['Signal'] = {
                'signal_type': signal_type
                }
        if datetime:
            metadata_dict['General'].update({
                'date': datetime[0],
                'time': datetime[1],
                'time_zone': self._get_local_time_zone()
            })
        return metadata_dict

    def _get_local_time_zone(self):
        return tz.tzlocal().tzname(datetime.today())

    def _make_mapping(self):
        return {
            'acquisition.scan.detectors.EDS.detector_azimuth': ('Acquisition_instrument.SEM.Detector.EDS.azimuth_angle', lambda x: math.degrees(float(x))),
            'acquisition.scan.detectors.EDS.detector_elevation': ('Acquisition_instrument.SEM.Detector.EDS.elevation_angle', lambda x: math.degrees(float(x))),
            'acquisition.scan.detectors.EDS.detector_resolution': ('Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa', lambda x: float(x)),
            'acquisition.scan.detectors.EDS.live_time': ('Acquisition_instrument.SEM.Detector.EDS.live_time', lambda x: float(x)),
            'acquisition.scan.detectors.EDS.real_time': ('Acquisition_instrument.SEM.Detector.EDS.real_time', lambda x: float(x)),
            'acquisition.scan.detectors.EDS.high_tension': ('Acquisition_instrument.SEM.beam_energy', lambda x: float(x) / 1e3),
            'acquisition.scan.highVoltage.value': ('Acquisition_instrument.SEM.beam_energy', lambda x: -float(x)),
            'instrument.uniqueID': ('Acquisition_instrument.SEM.microscope', lambda x: x),
            'samplePosition.x': ('Acquisition_instrument.SEM.Stage.x', lambda x: float(x) / 1e-3),
            'samplePosition.y': ('Acquisition_instrument.SEM.Stage.y', lambda x: float(x) / 1e-3),
            'acquisition.scan.detectors.EDS.sample_tilt_angle': ('Acquisition_instrument.SEM.Stage.tilt_alpha', lambda x: math.degrees(float(x))),
            'acquisition.scan.detectors.EDS.working_distance': ('Acquisition_instrument.SEM.working_distance', lambda x: float(x) / 1e-3)
        }

    def _make_spot_spectrum_dict(self, om, offset, dispersion, data, title):
        axes = [{
            'name': 'Energy',
            'offset': offset / 1e3,
            'scale': dispersion / 1e3,
            'size': len(data),
            'units': 'keV',
            'navigate': False}]
        dict = {
            'data': data,
            'axes': axes,
            'metadata': self._make_metadata_dict('EDS_SEM', title, self._get_datetime(om)),
            'original_metadata': om,
            'mapping': self._make_mapping()
        }
        return dict

    def _get_datetime(self, metadata):
        if 'time' in metadata:
            return metadata['time'].split('T')

    def _make_line_spectrum_dict(self, om, offset, dispersion, data, title):
        axes = [
        {
            'index_in_array': 0,
            'name': 'i',
            'offset': 0,
            'scale': 1,
            'size': data.shape[0],
            'units': 'points',
            'navigate': True
        },
        {
            'index_in_array': 1,
            'name': 'X-ray energy',
            'offset': offset / 1e3,
            'scale': dispersion / 1e3,
            'size': data.shape[1],
            'units': 'keV',
            'navigate': False
        }]
        dict = {
            'data': data,
            'axes': axes,
            'metadata': self._make_metadata_dict('EDS_SEM', title, self._get_datetime(om)),
            'original_metadata': om,
            'mapping': self._make_mapping()
        }
        return dict

    def _get_unit(self, value):
        if value > 1:
            return (1, '')
        elif value > 1e-3:
            return (1e-3, 'm')
        elif value > 1e-6:
            return (1e-6, 'µ')
        elif value > 1e-9:
            return (1e-9, 'n')
        else:
            return (1, '')

    def _make_map_spectrum_dict(self, om, offset, dispersion, data, title):
        size = om['acquisition']['scan']['fieldSize'] * float(om['acquisition']['scan']['scanScale'])
        (scale, prefix) = self._get_unit(size)
        unit = prefix + 'm'
        size = size / scale
        axes = [
        {
            'index_in_array': 0,
            'name': 'y',
            'offset': 0,
            'scale': size / data.shape[0],
            'size': data.shape[0],
            'units': unit,
            'navigate': True
        },
        {
            'index_in_array': 1,
            'name': 'x',
            'offset': 0,
            'scale': size / data.shape[1],
            'size': data.shape[1],
            'units': unit,
            'navigate': True
        },
        {
            'index_in_array': 2,
            'name': 'X-ray energy',
            'offset': offset / 1e3,
            'scale': dispersion / 1e3,
            'size': data.shape[2],
            'units': 'keV',
            'navigate': False
        }]
        dict = {
            'data': data,
            'axes': axes,
            'metadata': self._make_metadata_dict('EDS_SEM', title, self._get_datetime(om)),
            'original_metadata': om,
            'mapping': self._make_mapping()
        }
        return dict

    def _make_image_dict(self, om, data, title):
        if om:
            (scale, prefix) = self._get_unit(0.2 * om['acquisition']['scan']['fieldSize'])
            scale_x = self._get_value_with_unit(om['pixelWidth']) / scale
            scale_y = self._get_value_with_unit(om['pixelHeight']) / scale
            unit = prefix + 'm'
        else:
            scale_x = 1
            scale_y = 1
            unit = 'points'
        axes = [
        {
            'index_in_array': 0,
            'name': 'y',
            'offset': 0,
            'scale': scale_y,
            'size': data.shape[0],
            'units': unit,
            'navigate': True
        },
        {
            'index_in_array': 1,
            'name': 'x',
            'offset': 0,
            'scale': scale_x,
            'size': data.shape[1],
            'units': unit,
            'navigate': True
        }]
        dict = {
            'data': data,
            'axes': axes,
            'metadata': self._make_metadata_dict('', title, self._get_datetime(om)),
            'original_metadata': om,
            'mapping': self._make_mapping()
        }
        return dict

    def _read_MsaAnalysis(self, label, am):
        (om, sum_spectrum) = self._read_CommonAnalysis(am)
        original_metadata = copy.deepcopy(am)
        original_metadata.update(om)
        return self._make_spot_spectrum_dict(original_metadata, om['acquisition']['scan']['detectors']['EDS']['offset'], om['acquisition']['scan']['detectors']['EDS']['dispersion'], np.array(sum_spectrum), '{}, MSA {}'.format(label, om['acquisition']['scan']['detectors']['EDS']['order_nr']))

    def _read_SpotAnalysis(self, label, am):
        (om, sum_spectrum) = self._read_CommonAnalysis(am)
        x = self._read_float64()
        y = self._read_float64()
        original_metadata = copy.deepcopy(am)
        original_metadata['acquisition']['scan']['detectors']['EDS'] = om['acquisition']['scan']['detectors']['EDS']
        original_metadata['acquisition']['scan']['detectors']['EDS']['position'] = {'x': x, 'y': y}
        return self._make_spot_spectrum_dict(original_metadata, om['acquisition']['scan']['detectors']['EDS']['offset'], om['acquisition']['scan']['detectors']['EDS']['dispersion'], np.array(sum_spectrum), '{}, Spot {}'.format(label, om['acquisition']['scan']['detectors']['EDS']['order_nr']))

    def _read_LineScanAnalysis(self, label, am):
        (om, sum_spectrum) = self._read_CommonAnalysis(am)
        x1 = self._read_float64()
        y1 = self._read_float64()
        x2 = self._read_float64()
        y2 = self._read_float64()
        size = self._read_uint32()
        bins = self._read_uint32()
        offset = self._read_float64()
        dispersion = self._read_float64()
        eds_metadata = self._read_eds_metadata(am)
        eds_metadata['live_time'] = om['acquisition']['scan']['detectors']['EDS']['live_time']
        eds_metadata['real_time'] = om['acquisition']['scan']['detectors']['EDS']['real_time']
        eds_metadata['begin'] = { 'x': x1, 'y': y1 }
        eds_metadata['end'] = { 'x': x2, 'y': y2 }
        eds_metadata['offset'] = offset
        eds_metadata['dispersion'] = dispersion
        has_variable_real_time = self._read_bool()
        has_variable_live_time = self._read_bool()
        data = np.empty([size, bins], dtype=np.uint32)
        for i in range(size):
            for bin in range(bins):
                data[i, bin] = self._read_varuint32()
        if has_variable_real_time:
            eds_metadata['real_time_values'] = [self._read_float64() for _ in range(size)]
        else:
            eds_metadata['real_time_values'] = [self._read_float64()] * size
        if has_variable_live_time:
            eds_metadata['live_time_values'] = [self._read_float64() for _ in range(size)]
        else:
            eds_metadata['live_time_values'] = [self._read_float64()] * size
        eds_metadata['high_accuracy_quantification'] = self._read_bool()
        original_metadata = copy.deepcopy(am)
        original_metadata['acquisition']['scan']['detectors']['EDS'] = eds_metadata
        return self._make_line_spectrum_dict(original_metadata, om['acquisition']['scan']['detectors']['EDS']['offset'], om['acquisition']['scan']['detectors']['EDS']['dispersion'], data, '{}, Line {}'.format(label, om['acquisition']['scan']['detectors']['EDS']['order_nr']))

    def _read_MapAnalysis(self, label, am):
        (om, sum_spectrum) = self._read_CommonAnalysis(am)
        left = self._read_float64()
        top = self._read_float64()
        right = self._read_float64()
        bottom = self._read_float64()
        color_intensities = self._read_float64s()
        width = self._read_uint32()
        height = self._read_uint32()
        bins = self._read_uint32()
        offset = self._read_float64()
        dispersion = self._read_float64()
        original_metadata = copy.deepcopy(am)
        eds_metadata = self._read_eds_metadata(am)
        eds_metadata['live_time'] = om['acquisition']['scan']['detectors']['EDS']['live_time']
        eds_metadata['real_time'] = om['acquisition']['scan']['detectors']['EDS']['real_time']
        original_metadata['acquisition']['scan']['detectors']['EDS'] = eds_metadata
        has_variable_real_time = self._read_bool()
        has_variable_live_time = self._read_bool()
        data = np.empty([height, width, bins], dtype=np.uint32)
        for y in range(height):
            for x in range(width):
                for bin in range(bins):
                    data[y, x, bin] = self._read_varuint32()
        if has_variable_real_time:
            real_time_values = np.empty([height, width], dtype=np.float64)
            for y in range(height):
                for x in range(width):
                    real_time_values[y, x] = self._read_float64()
            eds_metadata['real_time_values'] = real_time_values
        else:
            eds_metadata['real_time_values'] = np.full([height, width], self._read_float64())
        if has_variable_live_time:
            live_time_values = np.empty([height, width], dtype=np.float64)
            for y in range(height):
                for x in range(width):
                    live_time_values[y, x] = self._read_float64()
            eds_metadata['live_time_values'] = live_time_values
        else:
            eds_metadata['live_time_values'] = np.full([height, width], self._read_float64())
        return self._make_map_spectrum_dict(original_metadata, om['acquisition']['scan']['detectors']['EDS']['offset'], om['acquisition']['scan']['detectors']['EDS']['dispersion'], data, '{}, Map {}'.format(label, om['acquisition']['scan']['detectors']['EDS']['order_nr']))

    def _read_DifferenceAnalysis(self, label, am):
        (om, sum_spectrum) = self._read_CommonAnalysis(am)
        minuend = self._read_uint32()
        subtrahend = self._read_uint32()
        original_metadata = copy.deepcopy(am)
        original_metadata['acquisition']['scan']['detectors']['EDS'] = om['acquisition']['scan']['detectors']['EDS']
        return self._make_spot_spectrum_dict(original_metadata, om['acquisition']['scan']['detectors']['EDS']['offset'], om['acquisition']['scan']['detectors']['EDS']['dispersion'], np.array(sum_spectrum), '{}, Difference {} - {}'.format(label, minuend, subtrahend))

    def _read_RegionAnalysis(self, label, am):
        (om, sum_spectrum) = self._read_CommonAnalysis(am)
        left = self._read_float64()
        top = self._read_float64()
        right = self._read_float64()
        bottom = self._read_float64()
        original_metadata = copy.deepcopy(am)
        original_metadata['acquisition']['scan']['detectors']['EDS'] = om['acquisition']['scan']['detectors']['EDS']
        original_metadata['acquisition']['scan']['detectors']['EDS']['rectangle'] = {
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom
        }
        return self._make_spot_spectrum_dict(original_metadata, om['acquisition']['scan']['detectors']['EDS']['offset'], om['acquisition']['scan']['detectors']['EDS']['dispersion'], np.array(sum_spectrum), '{}, Region {}'.format(label, om['acquisition']['scan']['detectors']['EDS']['order_nr']))

    def _read_ConstructiveAnalysisSource(self):
        analysis_index = self._read_uint32()
        weight_factor = self._read_float64()
        return (analysis_index, weight_factor)

    def _read_ConstructiveAnalysisSources(self):
        n = self._read_uint32()
        return [self._read_ConstructiveAnalysisSource() for _ in range(n)]

    def _read_ConstructiveAnalysis(self, label, metadata):
        self._read_CommonAnalysis()
        description = self._read_string()
        sources = self._read_ConstructiveAnalysisSources()

    def _read_ConstructiveAnalyses(self):
        n = self._read_uint32()
        return [self._read_ConstructiveAnalysis('', {}) for _ in range(n)]

    def _read_Analysis(self, label, am):
        type = self._read_uint8()
        if type == 1:
            return self._read_MsaAnalysis(label, am)
        elif type == 2:
            return self._read_SpotAnalysis(label, am)
        elif type == 3:
            return self._read_LineScanAnalysis(label, am)
        elif type == 4:
            return self._read_MapAnalysis(label, am)
        elif type == 5:
            return self._read_DifferenceAnalysis(label, am)
        elif type == 6:
            return self._read_RegionAnalysis(label, am)
        elif type == 7:
            return self._read_ConstructiveAnalysis(label, am)
        else:
            raise Exception('Unknown Analysis type')

    def _read_Analyses(self, label, metadata):
        n = self._read_uint32()
        return [self._read_Analysis(label, metadata) for _ in range(n)]

    def _read_Image(self):
        tiff = self._read_tiff()
        label = self._read_string()
        dictionaries = []
        if tiff:
            dictionaries.append(self._make_image_dict(tiff[0], tiff[1], label))
        dictionaries.extend(self._read_Analyses(label, tiff[0]))
        return dictionaries

    def _read_Images(self):
        n = self._read_uint32()
        dictionaries = []
        for _ in range(n):
            dictionaries.extend(self._read_Image())
        return dictionaries

    def _read_Project(self):
        dictionaries = self._read_Images()
        if self._version >= 1:
            self._read_Analyses('', {})
            self._read_ConstructiveAnalyses()
        return [dict for dict in dictionaries if dict]


def file_reader(filename, log_info=False, lazy=False, **kwds):
    reader = ElidReader(filename)
    return reader.dictionaries
