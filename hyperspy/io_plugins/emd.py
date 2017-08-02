# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

# The EMD format is a hdf5 standard proposed at Lawrence Berkeley
# National Lab (see http://emdatasets.com/ for more information).
# NOT to be confused with the FEI EMD format which was developed later.


import re
import h5py
import numpy as np
from dask.array import from_array
import json
import os
from datetime import datetime
from dateutil import tz
import pint

import logging


# Plugin characteristics
# ----------------------
format_name = 'Electron Microscopy Data (EMD)'
description = 'Read data from Berkeleys EMD files.'
full_support = False  # Hopefully?
# Recognised file extension
file_extensions = ('emd', 'EMD')
default_extension = 0
# Writing features
writes = True
EMD_VERSION = '0.2'
# ----------------------

_logger = logging.getLogger(__name__)


def jit_ifnumba(func):
    try:
        import numba
        return numba.jit(func)
    except ImportError:
        _logger.warning('Numba is not installed, reading spectrum image will '
                        'be slow.')
        return func


class EMD(object):

    """Class for storing electron microscopy datasets.

    The :class:`~.EMD` class can hold an arbitrary amount of datasets in the
    `signals` dictionary. These are saved as HyperSpy
    :class:`~hyperspy.signal.Signal` instances. Global metadata are saved in
    four dictionaries (`user`, `microscope`, `sample`, `comments`). To print
    relevant information about the EMD instance use the :func:`~.log_info`
    function. EMD instances can be loaded from and saved to emd-files, an
    hdf5 standard developed at Lawrence
    Berkeley National Lab (https://emdatasets.com/).

    Attributes
    ----------
    signals: dictionary
        Dictionary which contains all datasets as
        :class:`~hyperspy.signal.Signal` instances.
    user : dictionary
        Dictionary which contains user related metadata.
    microscope : dictionary
        Dictionary which contains microscope related metadata.
    sample : dictionary
        Dictionary which contains sample related metadata.
    comments : dictionary
        Dictionary which contains additional commentary metadata.

    """

    _log = logging.getLogger(__name__)

    def __init__(self, signals=None, user=None,
                 microscope=None, sample=None, comments=None):
        self._log.debug('Calling __init__')
        # Create dictionaries if not present:
        if signals is None:
            signals = {}
        if user is None:
            user = {}
        if microscope is None:
            microscope = {}
        if sample is None:
            sample = {}
        if comments is None:
            comments = {}
        # Make sure some default keys are present in user:
        for key in ['name', 'institution', 'department', 'email']:
            if key not in user:
                user[key] = ''
        self.user = user
        # Make sure some default keys are present in microscope:
        for key in ['name', 'voltage']:
            if key not in microscope:
                microscope[key] = ''
        self.microscope = microscope
        # Make sure some default keys are present in sample:
        for key in ['material', 'preparation']:
            if key not in sample:
                sample[key] = ''
        self.sample = sample
        # Add comments:
        self.comments = comments
        # Make sure the signals are added properly to signals:
        self.signals = {}
        for name, signal in signals.items():
            self.add_signal(signal, name)

    def __getitem__(self, key):
        # This is for accessing the raw data easily. For the signals use
        # emd.signals[key]!
        return self.signals[key].data

    def _write_signal_to_group(self, signal_group, signal):
        self._log.debug('Calling _write_signal_to_group')
        # Save data:
        dataset = signal_group.require_group(signal.metadata.General.title)
        maxshape = tuple(None for _ in signal.data.shape)
        dataset.create_dataset(
            'data',
            data=signal.data,
            chunks=True,
            maxshape=maxshape)
        # Iterate over all dimensions:
        for i in range(len(signal.data.shape)):
            key = 'dim{}'.format(i + 1)
            axis = signal.axes_manager._axes[i]
            offset = axis.offset
            scale = axis.scale
            dim = dataset.create_dataset(key, data=[offset, offset + scale])
            name = axis.name
            from traits.trait_base import _Undefined
            if isinstance(name, _Undefined):
                name = ''
            dim.attrs['name'] = name
            units = axis.units
            if isinstance(units, _Undefined):
                units = ''
            else:
                units = '[{}]'.format('_'.join(list(units)))
            dim.attrs['units'] = units
        # Write metadata:
        dataset.attrs['emd_group_type'] = 1
        for key, value in signal.metadata.Signal:
            try:  # If something h5py can't handle is saved in the metadata...
                dataset.attrs[key] = value
            except Exception:  # ...let the user know what could not be added!
                self._log.exception(
                    'The hdf5 writer could not write the following '
                    'information in the file: %s : %s', key, value)

    def _read_signal_from_group(self, name, group, lazy=False):
        self._log.debug('Calling _read_signal_from_group')
        from hyperspy import signals
        # Extract essential data:
        data = group.get('data')
        if lazy:
            data = from_array(data, chunks=data.chunks)
        else:
            data = np.asanyarray(data)
        # EMD does not have a standard way to describe the signal axis.
        # Therefore we return a BaseSignal
        signal = signals.BaseSignal(data)
        # Set signal properties:
        signal.set_signal_origin = group.attrs.get('signal_origin', '')
        signal.set_signal_type = group.attrs.get('signal_type', '')
        # Iterate over all dimensions:
        for i in range(len(data.shape)):
            dim = group.get('dim{}'.format(i + 1))
            axis = signal.axes_manager._axes[i]
            axis_name = dim.attrs.get('name', '')
            if isinstance(axis_name, bytes):
                axis_name = axis_name.decode('utf-8')
            axis.name = axis_name

            axis_units = dim.attrs.get('units', '')
            if isinstance(axis_units, bytes):
                axis_units = axis_units.decode('utf-8')
            units = re.findall('[^_\W]+', axis_units)
            axis.units = ''.join(units)
            try:
                if len(dim) == 1:
                    axis.scale = 1.
                    self._log.warning(
                        'Could not calculate scale of axis {}. '
                        'Setting scale to 1'.format(i))
                else:
                    axis.scale = dim[1] - dim[0]
                axis.offset = dim[0]
            # HyperSpy then uses defaults (1.0 and 0.0)!
            except (IndexError, TypeError) as e:
                self._log.warning(
                    'Could not calculate scale/offset of '
                    'axis {}: {}'.format(i, e))
        # Extract metadata:
        metadata = {}
        for key, value in group.attrs.items():
            metadata[key] = value
        if signal.data.dtype == np.object:
            self._log.warning('HyperSpy could not load the data in {}, '
                              'skipping it'.format(name))
        else:
            # Add signal:
            self.add_signal(signal, name, metadata)

    def add_signal(self, signal, name=None, metadata=None):
        """Add a HyperSpy signal to the EMD instance and make sure all
        metadata is present.

        Parameters
        ----------
        signal : :class:`~hyperspy.signal.Signal`
            HyperSpy signal which should be added to the EMD instance.
        name : string, optional
            Name of the (used as a key for the `signals` dictionary). If not
            specified, `signal.metadata.General.title` will be used. If this
            is an empty string, both name and signal title are set to 'dataset'
            per default. If specified, `name` overwrites the
            signal title.
        metadata : dictionary
            Dictionary which holds signal specific metadata which will
            be added to the signal.

        Returns
        -------
        None

        Notes
        -----
        This is the preferred way to add signals to the EMD instance.
        Directly adding to the `signals` dictionary is possible but does not
        make sure all metadata are correct. This method is also called in
        the standard constructor on all entries in the `signals` dictionary!

        """
        self._log.debug('Calling add_signal')
        # Create metadata if not present:
        if metadata is None:
            metadata = {}
        # Check and save title:
        if name is not None:  # Overwrite Signal title!
            signal.metadata.General.title = name
        else:
            # Take title of Signal!
            if signal.metadata.General.title is not '':
                name = signal.metadata.General.title
            else:  # Take default!
                name = '__unnamed__'
                signal.metadata.General.title = name
        # Save signal metadata:
        signal.metadata.Signal.add_dictionary(metadata)
        # Save global metadata:
        signal.metadata.General.add_node('user')
        signal.metadata.General.user.add_dictionary(self.user)
        signal.metadata.General.add_node('microscope')
        signal.metadata.General.microscope.add_dictionary(self.microscope)
        signal.metadata.General.add_node('sample')
        signal.metadata.General.sample.add_dictionary(self.sample)
        signal.metadata.General.add_node('comments')
        signal.metadata.General.comments.add_dictionary(self.comments)
        # Also save metadata as original_metadata:
        signal.original_metadata.add_dictionary(
            signal.metadata.as_dictionary())
        # Add signal:
        self.signals[name] = signal

    @classmethod
    def load_from_emd(cls, filename, lazy=False):
        """Construct :class:`~.EMD` object from an emd-file.

        Parameters
        ----------
        filename : string
            The name of the emd-file from which to load the signals. Standard
            format is '*.emd'.
        False : bool, optional
            If False (default) loads data to memory. If True, enables loading
            only if requested.

        Returns
        -------
        emd : :class:`~.EMD`
            A :class:`~.EMD` object containing the loaded signals.

        """
        cls._log.debug('Calling load_from_emd')
        # Read in file:
        emd_file = h5py.File(filename, 'r')
        # Creat empty EMD instance:
        emd = cls()
        # Extract user:
        user_group = emd_file.get('user')
        if user_group is not None:
            for key, value in user_group.attrs.items():
                emd.user[key] = value
        # Extract microscope:
        microscope_group = emd_file.get('microscope')
        if microscope_group is not None:
            for key, value in microscope_group.attrs.items():
                emd.microscope[key] = value
        # Extract sample:
        sample_group = emd_file.get('sample')
        if sample_group is not None:
            for key, value in sample_group.attrs.items():
                emd.sample[key] = value
        # Extract comments:
        comments_group = emd_file.get('comments')
        if comments_group is not None:
            for key, value in comments_group.attrs.items():
                emd.comments[key] = value
        # Extract signals:
        node_list = list(emd_file.keys())
        for key in ['user', 'microscope',
                    'sample', 'comments']:  # Nodes which are not the data!
            if key in node_list:
                node_list.pop(node_list.index(key))  # Pop all unwanted nodes!
        # One node should remain, the data node (named 'data', 'signals',
        # 'experiments', ...)!
        assert len(node_list) == 1, 'Dataset location is ambiguous!'
        data_group = emd_file.get(node_list[0])
        if data_group is not None:
            for name, group in data_group.items():
                if isinstance(group, h5py.Group):
                    if group.attrs.get('emd_group_type') == 1:
                        emd._read_signal_from_group(
                            name, group, lazy)
        # Close file and return EMD object:
        if not lazy:
            emd_file.close()
        return emd

    def save_to_emd(self, filename='datacollection.emd'):
        """Save :class:`~.EMD` data in a file with emd(hdf5)-format.

        Parameters
        ----------
        filename : string, optional
            The name of the emd-file in which to store the signals.
            The default is 'datacollection.emd'.

        Returns
        -------
        None

        """
        self._log.debug('Calling save_to_emd')
        # Open file:
        emd_file = h5py.File(filename, 'w')
        # Write version:
        ver_maj, ver_min = EMD_VERSION.split('.')
        emd_file.attrs['version_major'] = ver_maj
        emd_file.attrs['version_minor'] = ver_min
        # Write user:
        user_group = emd_file.require_group('user')
        for key, value in self.user.items():
            user_group.attrs[key] = value
        # Write microscope:
        microscope_group = emd_file.require_group('microscope')
        for key, value in self.microscope.items():
            microscope_group.attrs[key] = value
        # Write sample:
        sample_group = emd_file.require_group('sample')
        for key, value in self.sample.items():
            sample_group.attrs[key] = value
        # Write comments:
        comments_group = emd_file.require_group('comments')
        for key, value in self.comments.items():
            comments_group.attrs[key] = value
        # Write signals:
        signal_group = emd_file.require_group('signals')
        for signal in self.signals.values():
            self._write_signal_to_group(signal_group, signal)
        # Close file and return EMD object:
        emd_file.close()

    def log_info(self):
        """Print all relevant information about the EMD instance."""
        self._log.debug('Calling log_info')
        pad_string0 = '-------------------------\n'
        pad_string1 = '\n-------------------------\n'
        info_str = '\nUser:' + pad_string1
        for key, value in self.user.items():
            info_str += '{:<15}: {}\n'.format(key, value)
        info_str += pad_string0 + '\nMicroscope:' + pad_string1
        for key, value in self.microscope.items():
            info_str += '{:<15}: {}\n'.format(key, value)
        info_str += pad_string0 + '\nSample:' + pad_string1
        for key, value in self.sample.items():
            info_str += '{:<15}: {}\n'.format(key, value)
        info_str += pad_string0 + '\nComments:' + pad_string1
        for key, value in self.comments.items():
            info_str += '{:<15}: {}\n'.format(key, value)
        info_str += pad_string0 + '\nData:' + pad_string1
        for key, value in self.signals.items():
            info_str += '{:<15}: {}\n'.format(key, value)
            sig_dict = value.metadata.Signal
            for k in sig_dict.keys():
                info_str += '  |-- {}: {}\n'.format(k, sig_dict[k])
        info_str += pad_string0
        self._log.info(info_str)


def fei_check(filename):
    """Function to check if the EMD file is an FEI file.

    Parameters
    ----------
    filename : string
        The name of the emd-file from which to load the signals. Standard
        format is '*.emd'.

    Returns
    -------
    Boolean

    """
    with h5py.File(filename, 'r') as f:
        if 'Version' in list(f.keys()):
            version = f.get('Version')
            v_dict = json.loads(version.value[0].decode('utf-8'))
            if v_dict['format'] == 'Velox':
                return True


def _get_keys_from_group(group):
    # Return a list of ids of items contains in the group
    return list(group.keys())


class FeiEMDReader(object):
    """
    Class for reading FEI electron microscopy datasets.

    The :class:`~.FeiEMDReader` reads EMD files saved by the FEI Velox 
    software package.

    Attributes
    ----------
    dictionaries: list
        List of dictionaries which are passed to the file_reader.
    im_type : string
        String specifying whether the data is an image, spectrum or 
        spectrum image.

    Methods
    ----------
    get_metadata_dict, get_original_metadata

    """

    def __init__(self, filename, first_frame=0, last_frame=None,
                 individual_frame=False, energy_rebin=1, SI_dtype=None,
                 read_SI_image_stack=False):
        self.filename = filename
        self.ureg = pint.UnitRegistry()
        self.dictionaries = []
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.individual_frame = individual_frame
        self.energy_rebin = energy_rebin
        self.SI_data_dtype = SI_dtype
        self.read_SI_image_stack = read_SI_image_stack
        with h5py.File(filename, 'r') as f:
            self.d_grp = f.get('Data')
            self._check_im_type()
            self._read_original_metadata(f)
            self._read_data()

    def _read_data(self):
        if self.im_type == 'Image':
            _logger.info('Reading the images')
            self._read_images()
        elif self.im_type == 'Spectrum':
            self._read_spectrum()
        elif self.im_type == 'SpectrumStream':
            _logger.info('Reading the spectrum image')
            self._read_images()
            self._read_spectrum_image()

    def _check_im_type(self):
        if 'Image' in self.d_grp:
            if 'SpectrumImage' in self.d_grp:
                self.im_type = 'SpectrumStream'
            else:
                self.im_type = 'Image'
        else:
            self.im_type = 'Spectrum'

    def _read_spectrum(self):
        self.record_by = 'spectrum'
        spec_grp = self.d_grp.get("Spectrum")
        data_grp = spec_grp[list(spec_grp.keys())[0]]
        dataset = data_grp['Data']
        data = dataset[:, 0]

        dispersion, offset = self._get_dispersion_offset()

        axes = [{'index_in_array': 0,
                 'name': 'E',
                 'offset': offset,
                 'scale': dispersion,
                 'size': data.shape[0],
                 'units': 'keV'}
                ]

        md = self._get_metadata_dict()
        md['Signal']['signal_type'] = 'EDS_TEM'

        self.dictionaries.append({'data': data,
                                  'axes': axes,
                                  'metadata': md,
                                  'original_metadata': self.original_metadata,
                                  'mapping': self._get_mapping()})

    def _read_images(self):
        self.record_by = 'image'
        # Get the image data group
        image_group = self.d_grp.get("Image")
        # Get all the subgroup of the image data group and read the image for
        # each of them
        for data_sub_group in _get_keys_from_group(image_group):
            self.dictionaries.append(
                self._read_image(image_group[data_sub_group]))

    def _read_image(self, data_sub_group):
        """ Return a dictionary ready to parse of return to io module"""

        read_stack = (self.read_SI_image_stack or self.im_type == 'Image')
        if read_stack:
            data = np.rollaxis(np.array(data_sub_group['Data']), axis=2)
            # Get the scanning area shape of the SI from the images
            self.SI_shape = data.shape[1:]
        else:
            data = data_sub_group['Data'][:, :, 0]
            # Get the scanning area shape of the SI from the images
            self.SI_shape = data.shape

        pix_scale = self.original_metadata['BinaryResult']['PixelSize']
        offsets = self.original_metadata['BinaryResult']['Offset']
        original_units = self.original_metadata['BinaryResult']['PixelUnitX']

        axes = []
        # stack of images
        if read_stack and data.shape[0] > 1:
            frame_time = self.original_metadata['Scan']['FrameTime']
            scale_time = self._convert_scale_units(
                frame_time, 's', 2 * data.shape[0])
            axes.append({'index_in_array': 0,
                         'name': 'Time',
                         'offset': 0,
                         'scale': scale_time[0],
                         'size': data.shape[0],
                         'units': scale_time[1]})
            i = 1
        else:
            if read_stack:
                # since there is only one image, need to remove the first axis
                data = data.squeeze(axis=0)
            i = 0
        scale_x = self._convert_scale_units(
            pix_scale['width'], original_units, data.shape[i + 1])
        scale_y = self._convert_scale_units(
            pix_scale['height'], original_units, data.shape[i])
        offset_x = self._convert_scale_units(
            offsets['x'], original_units, data.shape[i + 1])
        offset_y = self._convert_scale_units(
            offsets['y'], original_units, data.shape[i])
        axes.extend([{'index_in_array': i,
                      'name': 'y',
                      'offset': offset_y[0],
                      'scale': scale_y[0],
                      'size': data.shape[i],
                      'units': scale_y[1]},
                     {'index_in_array': i + 1,
                      'name': 'x',
                      'offset': offset_x[0],
                      'scale': scale_x[0],
                      'size': data.shape[i + 1],
                      'units': scale_x[1]}
                     ])

        md = self._get_metadata_dict()
        md['Signal']['signal_type'] = 'image'

        return {'data': data,
                'axes': axes,
                'metadata': md,
                'original_metadata': self.original_metadata,
                'mapping': self._get_mapping()}

    def _read_spectrum_image(self):
        self.record_by = 'spectrum'
        # Spectrum stream group
        si_grp = self.d_grp.get("SpectrumStream")

        # Read spectrum stream
        stream = FeiSpectrumStream(si_grp, self.energy_rebin,
                                   data_dtype=self.SI_data_dtype)

        self.si_grp = si_grp

        stream.get_spectrum_image(self.SI_shape, self.first_frame,
                                  self.last_frame, self.individual_frame)
        self.original_metadata['AcquisitionSettings'] = stream.acquisition_settings
        self.original_metadata['ImportedDataParameter'] = {'First_frame': stream.first_frame,
                                                           'Last_frame': stream.last_frame,
                                                           'Frame_number': stream.frame_number}

        pix_scale = self.original_metadata['BinaryResult']['PixelSize']
        offsets = self.original_metadata['BinaryResult']['Offset']
        original_units = self.original_metadata['BinaryResult']['PixelUnitX']

        dispersion, offset = self._get_dispersion_offset()

        scale_x = self._convert_scale_units(
            pix_scale['width'], original_units, stream.spectrum_image.shape[1])
        scale_y = self._convert_scale_units(
            pix_scale['height'], original_units, stream.spectrum_image.shape[0])
        offset_x = self._convert_scale_units(
            offsets['x'], original_units, stream.spectrum_image.shape[1])
        offset_y = self._convert_scale_units(
            offsets['y'], original_units, stream.spectrum_image.shape[0])

        i = 0
        axes = []
        # add a supplementary axes when we import all frames individualy
        if self.individual_frame:
            frame_time = float(self.original_metadata['Scan']['FrameTime'])
            axes.append({'index_in_array': i,
                         'name': 'Time',
                         'offset': 0,
                         'scale': frame_time,
                         'size': stream.spectrum_image.shape[i],
                         'units': 's'})
            i += 1
        axes.extend([{'index_in_array': i,
                      'name': 'y',
                      'offset': offset_y[0],
                      'scale': scale_y[0],
                      'size': stream.spectrum_image.shape[i],
                      'units': scale_y[1]},
                     {'index_in_array': i + 1,
                      'name': 'x',
                      'offset': offset_x[0],
                      'scale': scale_x[0],
                      'size': stream.spectrum_image.shape[i + 1],
                      'units': scale_x[1]},
                     {'index_in_array': i + 2,
                      'name': 'X-ray energy',
                      'offset': offset,
                      'scale': dispersion,
                      'size': stream.spectrum_image.shape[i + 2],
                      'units': 'keV'}])

        md = self._get_metadata_dict()
        md['Signal']['signal_type'] = 'EDS_TEM'

        self.dictionaries.append({'data': stream.spectrum_image,
                                  'axes': axes,
                                  'metadata': md,
                                  'original_metadata': self.original_metadata,
                                  'mapping': self._get_mapping()})

    def _get_dispersion_offset(self):
        for detectorname, detector in self.original_metadata['Detectors'].items():
            _logger.debug('Detector: %s' % detector['DetectorName'])
            if detector['DetectorName'] == 'SuperXG21' or detector['DetectorName'] == 'SuperXG11':
                dispersion = float(
                    detector['Dispersion']) / 1000.0 * self.energy_rebin
                offset = float(
                    detector['OffsetEnergy']) / 1000.0

        return dispersion, offset

    def _convert_scale_units(self, value, units, factor=1):
        factor /= 2
        v = np.float(value) * self.ureg(units)
        converted_v = (factor * v).to_compact()
        converted_value = float(converted_v.magnitude / factor)
        converted_units = '{:~}'.format(converted_v.units)
        return converted_value, converted_units

    def _get_metadata_dict(self):
        meta_gen = {}
        meta_gen['original_filename'] = os.path.split(self.filename)[1]
        meta_gen['title'] = meta_gen['original_filename'].rpartition('.')[0]

        meta_sig = {}
        meta_sig['record_by'] = self.record_by
        meta_sig['signal_type'] = ''

        return {'General': meta_gen, 'Signal': meta_sig}

    def _read_original_metadata(self, f):
        data_group = f['Data'][self.im_type]
        metadata_array = data_group[list(data_group.keys())[
            0]]['Metadata'][:, 0]
        mdata_string = metadata_array.tostring().decode("utf-8")
        self.original_metadata = json.loads(mdata_string.rstrip('\x00'))

    def _get_mapping(self):
        mapping = {
            'Acquisition.AcquisitionStartDatetime.DateTime': (
                "General.time", self._convert_time),
            'Acquisition.AcquisitionStartDatetime': (
                "General.time_zone", lambda x: self._get_local_time_zone()),
            'Optics.AccelerationVoltage': (
                "Acquisition_instrument.TEM.beam_energy", lambda x: float(x) / 1e3),
            'Optics.CameraLength': (
                "Acquisition_instrument.TEM.camera_length", lambda x: float(x) * 1e3),
            'CustomProperties.StemMagnification.value': (
                "Acquisition_instrument.TEM.magnification", lambda x: float(x)),
            'Instrument.InstrumentClass': (
                "Acquisition_instrument.TEM.microscope", None),
            'Stage.AlphaTilt': (
                "Acquisition_instrument.TEM.tilt_stage",
                lambda x: '{:.2f}'.format(float(x))),
            'ImportedDataParameter.Frame_number': (
                "Acquisition_instrument.TEM.Detector.EDS.frame_number", None)
        }

        return mapping

    def _convert_time(self, unix_time):
        # Since we don't know the actual time zone of where the data have been
        # acquired, we convert the datetime to the local time for convinience
        dt = datetime.fromtimestamp(float(unix_time), tz=tz.tzutc())
        return dt.astimezone(tz.tzlocal()).isoformat().split('+')[0]

    def _get_local_time_zone(self):
        return tz.tzlocal().tzname(datetime.today())


class FeiSpectrumStream(object):

    """
    Below some information we have got from FEI:
    'The SI data is stored as a spectrum stream, ‘65535’ means next pixel,
    other numbers mean a spectrum count in that bin for that pixel.
    For the size of the spectrum image and dispersion you have to look in
    AcquisitionSettings.
    The spectrum image cube itself stored in a compressed format, that is
    not easy to decode.'

    When we read the array, we could add an option to import only X-rays
    acquire within a "frame range" (specific scanning passes). Parsing each
    frame to hyperspy signal will take too much memory, since the data are
    highly sparse.
    """

    def __init__(self, spectrum_stream_group, energy_rebin, data_dtype=None):
        self.spectrum_stream_group = spectrum_stream_group
        self.energy_rebin = energy_rebin
        self.data_dtype = data_dtype

        # Find sub group name
        sub_group_name = _get_keys_from_group(spectrum_stream_group)[0]

        # Read acquisition settings
        self.stream = spectrum_stream_group['%s/Data' % sub_group_name][:, 0]
        self._parse_acquisition_settings(spectrum_stream_group, sub_group_name)

    def _parse_acquisition_settings(self, group, sub_group_name):
        acquisition_key = '%s/AcquisitionSettings' % sub_group_name
        settings = json.loads(group[acquisition_key].value[0].decode('utf-8'))
        self.bin_count = int(settings['bincount'])
        if self.data_dtype is None:
            self.data_dtype = settings['StreamEncoding']

        self.acquisition_settings = settings

    def get_spectrum_image(self, shape, first_frame=0, last_frame=None,
                           individual_frame=False):
        if last_frame is None:
            last_frame = int(
                np.ceil((self.stream == 65535).sum() / (shape[0] * shape[1])))
        if self.bin_count % self.energy_rebin != 0:
            raise ValueError('The `energy_rebin` need to be a divisor of the ',
                             'total number of channels.')
        if individual_frame:
            SI = np.zeros((last_frame, shape[0], shape[1],
                           int(self.bin_count / self.energy_rebin)),
                          dtype=self.data_dtype)
            self.spectrum_image, frame_number = get_spectrum_image_individual(
                SI, self.stream, self.energy_rebin)
        else:
            SI = np.zeros((shape[0], shape[1],
                           int(self.bin_count / self.energy_rebin)),
                          dtype=self.data_dtype)
            # frame_number is from 0 to last frame
            self.spectrum_image, frame_number = get_spectrum_image(
                SI, self.stream, first_frame, last_frame, self.energy_rebin)
        self.first_frame = first_frame
        self.last_frame = frame_number
        self.frame_number = self.last_frame - self.first_frame


@jit_ifnumba
def get_spectrum_image(spectrum_image, stream, first_frame, last_frame,
                       energy_rebin=1):
    # jit speeds up this function by a factor of ~ 30
    navigation_index = 0
    frame_number = 0
    shape = spectrum_image.shape
    for count_channel in np.nditer(stream):
        # when we reach the end of the frame, reset the navigation index to 0
        if navigation_index == (shape[0] * shape[1]):
            navigation_index = 0
            frame_number += 1
            # break the for loop when we reach the last frame we want to read
            if frame_number == last_frame:
                break
            # Comment this line, if not numba can not compile this loop and it
            # is as slow as pure python
#            _logger.debug('Frame #%i'%frame_number)
        # if different of ‘65535’, add a count to the corresponding channel
        if count_channel != 65535:
            if first_frame <= frame_number:
                spectrum_image[navigation_index // shape[1],
                               navigation_index % shape[1],
                               count_channel // energy_rebin] += 1
        else:
            navigation_index += 1

    return spectrum_image, frame_number


@jit_ifnumba
def get_spectrum_image_individual(spectrum_image, stream, energy_rebin=1):
    navigation_index = 0
    frame_number = 0
    shape = spectrum_image.shape
    for count_channel in np.nditer(stream):
        # when we reach the end of the frame, reset the navigation index to 0
        if navigation_index == (shape[1] * shape[2]):
            navigation_index = 0
            frame_number += 1
        # if different of ‘65535’, add a count to the corresponding channel
        if count_channel != 65535:
            spectrum_image[frame_number,
                           navigation_index // shape[2],
                           navigation_index % shape[2],
                           count_channel // energy_rebin] += 1
        else:
            navigation_index += 1

    return spectrum_image, frame_number


def file_reader(filename, log_info=False,
                lazy=False, **kwds):
    dictionaries = []
    if fei_check(filename) == True:
        _logger.debug('EMD is FEI format')
        emd = FeiEMDReader(filename, **kwds)
        dictionaries = emd.dictionaries
    else:
        emd = EMD.load_from_emd(filename, lazy)
        if log_info:
            emd.log_info()
        for signal in emd.signals.values():
            dictionaries.append(signal._to_dictionary())

    return dictionaries


def file_writer(filename, signal, signal_metadata=None, user=None,
                microscope=None, sample=None, comments=None, **kwds):
    metadata = signal.metadata.General.as_dictionary()
    user = user or metadata.get('user', None)
    microscope = microscope or metadata.get('microscope', None)
    sample = sample or metadata.get('sample', None)
    comments = comments or metadata.get('comments', None)
    emd = EMD(
        user=user,
        microscope=microscope,
        sample=sample,
        comments=comments)
    emd.add_signal(signal, metadata=signal_metadata)
    emd.save_to_emd(filename)
