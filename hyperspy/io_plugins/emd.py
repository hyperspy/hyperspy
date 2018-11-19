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
# FEI later developed another EMD format, also based on the hdf5 standard. This
# reader first checked if the file have been saved by Velox (FEI EMD format)
# and use either the EMD class or the FEIEMDReader class to read the file.
# Writing file is only supported for EMD Berkeley file.


import re
import json
import os
from datetime import datetime
import time
import logging
import traits.api as t

import h5py
import numpy as np
import dask.array as da
from dateutil import tz
import pint

from hyperspy.misc.elements import atomic_number2name
import hyperspy.misc.io.fei_stream_readers as stream_readers

# Plugin characteristics
# ----------------------
format_name = 'Electron Microscopy Data (EMD)'
description = 'Read data from Berkeleys EMD files.'
full_support = False  # Hopefully?
# Recognised file extension
file_extensions = ('emd', 'EMD')
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing features
writes = True  # Only Berkeley emd
EMD_VERSION = '0.2'
# ----------------------

_logger = logging.getLogger(__name__)


def calculate_chunks(shape, dtype, chunk_size_mb=100):
    """Calculate chunks to get target chunk size.

    The chunks are optimized for C-order reading speed.

    Parameters
    ----------
    shape: tuple of ints
        The shape of the array
    dtype: string or numpy dtype
        The dtype of the array
    chunk_size_mb: int
        The maximum size of the resulting chunks in MB. The default is
        100MB as reccommended by the dask documentation.

    """

    target = chunk_size_mb * 1e6
    items = int(target // np.dtype(dtype).itemsize)
    chunks = ()
    dimsize = np.cumprod(shape[::-1])[::-1][1:]
    for i, ds in enumerate(dimsize):
        chunk = items // ds
        if not chunk:
            chunks += (1,)
        elif chunk <= shape[i]:
            chunks += (chunk, )
        else:
            chunks += (shape[i],)
    # At least one signal
    chunks += (shape[-1], )
    return chunks


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
            data = da.from_array(data, chunks=data.chunks)
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
            units = re.findall(r'[^_\W]+', axis_units)
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
    def load_from_emd(cls, filename, lazy=False, dataset_name=None):
        """Construct :class:`~.EMD` object from an emd-file.

        Parameters
        ----------
        filename : string
            The name of the emd-file from which to load the signals. Standard
            format is '*.emd'.
        False : bool, optional
            If False (default) loads data to memory. If True, enables loading
            only if requested.
        dataset_name : string or iterable, optional
            Only add dataset with specific name. Note, this has to be the full
            group path in the file. For example `/experimental/science_data'.
            If the dataset is not found, an IOError with the possible
            datasets will be raised. Several names can be specified
            in the form of a list.

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
        dataset_in_file_list = []
        for node in node_list:
            data_group = emd_file.get(node)
            if data_group is not None:
                for group in data_group.values():
                    name = group.name
                    if isinstance(group, h5py.Group):
                        if group.attrs.get('emd_group_type') == 1:
                            dataset_in_file_list.append(name)
        if len(dataset_in_file_list) == 0:
            raise IOError("No datasets found in {0}".format(filename))
        dataset_read_list = []
        if dataset_name is not None:
            if isinstance(dataset_name, str):
                dataset_name = [dataset_name]

            for temp_dataset_name in dataset_name:
                if temp_dataset_name in dataset_in_file_list:
                    dataset_read_list.append(temp_dataset_name)
                else:
                    raise IOError(
                        "Dataset with name {0} not found in the file. "
                        "Possible datasets are {1}.".format(
                            temp_dataset_name,
                            ', '.join(dataset_in_file_list)))
        else:
            dataset_read_list = dataset_in_file_list
        for dataset_read in dataset_read_list:
            group = emd_file[dataset_read]
            emd._read_signal_from_group(dataset_read, group, lazy)

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
        """( all relevant information about the EMD instance."""
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
            if v_dict['format'] in ['Velox', 'DevelopersKit']:
                return True


def _get_keys_from_group(group):
    # Return a list of ids of items contains in the group
    return list(group.keys())


def _parse_sub_data_group_metadata(sub_data_group):
    metadata_array = sub_data_group['Metadata'][:, 0].T
    mdata_string = metadata_array.tostring().decode("utf-8")
    return json.loads(mdata_string.rstrip('\x00'))


def _parse_metadata(data_group, sub_group_key):
    return _parse_sub_data_group_metadata(data_group[sub_group_key])


def _parse_detector_name(original_metadata):
    try:
        name = original_metadata['BinaryResult']['Detector']
    except KeyError:
        # if the `BinaryResult/Detector` is not available, there should be
        # only one detector in `Detectors`
        name = original_metadata['Detectors']['Detector-01']['DetectorName']
    return name


def _get_detector_metadata_dict(om, detector_name):
    detectors_dict = om['Detectors']
    # find detector dict from the detector_name
    for key in detectors_dict:
        if detectors_dict[key]['DetectorName'] == detector_name:
            return detectors_dict[key]


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

    """

    def __init__(self, filename, select_type=None, first_frame=0,
                 last_frame=None, sum_frames=True, sum_EDS_detectors=True,
                 rebin_energy=1, SI_dtype=None, load_SI_image_stack=False,
                 lazy=False):
        # TODO: Finish lazy implementation using the `FrameLocationTable`
        # Parallelise streams reading
        self.filename = filename
        self.ureg = pint.UnitRegistry()
        self.dictionaries = []
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.sum_frames = sum_frames
        self.sum_EDS_detectors = sum_EDS_detectors
        self.rebin_energy = rebin_energy
        self.SI_data_dtype = SI_dtype
        self.load_SI_image_stack = load_SI_image_stack
        self.lazy = lazy
        self.detector_name = None

        self.original_metadata = {}
        try:
            f = h5py.File(filename, 'r')
            self.d_grp = f.get('Data')
            self._check_im_type()
            self._parse_metadata_group(f.get('Operations'), 'Operations')
            if self.im_type == 'SpectrumStream':
                self.p_grp = f.get('Presentation')
                self._parse_image_display()
            self._read_data(select_type)
        except Exception as e:
            raise e
        finally:
            if not self.lazy:
                f.close()

    def _read_data(self, select_type):
        self.load_images = self.load_SI = self.load_single_spectrum = True
        if select_type == 'single_spectrum':
            self.load_images = self.load_SI = False
        elif select_type == 'images':
            self.load_SI = self.load_single_spectrum = False
        elif select_type == 'spectrum_image':
            self.load_images = self.load_single_spectrum = False
        elif select_type is None:
            pass
        else:
            raise ValueError("`select_type` parameter takes only: `None`, "
                             "'single_spectrum', 'images' or 'spectrum_image'.")

        if self.im_type == 'Image':
            _logger.info('Reading the images.')
            self._read_images()
        elif self.im_type == 'Spectrum':
            self._read_single_spectrum()
            self._read_images()
        elif self.im_type == 'SpectrumStream':
            self._read_single_spectrum()
            _logger.info('Reading the spectrum image.')
            t0 = time.time()
            self._read_images()
            t1 = time.time()
            self._read_spectrum_stream()
            t2 = time.time()
            _logger.info('Time to load images: {} s.'.format(t1 - t0))
            _logger.info('Time to load spectrum image: {} s.'.format(t2 - t1))

    def _check_im_type(self):
        if 'Image' in self.d_grp:
            if 'SpectrumImage' in self.d_grp:
                self.im_type = 'SpectrumStream'
            else:
                self.im_type = 'Image'
        else:
            self.im_type = 'Spectrum'

    def _read_single_spectrum(self):
        if not self.load_single_spectrum:
            return
        spectrum_grp = self.d_grp.get("Spectrum")
        if spectrum_grp is None:
            return  # No spectra in the file
        self.detector_name = 'EDS'
        for spectrum_sub_group_key in _get_keys_from_group(spectrum_grp):
            self.dictionaries.append(
                self._read_spectrum(spectrum_grp, spectrum_sub_group_key))

    def _read_spectrum(self, spectrum_group, spectrum_sub_group_key):
        spectrum_sub_group = spectrum_group[spectrum_sub_group_key]
        dataset = spectrum_sub_group['Data']
        if self.lazy:
            data = da.from_array(dataset, chunks=dataset.chunks).T
        else:
            data = dataset[:].T
        original_metadata = _parse_metadata(spectrum_group,
                                            spectrum_sub_group_key)
        original_metadata.update(self.original_metadata)

        dispersion, offset, unit = self._get_dispersion_offset(
            original_metadata)
        axes = []
        if len(data.shape) == 2:
            if data.shape[0] == 1:
                # squeeze
                data = data[0, :]
            else:
                axes = [{
                    'name': 'Stack',
                    'offset': 0,
                    'scale': 1,
                    'size': data.shape[0],
                    'navigate': True,
                }
                ]
        axes.append({
            'name': 'Energy',
            'offset': offset,
            'scale': dispersion,
            'size': data.shape[-1],
            'units': 'keV',
            'navigate': False},
        )

        md = self._get_metadata_dict(original_metadata)
        md['Signal']['signal_type'] = 'EDS_TEM'

        return {'data': data,
                'axes': axes,
                'metadata': md,
                'original_metadata': original_metadata,
                'mapping': self._get_mapping()}

    def _read_images(self):
        # We need to read the image to get the shape of the spectrum image
        if not self.load_images and not self.load_SI:
            return
        # Get the image data group
        image_group = self.d_grp.get("Image")
        if image_group is None:
            return  # No images in the file
        # Get all the subgroup of the image data group and read the image for
        # each of them
        for image_sub_group_key in _get_keys_from_group(image_group):
            image = self._read_image(image_group, image_sub_group_key)
            if not self.load_images:
                # If we don't want to load the images, we stop here
                return
            self.dictionaries.append(image)

    def _read_image(self, image_group, image_sub_group_key):
        """ Return a dictionary ready to parse of return to io module"""
        image_sub_group = image_group[image_sub_group_key]
        original_metadata = _parse_metadata(image_group, image_sub_group_key)
        original_metadata.update(self.original_metadata)
        if 'Detector' in original_metadata['BinaryResult'].keys():
            self.detector_name = _parse_detector_name(original_metadata)

        read_stack = (self.load_SI_image_stack or self.im_type == 'Image')
        h5data = image_sub_group['Data']
        # Get the scanning area shape of the SI from the images
        self.spatial_shape = h5data.shape[:-1]
        # Set the axes in frame, y, x order
        if self.lazy:
            data = da.transpose(
                da.from_array(
                    h5data,
                    chunks=h5data.chunks),
                axes=[2, 0, 1])
        else:
            # Workaround for a h5py bug https://github.com/h5py/h5py/issues/977
            # Change back to standard API once issue #977 is fixed.
            # Preallocate the numpy array and use read_direct method, which is
            # much faster in case of chunked data.
            data = np.empty(h5data.shape)
            h5data.read_direct(data)
            data = np.rollaxis(data, axis=2)

        pix_scale = original_metadata['BinaryResult'].get(
            'PixelSize', {'height': 1.0, 'width': 1.0})
        offsets = original_metadata['BinaryResult'].get(
            'Offset', {'x': 0.0, 'y': 0.0})
        original_units = original_metadata['BinaryResult'].get(
            'PixelUnitX', '')

        axes = []
        # stack of images
        if not read_stack:
            data = data[0:1, ...]

        if data.shape[0] == 1:
            # Squeeze
            data = data[0, ...]
            i = 0
        else:
            frame_time = original_metadata['Scan']['FrameTime']
            frame_time, time_unit = self._convert_scale_units(
                frame_time, 's', 2 * data.shape[0])
            axes.append({'index_in_array': 0,
                         'name': 'Time',
                         'offset': 0,
                         'scale': frame_time,
                         'size': data.shape[0],
                         'units': time_unit,
                         'navigate': True})
            i = 1
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
                      'units': scale_y[1],
                      'navigate': False},
                     {'index_in_array': i + 1,
                      'name': 'x',
                      'offset': offset_x[0],
                      'scale': scale_x[0],
                      'size': data.shape[i + 1],
                      'units': scale_x[1],
                      'navigate': False}
                     ])

        md = self._get_metadata_dict(original_metadata)
        md['Signal']['signal_type'] = 'image'
        if self.detector_name is not None:
            original_metadata['DetectorMetadata'] = _get_detector_metadata_dict(
                original_metadata,
                self.detector_name)
        if hasattr(self, 'map_label_dict'):
            if image_sub_group_key in self.map_label_dict:
                md['General']['title'] = self.map_label_dict[image_sub_group_key]

        return {'data': data,
                'axes': axes,
                'metadata': md,
                'original_metadata': original_metadata,
                'mapping': self._get_mapping(map_selected_element=False,
                                             parse_individual_EDS_detector_metadata=False)}

    def _parse_frame_time(self, original_metadata, factor=1):
        try:
            frame_time = original_metadata['Scan']['FrameTime']
            time_unit = 's'
        except KeyError:
            frame_time, time_unit = None, t.Undefined

        frame_time, time_unit = self._convert_scale_units(
            frame_time, time_unit, factor)
        return frame_time, time_unit

    def _parse_image_display(self):
        try:
            image_display_group = self.p_grp.get('Displays/ImageDisplay')
            key_list = _get_keys_from_group(image_display_group)
            self.map_label_dict = {}
            for key in key_list:
                v = json.loads(
                    image_display_group[key].value[0].decode('utf-8'))
                data_key = v['dataPath'].split('/')[-1]  # key in data group
                self.map_label_dict[data_key] = v['display']['label']
        except KeyError:
            _logger.warning("The image label can't be read from the metadata.")
            pass

    def _parse_metadata_group(self, group, group_name):
        d = {}
        try:
            for group_key in _get_keys_from_group(group):
                subgroup = group.get(group_key)
                if hasattr(subgroup, 'keys'):
                    sub_dict = {}
                    for subgroup_key in _get_keys_from_group(subgroup):
                        v = json.loads(
                            subgroup[subgroup_key].value[0].decode('utf-8'))
                        sub_dict[subgroup_key] = v
                else:
                    sub_dict = json.loads(subgroup.value[0].decode('utf-8'))
                d[group_key] = sub_dict
        except IndexError:
            _logger.warning("Some metadata can't be read.")
        self.original_metadata.update({group_name: d})

    def _read_spectrum_stream(self):
        if not self.load_SI:
            return
        self.detector_name = 'EDS'
        # Try to read the number of frames from Data/SpectrumImage
        try:
            sig = self.d_grp["SpectrumImage"]
            self.number_of_frames = int(
                json.loads(
                    sig[next(iter(sig))]
                    ["SpectrumImageSettings"][0].decode("utf8")
                )["endFramePosition"])
        except Exception as e:
            _logger.exception(
                "Failed to read the number of frames from Data/SpectrumImage")
            self.number_of_frames = None
        if self.last_frame is None:
            self.last_frame = self.number_of_frames
        elif self.number_of_frames and self.last_frame > self.number_of_frames:
            raise ValueError(
                "The `last_frame` cannot be greater than"
                " the number of frames, %i for this file."
                % self.number_of_frames
            )

        spectrum_stream_group = self.d_grp.get("SpectrumStream")
        if spectrum_stream_group is None:
            _logger.warning("No spectrum stream is present in the file. It "
                            "is possible that the file has been pruned: use "
                            "Velox to read the spectrum image (proprietary "
                            "format). If you want to open FEI emd file with "
                            "HyperSpy don't prune the file when saving it in "
                            "Velox.")
            return

        def _read_stream(key):
            stream = FeiSpectrumStream(spectrum_stream_group[key], self)
            return stream

        subgroup_keys = _get_keys_from_group(spectrum_stream_group)
        if self.sum_EDS_detectors:
            if len(subgroup_keys) == 1:
                _logger.warning("The file contains only one spectrum stream")
            # Read the first stream
            s0 = _read_stream(subgroup_keys[0])
            streams = [s0]
            # add other stream streams
            if len(subgroup_keys) > 1:
                for key in subgroup_keys[1:]:
                    stream_data = spectrum_stream_group[key]['Data'][:].T[0]
                    if self.lazy:
                        s0.spectrum_image = (
                            s0.spectrum_image +
                            s0.stream_to_sparse_array(stream_data=stream_data)
                        )
                    else:
                        s0.stream_to_array(stream_data=stream_data,
                                           spectrum_image=s0.spectrum_image)
        else:
            streams = [_read_stream(key) for key in subgroup_keys]
        if self.lazy:
            for stream in streams:
                sa = stream.spectrum_image.astype(self.SI_data_dtype)
                stream.spectrum_image = sa

        spectrum_image_shape = streams[0].shape
        original_metadata = streams[0].original_metadata
        original_metadata.update(self.original_metadata)

        pixel_size, offsets, original_units = \
            streams[0].get_pixelsize_offset_unit()
        dispersion, offset, unit = self._get_dispersion_offset(
            original_metadata)

        scale_x = self._convert_scale_units(
            pixel_size['width'], original_units, spectrum_image_shape[1])
        scale_y = self._convert_scale_units(
            pixel_size['height'], original_units, spectrum_image_shape[0])
        offset_x = self._convert_scale_units(
            offsets['x'], original_units, spectrum_image_shape[1])
        offset_y = self._convert_scale_units(
            offsets['y'], original_units, spectrum_image_shape[0])

        i = 0
        axes = []
        # add a supplementary axes when we import all frames individualy
        if not self.sum_frames:
            frame_time, time_unit = self._parse_frame_time(original_metadata,
                                                           spectrum_image_shape[i])
            axes.append({'index_in_array': i,
                         'name': 'Time',
                         'offset': 0,
                         'scale': frame_time,
                         'size': spectrum_image_shape[i],
                         'units': time_unit,
                         'navigate': True})
            i = 1
        axes.extend([{'index_in_array': i,
                      'name': 'y',
                      'offset': offset_y[0],
                      'scale': scale_y[0],
                      'size': spectrum_image_shape[i],
                      'units': scale_y[1],
                      'navigate': True},
                     {'index_in_array': i + 1,
                      'name': 'x',
                      'offset': offset_x[0],
                      'scale': scale_x[0],
                      'size': spectrum_image_shape[i + 1],
                      'units': scale_x[1],
                      'navigate': True},
                     {'index_in_array': i + 2,
                      'name': 'X-ray energy',
                      'offset': offset,
                      'scale': dispersion,
                      'size': spectrum_image_shape[i + 2],
                      'units': unit,
                      'navigate': False}])

        md = self._get_metadata_dict(original_metadata)
        md['Signal']['signal_type'] = 'EDS_TEM'

        for stream in streams:
            original_metadata = stream.original_metadata
            original_metadata.update(self.original_metadata)
            self.dictionaries.append({'data': stream.spectrum_image,
                                      'axes': axes,
                                      'metadata': md,
                                      'original_metadata': original_metadata,
                                      'mapping': self._get_mapping(
                                          parse_individual_EDS_detector_metadata=not self.sum_frames)})

    def _get_dispersion_offset(self, original_metadata):
        try:
            for detectorname, detector in original_metadata['Detectors'].items(
            ):
                if original_metadata['BinaryResult']['Detector'] in detector['DetectorName']:
                    dispersion = float(
                        detector['Dispersion']) / 1000.0 * self.rebin_energy
                    offset = float(
                        detector['OffsetEnergy']) / 1000.0
                    return dispersion, offset, 'keV'
        except KeyError:
            _logger.warning("The spectrum calibration can't be loaded.")
            return 1, 0, t.Undefined

    def _convert_scale_units(self, value, units, factor=1):
        if units == t.Undefined:
            return value, units
        factor /= 2
        v = np.float(value) * self.ureg(units)
        converted_v = (factor * v).to_compact()
        converted_value = float(converted_v.magnitude / factor)
        converted_units = '{:~}'.format(converted_v.units)
        return converted_value, converted_units

    def _get_metadata_dict(self, om):
        meta_gen = {}
        meta_gen['original_filename'] = os.path.split(self.filename)[1]
        if self.detector_name is not None:
            meta_gen['title'] = self.detector_name
        # We have only one entry in the original_metadata, so we can't use
        # the mapping of the original_metadata to set the date and time in
        # the metadata: need to set it manually here
        try:
            if 'AcquisitionStartDatetime' in om['Acquisition'].keys():
                unix_time = om['Acquisition']['AcquisitionStartDatetime']['DateTime']
            # Workaround when the 'AcquisitionStartDatetime' key is missing
            # This timestamp corresponds to when the data is stored
            elif (not isinstance(om['CustomProperties'], str) and
                  'Detectors[BM-Ceta].TimeStamp' in om['CustomProperties'].keys()):
                unix_time = float(
                    om['CustomProperties']['Detectors[BM-Ceta].TimeStamp']['value']) / 1E6
            date, time = self._convert_datetime(unix_time).split('T')
            meta_gen['date'] = date
            meta_gen['time'] = time
            meta_gen['time_zone'] = self._get_local_time_zone()
        except (UnboundLocalError):
            pass

        meta_sig = {}
        meta_sig['signal_type'] = ''

        return {'General': meta_gen, 'Signal': meta_sig}

    def _get_mapping(self, map_selected_element=True,
                     parse_individual_EDS_detector_metadata=True):
        mapping = {
            'Acquisition.AcquisitionStartDatetime.DateTime': (
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
                "Acquisition_instrument.TEM.Stage.tilt_alpha",
                lambda x: round(np.degrees(float(x)), 3)),
            'Stage.BetaTilt': (
                "Acquisition_instrument.TEM.Stage.tilt_beta",
                lambda x: round(np.degrees(float(x)), 3)),
            'Stage.Position.x': (
                "Acquisition_instrument.TEM.Stage.x",
                lambda x: round(float(x), 6)),
            'Stage.Position.y': (
                "Acquisition_instrument.TEM.Stage.y",
                lambda x: round(float(x), 6)),
            'Stage.Position.z': (
                "Acquisition_instrument.TEM.Stage.z",
                lambda x: round(float(x), 6)),
            'ImportedDataParameter.Number_of_frames': (
                "Acquisition_instrument.TEM.Detector.EDS.number_of_frames", None),
            'DetectorMetadata.ElevationAngle': (
                "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                lambda x: round(float(x), 3)),
            'DetectorMetadata.Gain': (
                "Signal.Noise_properties.Variance_linear_model.gain_factor",
                lambda x: float(x)),
            'DetectorMetadata.Offset': (
                "Signal.Noise_properties.Variance_linear_model.gain_offset",
                lambda x: float(x)),
        }

        # Parse individual metadata for each EDS detector
        if parse_individual_EDS_detector_metadata:
            mapping.update({
                'DetectorMetadata.AzimuthAngle': (
                    "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                    lambda x: '{:.3f}'.format(np.degrees(float(x)))),
                'DetectorMetadata.LiveTime': (
                    "Acquisition_instrument.TEM.Detector.EDS.live_time",
                    lambda x: '{:.6f}'.format(float(x))),
                'DetectorMetadata.RealTime': (
                    "Acquisition_instrument.TEM.Detector.EDS.real_time",
                    lambda x: '{:.6f}'.format(float(x))),
                'DetectorMetadata.DetectorName': (
                    "General.title", None),
            })

        # Add selected element
        if map_selected_element:
            mapping.update({'Operations.ImageQuantificationOperation': (
                            'Sample.elements',
                            self._convert_element_list),
                            })

        return mapping

    def _convert_element_list(self, d):
        atomic_number_list = d[d.keys()[0]]['elementSelection']
        return [atomic_number2name[int(atomic_number)]
                for atomic_number in atomic_number_list]

    def _convert_datetime(self, unix_time):
        # Since we don't know the actual time zone of where the data have been
        # acquired, we convert the datetime to the local time for convenience
        dt = datetime.fromtimestamp(float(unix_time), tz=tz.tzutc())
        return dt.astimezone(tz.tzlocal()).isoformat().split('+')[0]

    def _get_local_time_zone(self):
        return tz.tzlocal().tzname(datetime.today())


# Below some information we have got from FEI about the format of the stream:
#
# The SI data is stored as a spectrum stream, ‘65535’ means next pixel
# (these markers are also called `Gate pulse`), other numbers mean a spectrum
# count in that bin for that pixel.
# For the size of the spectrum image and dispersion you have to look in
# AcquisitionSettings.
# The spectrum image cube itself stored in a compressed format, that is
# not easy to decode.

class FeiSpectrumStream(object):
    """Read spectrum image stored in FEI's stream format

    Once initialized, the instance of this class supports numpy style
    indexing and slicing of the data stored in the stream format.
    """

    def __init__(self, stream_group, reader):
        self.reader = reader
        self.stream_group = stream_group
        # Parse acquisition settings to get bin_count and dtype
        acquisition_settings_group = stream_group['AcquisitionSettings']
        acquisition_settings = json.loads(
            acquisition_settings_group.value[0].decode('utf-8'))
        self.bin_count = int(acquisition_settings['bincount'])
        if self.bin_count % self.reader.rebin_energy != 0:
            raise ValueError('The `rebin_energy` needs to be a divisor of the',
                             ' total number of channels.')
        if self.reader.SI_data_dtype is None:
            self.reader.SI_data_dtype = acquisition_settings['StreamEncoding']
        # Parse the rest of the metadata for storage
        self.original_metadata = _parse_sub_data_group_metadata(stream_group)
        # If last_frame is None, compute it
        stream_data = self.stream_group['Data'][:].T[0]
        if self.reader.last_frame is None:
            # The information could not be retrieved from metadata
            # we compute, which involves iterating once over the whole stream.
            # This is required to support the `last_frame` feature without
            # duplicating the functions as currently numba does not support
            # parametetrization.
            spatial_shape = self.reader.spatial_shape
            last_frame = int(
                np.ceil((stream_data == 65535).sum() /
                        (spatial_shape[0] * spatial_shape[1])))
            self.reader.last_frame = last_frame
            self.reader.number_of_frames = last_frame
        self.original_metadata['ImportedDataParameter'] = {
            'First_frame': self.reader.first_frame,
            'Last_frame': self.reader.last_frame,
            'Number_of_frames': self.reader.number_of_frames,
            'Rebin_energy': self.reader.rebin_energy,
            'Number_of_channels': self.bin_count, }
        # Convert stream to spectrum image
        if self.reader.lazy:
            self.spectrum_image = self.stream_to_sparse_array(
                stream_data=stream_data
            )
        else:
            self.spectrum_image = self.stream_to_array(
                stream_data=stream_data
            )

    @property
    def shape(self):
        return self.spectrum_image.shape

    def get_pixelsize_offset_unit(self):
        om_br = self.original_metadata['BinaryResult']
        return om_br['PixelSize'], om_br['Offset'], om_br['PixelUnitX']

    def stream_to_sparse_array(self, stream_data):
        """Convert stream in sparse array

        Parameters
        ----------
        stream_data: array

        """
        # Here we load the stream data into memory, which is fine is the
        # arrays are small. We could load them lazily when lazy.
        stream_data = self.stream_group['Data'][:].T[0]
        sparse_array = stream_readers.stream_to_sparse_COO_array(
            stream_data=stream_data,
            spatial_shape=self.reader.spatial_shape,
            first_frame=self.reader.first_frame,
            last_frame=self.reader.last_frame,
            channels=self.bin_count,
            sum_frames=self.reader.sum_frames,
            rebin_energy=self.reader.rebin_energy,
        )
        return sparse_array

    def stream_to_array(self, stream_data, spectrum_image=None):
        """Convert stream to array.

        Parameters
        ----------
        stream_data: array
        spectrum_image: array or None
            If array, the data from the stream are added to the array.
            Otherwise it creates a new array and returns it.

        """
        spectrum_image = stream_readers.stream_to_array(
            stream=stream_data,
            spatial_shape=self.reader.spatial_shape,
            channels=self.bin_count,
            first_frame=self.reader.first_frame,
            last_frame=self.reader.last_frame,
            rebin_energy=self.reader.rebin_energy,
            sum_frames=self.reader.sum_frames,
            spectrum_image=spectrum_image,
            dtype=self.reader.SI_data_dtype,
        )
        return spectrum_image


def file_reader(filename, log_info=False,
                lazy=False, **kwds):
    dictionaries = []
    if fei_check(filename) == True:
        _logger.debug('EMD is FEI format')
        emd = FeiEMDReader(filename, lazy=lazy, **kwds)
        dictionaries = emd.dictionaries
    else:
        emd = EMD.load_from_emd(filename, lazy, **kwds)
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
