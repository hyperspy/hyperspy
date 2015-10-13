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

# INFORMATION ABOUT THE EMD FORMAT CAN BE FOUND HERE:
# http://emdatasets.lbl.gov/


import re
import h5py

import logging


# Plugin characteristics
# ----------------------
format_name = 'Electron Microscopy Data (EMD)'
description = 'Read data from Berkleys EMD files.'
full_support = True  # Hopefully?
# Recognised file extension
file_extensions = ('emd', 'EMD')
default_extension = 0
# Writing features
writes = True
EMD_VERSION = '0.2'
# ----------------------


class EMD(object):

    '''Class for storing electron microscopy datasets.

    The :class:`~.EMD` class can hold an arbitrary amount of datasets in the `signals` dictionary.
    These are saved as HyperSpy :class:`~hyperspy.signal.Signal` instances. Global metadata
    are saved in four dictionaries (`user`, `microscope`, `sample`, `comments`). To print
    relevant information about the EMD instance use the :func:`~.print_info` function. EMD
    instances can be loaded from and saved to emd-files, an hdf5 standard developed at Lawrence
    Berkeley National Lab (http://emdatasets.lbl.gov/).

    Attributes
    ----------
    signals: dictionary
        Dictionary which contains all datasets as :class:`~hyperspy.signal.Signal` instances.
    user: dictionary
        Dictionary which contains user related metadata.
    microscope: dictionary
        Dictionary which contains microscope related metadata.
    sample: dictionary
        Dictionary which contains sample related metadata.
    comments: dictionary
        Dictionary which contains additional commentary metadata.

    '''

    _log = logging.getLogger(__name__)

    def __init__(self, signals={}, user={}, microscope={}, sample={}, comments={}):
        self._log.debug('Calling __init__')
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
        for name, signal in signals.iteritems():
            self.add_signal(signal, name)

    def __getitem__(self, key):
        # This is for accessing the raw data easily. For the signals use emd.signals[key]!
        return self.signals[key].data

    def _write_signal_to_group(self, data_group, signal):
        self._log.debug('Calling _write_signal_to_group')
        # Save data:

        dataset = data_group.create_group(signal.metadata.General.title)
        dataset['data'] = signal.data
        # Iterate over all dimensions:
        for i in range(len(signal.data.shape)):
            key = 'dim{}'.format(i+1)
            offset = signal.axes_manager[i].offset
            scale = signal.axes_manager[i].scale
            dim = dataset.create_dataset(key, data=[offset, offset+scale])
            name = signal.axes_manager[i].name
            from traits.trait_base import _Undefined
            if type(name) is _Undefined:
                name = ''
            dim.attrs['name'] = name
            units = signal.axes_manager[i].units
            if type(units) is _Undefined:
                units = ''
            else:
                units = '[{}]'.format('_'.join(list(units)))
            dim.attrs['units'] = units
        # Write metadata:
        dataset.attrs['emd_group_type'] = 1
        for key, value in signal.metadata.Signal:
            dataset.attrs[key] = value

    def _read_signal_from_group(self, name, group):
        self._log.debug('Calling _read_signal_from_group')
        import hyperspy.api as hp
        # Extract essential data:
        data = group.get('data')[...]
        record_by = group.attrs.get('record_by', '')
        # Create Signal, Image or Spectrum:
        if record_by == 'spectrum':
            signal = hp.signals.Spectrum(data)
        if record_by == 'image':
            signal = hp.signals.Image(data)
        else:
            signal = hp.signals.Signal(data)
        # Set signal properties:
        signal.set_signal_origin = group.attrs.get('signal_origin', '')
        signal.set_signal_type = group.attrs.get('signal_type', '')
        # Iterate over all dimensions:
        for i in range(len(data.shape)):
            dim = group.get('dim{}'.format(i+1))
            signal.axes_manager[i].name = dim.attrs.get('name', '')
            units = re.findall('[^_\W]+', dim.attrs.get('units', ''))
            signal.axes_manager[i].units = ''.join(units)
            try:
                signal.axes_manager[i].scale = dim[1] - dim[0]
                signal.axes_manager[i].offset = dim[0]
            except (IndexError, TypeError) as e:  # Hyperspy uses defaults (1.0 and 0.0)!
                self._log.warning('Could not calculate scale/offset of axis {}: {}'.format(i, e))
        # Extract metadata:
        metadata = {}
        for key, value in group.attrs.iteritems():
            metadata[key] = value
        # Add signal:
        self.add_signal(signal, name, metadata)

    def add_signal(self, signal, name=None, metadata={}):
        '''Add a hyperspy signal to the EMD instance and make sure all metadata is present.

        Parameters
        ----------
        signal: :class:`~hyperspy.signal.Signal`
            HyperSpy signal which should be added to the EMD instance.
        name: string, optional
            Name of the (used as a key for the `signals` dictionary). If not specified,
            `signal.metadata.General.title` will be used. If this is an empty string, both name
            and signal title are set to 'dataset' per default. If specified, `name` overwrites the
            signal title.
        metadata: dictionary
            Dictionary which holds signal specific metadata which will be added to the signal.

        Returns
        -------
        None

        Notes
        -----
        This is the preferred way to add signals to the EMD instance. Directly adding to the
        `signals` dictionary is possible but does not make sure all metadata are correct. This is
        called in the standard constructor on all entries in the `signals` dictionary!

        '''
        self._log.debug('Calling add_signal')
        # Check and save title:
        if name is not None:  # Overwrite Signal title!
            signal.metadata.General.title = name
        else:
            if signal.metadata.General.title is not '':  # Take title of Signal!
                name = signal.metadata.General.title
            else:  # Take default!
                name = 'dataset'
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
        # Add signal:
        self.signals[name] = signal

    @classmethod
    def load_from_emd(cls, filename):
        '''Construct :class:`~.EMD` object from an emd-file.

        Parameters
        ----------
        filename : string
            The name of the emd-file from which to load the signals. Standard format is '*.emd'.

        Returns
        -------
        emd: :class:`~.EMD`
            A :class:`~.EMD` object containing the loaded signals.

        '''
        cls._log.debug('Calling load_from_ems')
        # Read in file:
        emd_file = h5py.File(filename, 'r')
        # Creat empty EMD instance:
        emd = EMD()
        # Extract user:
        user_group = emd_file.get('user')
        if user_group is not None:
            for key, value in user_group.attrs.iteritems():
                emd.user[key] = value
        # Extract microscope:
        microscope_group = emd_file.get('microscope')
        if microscope_group is not None:
            for key, value in microscope_group.attrs.iteritems():
                emd.microscope[key] = value
        # Extract sample:
        sample_group = emd_file.get('sample')
        if sample_group is not None:
            for key, value in sample_group.attrs.iteritems():
                emd.sample[key] = value
        # Extract comments:
        comments_group = emd_file.get('comments')
        if comments_group is not None:
            for key, value in comments_group.attrs.iteritems():
                emd.comments[key] = value
        # Extract signals:
        node_list = emd_file.keys()
        for key in ['user', 'microscope', 'sample', 'comments']:  # Nodes which are not the data!
            if key in node_list:
                node_list.pop(node_list.index(key))  # Pop all unwanted nodes!
        # One node should remain, the data node (named 'data', 'signals', 'experiments', ...)!
        assert len(node_list) == 1, 'Dataset location is ambiguous!'
        data_group = emd_file.get(node_list[0])
        if data_group is not None:
            for name, group in data_group.iteritems():
                if isinstance(group, h5py.Group):
                    if group.attrs.get('emd_group_type') == 1:
                        emd._read_signal_from_group(name, group)
        # Close file and return EMD object:
        emd_file.close()
        return emd

    def save_to_emd(self, filename='datacollection.emd'):
        '''Save :class:`~.EMD` data in a file with emd(hdf5)-format.

        Parameters
        ----------
        filename : string, optional
            The name of the emd-file in which to store the signals.
            The default is 'datacollection.emd'.

        Returns
        -------
        None

        '''
        self._log.debug('Calling save_to_emd')
        # Open file:
        emd_file = h5py.File(filename, 'w')
        # Write version:
        ver_maj, ver_min = EMD_VERSION.split('.')
        emd_file.attrs['version_major'] = ver_maj
        emd_file.attrs['version_minor'] = ver_min
        # Write user:
        user_group = emd_file.create_group('user')
        for key, value in self.user.iteritems():
            user_group.attrs[key] = value
        # Write microscope:
        microscope_group = emd_file.create_group('microscope')
        for key, value in self.microscope.iteritems():
            microscope_group.attrs[key] = value
        # Write sample:
        sample_group = emd_file.create_group('sample')
        for key, value in self.sample.iteritems():
            sample_group.attrs[key] = value
        # Write comments:
        comments_group = emd_file.create_group('comments')
        for key, value in self.comments.iteritems():
            comments_group.attrs[key] = value
        # Write signals:
        data_group = emd_file.create_group('signals')
        for signal in self.signals.values():
            self._write_signal_to_group(data_group, signal)
        # Close file and return EMD object:
        emd_file.close()

    def print_info(self):
        '''Print all relevant information about the EMD instance.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        self._log.debug('Calling print_info')
        print '\nUser:\n--------------------'
        for key, value in self.user.iteritems():
            print '{}:'.format(key).ljust(15), value
        print '--------------------\n\nMicroscope:\n--------------------'
        for key, value in self.microscope.iteritems():
            print '{}:'.format(key).ljust(15), value
        print '--------------------\n\nSample:\n--------------------'
        for key, value in self.sample.iteritems():
            print '{}:'.format(key).ljust(15), value
        print '--------------------\n\nComments:\n--------------------'
        for key, value in self.comments.iteritems():
            print '{}:'.format(key).ljust(15), value
        print '--------------------\n\nData:\n--------------------'
        for key, value in self.signals.iteritems():
            print '{}:'.format(key).ljust(15), value
            print value.metadata.Signal
        print '--------------------\n'


def file_reader(filename, print_info=False, **kwds):
    emd = EMD.load_from_emd(filename)
    if print_info:
        emd.print_info()
    dictionaries = []
    for signal in emd.signals.values():
        axes = []
        for i in range(len(signal.data.shape)):
            axes.append({'size': signal.axes_manager[i].size,
                         'index_in_array': signal.axes_manager[i].index_in_array,
                         'name': signal.axes_manager[i].name,
                         'scale': signal.axes_manager[i].scale,
                         'offset': signal.axes_manager[i].offset,
                         'units': signal.axes_manager[i].units})
        dictionary = {'data': signal.data,
                      'axes': axes,
                      'metadata': signal.metadata.as_dictionary(),
                      'original_metadata': signal.original_metadata.as_dictionary()}
        dictionaries.append(dictionary)
    return dictionaries


def file_writer(filename, signal, signal_metadata={}, user={},
                microscope={}, sample={}, comments={}, **kwds):
    if not signal.metadata.General.title:
        signal.metadata.General.title = '__unnamed__'
        print signal.metadata.General.title
    name = signal.metadata.General.as_dictionary().get('title', '__unnamed__')
    emd = EMD(user=user, microscope=microscope, sample=sample, comments=comments)
    emd.add_signal(signal, name, metadata=signal_metadata)
    emd.save_to_emd(filename)
