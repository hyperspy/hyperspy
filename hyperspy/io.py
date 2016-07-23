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
import glob
import logging

import numpy as np

import hyperspy.defaults_parser

import hyperspy.misc.utils
from hyperspy.misc.io.tools import ensure_directory
from hyperspy.misc.utils import strlist2enumeration
from natsort import natsorted
import hyperspy.misc.io.tools
from hyperspy.io_plugins import io_plugins, default_write_ext

_logger = logging.getLogger(__name__)


def load(filenames=None,
         signal_type=None,
         stack=False,
         stack_axis=None,
         new_axis_name="stack_element",
         mmap=False,
         mmap_dir=None,
         **kwds):
    """
    Load potentially multiple supported file into an hyperspy structure
    Supported formats: HDF5, msa, Gatan dm3, Ripple (rpl+raw), Bruker bcf,
    FEI ser and emi, hdf5, SEMPER unf, EMD, tif and a number of image formats.
    Any extra keyword is passed to the corresponsing reader. For
    available options see their individual documentation.
    Parameters
    ----------
    filenames :  None, str or list of strings
        The filename to be loaded. If None, a window will open to select
        a file to load. If a valid filename is passed in that single
        file is loaded. If multiple file names are passed in
        a list, a list of objects or a single object containing the data
        of the individual files stacked are returned. This behaviour is
        controlled by the `stack` parameter (see bellow). Multiple
        files can be loaded by using simple shell-style wildcards,
        e.g. 'my_file*.msa' loads all the files that starts
        by 'my_file' and has the '.msa' extension.
    signal_type : {None, "EELS", "EDS_TEM", "EDS_SEM", "", str}
        The acronym that identifies the signal type.
        The value provided may determine the Signal subclass assigned to the
        data.
        If None the value is read/guessed from the file. Any other value
        overrides the value stored in the file if any.
        For electron energy-loss spectroscopy use "EELS".
        For energy dispersive x-rays use "EDS_TEM"
        if acquired from an electron-transparent sample — as it is usually
        the case in a transmission electron  microscope (TEM) —,
        "EDS_SEM" if acquired from a non electron-transparent sample
        — as it is usually the case in a scanning electron  microscope (SEM) —.
        If "" (empty string) the value is not read from the file and is
        considered undefined.
    stack : bool
        If True and multiple filenames are passed in, stacking all
        the data into a single object is attempted. All files must match
        in shape. It is possible to store the data in a memory mapped
        temporary file instead of in memory setting mmap_mode. The title is set
        to the name of the folder containing the files.
    stack_axis : {None, int, str}
        If None, the signals are stacked over a new axis. The data must
        have the same dimensions. Otherwise the
        signals are stacked over the axis given by its integer index or
        its name. The data must have the same shape, except in the dimension
        corresponding to `axis`.
    new_axis_name : string
        The name of the new axis when `axis` is None.
        If an axis with this name already
        exists it automatically append '-i', where `i` are integers,
        until it finds a name that is not yet in use.
    mmap: bool
        If True and stack is True, then the data is stored
        in a memory-mapped temporary file.The memory-mapped data is
        stored on disk, and not directly loaded into memory.
        Memory mapping is especially useful for accessing small
        fragments of large files without reading the entire file into
        memory.
    mmap_dir : string
        If mmap_dir is not None, and stack and mmap are True, the memory
        mapped file will be created in the given directory,
        otherwise the default directory is used.
    load_to_memory: bool
        for HDF5 files, blockfiles and EMD files, if True (default) loads all
        data to memory. If False, enables only loading the data upon request
    mmap_mode: {'r', 'r+', 'c'}
        Used when loading blockfiles to determine which mode to use for when
        loading as memmap (i.e. when load_to_memory=False)

    print_info: bool
        For SEMPER unf- and EMD (Berkley)-files, if True (default is False)
        additional information read during loading is printed for a quick overview.


    Returns
    -------
    Signal instance or list of signal instances
    Examples
    --------
    Loading a single file providing the signal type:
    >>> d = hs.load('file.dm3', signal_type='EDS_TEM')

    Loading multiple files:
    >>> d = hs.load('file1.dm3','file2.dm3')
    Loading multiple files matching the pattern:
    >>> d = hs.load('file*.dm3')
    """
    kwds['signal_type'] = signal_type
    if filenames is None:
        if hyperspy.defaults_parser.preferences.General.interactive is True:
            from hyperspy.gui.tools import Load
            load_ui = Load()
            load_ui.edit_traits()
            if load_ui.filename:
                filenames = load_ui.filename
        else:
            raise ValueError("No file provided to reader and "
                             "interactive mode is disabled")
        if filenames is None:
            raise ValueError("No file provided to reader")

    if isinstance(filenames, str):
        filenames = natsorted([f for f in glob.glob(filenames)
                               if os.path.isfile(f)])
        if not filenames:
            raise ValueError('No file name matches this pattern')
    elif not isinstance(filenames, (list, tuple)):
        raise ValueError(
            'The filenames parameter must be a list, tuple, string or None')
    if not filenames:
        raise ValueError('No file provided to reader.')
    else:
        if len(filenames) > 1:
            _logger.info('Loading individual files')
        if stack is True:
            signal = []
            for i, filename in enumerate(filenames):
                obj = load_single_file(filename,
                                       **kwds)
                signal.append(obj)
            signal = hyperspy.misc.utils.stack(signal,
                                               axis=stack_axis,
                                               new_axis_name=new_axis_name,
                                               mmap=mmap, mmap_dir=mmap_dir)
            signal.metadata.General.title = \
                os.path.split(
                    os.path.split(
                        os.path.abspath(filenames[0])
                    )[0]
                )[1]
            _logger.info('Individual files loaded correctly')
            _logger.info(signal._summary())
            objects = [signal, ]
        else:
            objects = [load_single_file(filename,
                                        **kwds)
                       for filename in filenames]

        if hyperspy.defaults_parser.preferences.Plot.plot_on_load:
            for obj in objects:
                obj.plot()
        if len(objects) == 1:
            objects = objects[0]
    return objects


def load_single_file(filename,
                     signal_type=None,
                     **kwds):
    """
    Load any supported file into an HyperSpy structure
    Supported formats: netCDF, msa, Gatan dm3, Ripple (rpl+raw),
    Bruker bcf, FEI ser and emi, hdf5 and SEMPER unf.
    Parameters
    ----------
    filename : string
        File name (including the extension)

    """
    extension = os.path.splitext(filename)[1][1:]

    i = 0
    while extension.lower() not in io_plugins[i].file_extensions and \
            i < len(io_plugins) - 1:
        i += 1
    if i == len(io_plugins):
        # Try to load it with the python imaging library
        try:
            from hyperspy.io_plugins import image
            reader = image
            return load_with_reader(filename, reader,
                                    signal_type=signal_type, **kwds)
        except:
            raise IOError('If the file format is supported'
                          ' please report this error')
    else:
        reader = io_plugins[i]
        return load_with_reader(filename=filename,
                                reader=reader,
                                signal_type=signal_type,
                                **kwds)


def load_with_reader(filename,
                     reader,
                     signal_type=None,
                     **kwds):
    file_data_list = reader.file_reader(filename,
                                        **kwds)
    objects = []

    for signal_dict in file_data_list:
        if 'metadata' in signal_dict:
            if "Signal" not in signal_dict["metadata"]:
                signal_dict["metadata"]["Signal"] = {}
            if signal_type is not None:
                signal_dict['metadata']["Signal"]['signal_type'] = signal_type
            objects.append(dict2signal(signal_dict))
            folder, filename = os.path.split(os.path.abspath(filename))
            filename, extension = os.path.splitext(filename)
            objects[-1].tmp_parameters.folder = folder
            objects[-1].tmp_parameters.filename = filename
            objects[-1].tmp_parameters.extension = extension.replace('.', '')
        else:
            # it's a standalone model
            continue

    if len(objects) == 1:
        objects = objects[0]
    return objects


def assign_signal_subclass(dtype,
                           signal_dimension,
                           signal_type=""):
    """Given record_by and signal_type return the matching Signal subclass.
    Parameters
    ----------
    dtype : :class:`~.numpy.dtype`
    record_by: {"spectrum", "image", ""}
    signal_type : {"EELS", "EDS", "EDS_TEM", "", str}


    Returns
    -------
    Signal or subclass
    """
    import hyperspy.signals
    from hyperspy.signal import BaseSignal
    # Check if parameter values are allowed:
    if np.issubdtype(dtype, complex):
        dtype = 'complex'
    elif ('float' in dtype.name or 'int' in dtype.name or
          'void' in dtype.name or 'bool' in dtype.name):
        dtype = 'real'
    else:
        raise ValueError('Data type "{}" not understood!'.format(dtype.name))
    if not isinstance(signal_dimension, int) or signal_dimension < 0:
        raise ValueError("signal_dimension must be a positive interger")
    signals = hyperspy.misc.utils.find_subclasses(hyperspy.signals, BaseSignal)
    dtype_matches = [s for s in signals.values() if dtype == s._dtype]
    dtype_dim_matches = [s for s in dtype_matches
                         if signal_dimension == s._signal_dimension]
    dtype_dim_type_matches = [
        s for s in dtype_dim_matches if signal_type == s._signal_type]

    if dtype_dim_type_matches:
        # Perfect match found, return it.
        return dtype_dim_type_matches[0]
    elif [s for s in dtype_dim_matches if s._signal_type == ""]:
        # just signal_dimension and dtype matches
        # Return a general class for the given signal dimension.
        return [s for s in dtype_dim_matches if s._signal_type == ""][0]
    else:
        # no signal_dimension match either, hence return the general subclass for
        # correct dtype
        return [s for s in dtype_matches if s._signal_dimension ==
                -1 and s._signal_type == ""][0]


def dict2signal(signal_dict):
    """Create a signal (or subclass) instance defined by a dictionary
    Parameters
    ----------
    signal_dict : dictionary
    Returns
    -------
    s : Signal or subclass
    """
    signal_dimension = -1  # undefined
    signal_type = ""
    if "metadata" in signal_dict:
        mp = signal_dict["metadata"]
        if "Signal" in mp and "record_by" in mp["Signal"]:
            record_by = mp["Signal"]['record_by']
            if record_by == "spectrum":
                signal_dimension = 1
            elif record_by == "image":
                signal_dimension = 2
            del mp["Signal"]['record_by']
        if "Signal" in mp and "signal_type" in mp["Signal"]:
            signal_type = mp["Signal"]['signal_type']
    # "Estimate" signal_dimension from axes. It takes precedence over record_by
    if ("axes" in signal_dict and
        len(signal_dict["axes"]) == len(
            [axis for axis in signal_dict["axes"] if "navigate" in axis])):
            # If navigate is defined for all axes
        signal_dimension = len(
            [axis for axis in signal_dict["axes"] if not axis["navigate"]])
    elif signal_dimension == -1:
        # If not defined, all dimension are categorised as signal
        signal_dimension = signal_dict["data"].ndim

    signal = assign_signal_subclass(signal_dimension=signal_dimension,
                                    signal_type=signal_type,
                                    dtype=signal_dict['data'].dtype,)(**signal_dict)
    if signal.axes_manager.signal_dimension != signal_dimension:
        # This may happen when the signal dimension couldn't be matched with
        # any specialised subclass
        signal.axes_manager.set_signal_dimension(signal_dimension)
    if "post_process" in signal_dict:
        for f in signal_dict['post_process']:
            signal = f(signal)
    if "mapping" in signal_dict:
        for opattr, (mpattr, function) in signal_dict["mapping"].items():
            if opattr in signal.original_metadata:
                value = signal.original_metadata.get_item(opattr)
                if function is not None:
                    value = function(value)
                signal.metadata.set_item(mpattr, value)
    return signal


def save(filename, signal, overwrite=None, **kwds):
    extension = os.path.splitext(filename)[1][1:]
    if extension == '':
        extension = \
            hyperspy.defaults_parser.preferences.General.default_file_format
        filename = filename + '.' + \
            hyperspy.defaults_parser.preferences.General.default_file_format
    writer = None
    for plugin in io_plugins:
        if extension.lower() in plugin.file_extensions:
            writer = plugin
            break

    if writer is None:
        raise ValueError(
            ('.%s does not correspond to any supported format. Supported ' +
             'file extensions are: %s') %
            (extension, strlist2enumeration(default_write_ext)))
    else:
        # Check if the writer can write
        sd = signal.axes_manager.signal_dimension
        nd = signal.axes_manager.navigation_dimension
        if writer.writes is False:
            raise ValueError('Writing to this format is not '
                             'supported, supported file extensions are: %s ' %
                             strlist2enumeration(default_write_ext))
        if writer.writes is not True and (sd, nd) not in writer.writes:
            yes_we_can = [plugin.format_name for plugin in io_plugins
                          if plugin.writes is True or
                          plugin.writes is not False and
                          (sd, nd) in plugin.writes]
            raise ValueError('This file format cannot write this data. '
                             'The following formats can: %s' %
                             strlist2enumeration(yes_we_can))
        ensure_directory(filename)
        if overwrite is None:
            overwrite = hyperspy.misc.io.tools.overwrite(filename)
        if overwrite is True:
            writer.file_writer(filename, signal, **kwds)
            _logger.info('The %s file was created' % filename)
            folder, filename = os.path.split(os.path.abspath(filename))
            signal.tmp_parameters.set_item('folder', folder)
            signal.tmp_parameters.set_item('filename',
                                           os.path.splitext(filename)[0])
            signal.tmp_parameters.set_item('extension', extension)
