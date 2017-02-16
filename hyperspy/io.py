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
import warnings
import logging

import numpy as np
from natsort import natsorted

from .misc.io.tools import ensure_directory
from .misc.io.tools import overwrite as overwrite_method
from .misc.utils import (strlist2enumeration, find_subclasses)
from .misc.utils import stack as stack_method
from .io_plugins import io_plugins, default_write_ext
from .exceptions import VisibleDeprecationWarning
from .defaults_parser import preferences

_logger = logging.getLogger(__name__)


# Utility string:
f_error_fmt = (
    "\tFile %d:\n"
    "\t\t%d signals\n"
    "\t\tPath: %s")


def load(filenames=None,
         signal_type=None,
         stack=False,
         stack_axis=None,
         new_axis_name="stack_element",
         lazy=None,
         **kwds):
    """
    Load potentially multiple supported file into an hyperspy structure

    Supported formats: HDF5, msa, Gatan dm3, Ripple (rpl+raw), Bruker bcf,
    FEI ser and emi, hdf5, SEMPER unf, EMD, EDAX spd/spc, tif, and a number
    of image formats.

    Any extra keyword is passed to the corresponding reader. For
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
    signal_type : {None, "EELS", "EDS_SEM", "EDS_TEM", "", str}
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
        in shape. If each file contains multiple (N) signals, N stacks will be
        created, with the requirement that each file contains the same number
        of signals.
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
    lazy : {None, bool}
        Open the data lazily - i.e. without actually reading the data from the
        disk until required. Allows opening arbitrary-sized datasets.
        If None, default from preferences is used.

    print_info: bool
        For SEMPER unf- and EMD (Berkley)-files, if True (default is False)
        additional information read during loading is printed for a quick
        overview.

    Returns
    -------
    Signal instance or list of signal instances

    Examples
    --------
    Loading a single file providing the signal type:
    >>> d = hs.load('file.dm3', signal_type="EDS_TEM")

    Loading multiple files:

    >>> d = hs.load('file1.dm3','file2.dm3')

    Loading multiple files matching the pattern:

    >>> d = hs.load('file*.dm3')

    Loading (potentially larger than the available memory) files lazily and
    stacking:

    >>> s = hs.load('file*.blo', lazy=True, stack=True)

    """
    deprecated = ['mmap_dir', 'load_to_memory']
    warn_str = "'{}' argument is deprecated, please use 'lazy' instead"
    for k in deprecated:
        if k in kwds:
            lazy=True
            warnings.warn(warn_str.format(k), VisibleDeprecationWarning)
            del kwds[k]
    if lazy is None:
        lazy = preferences.General.lazy
    kwds['signal_type'] = signal_type

    if filenames is None:
        if preferences.General.interactive is True:
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
            # We are loading a stack!
            # Note that while each file might contain several signals, all
            # files are required to contain the same number of signals. We
            # therefore use the first file to determine the number of signals.
            for i, filename in enumerate(filenames):
                obj = load_single_file(filename, lazy=lazy,
                                       **kwds)
                if i == 0:
                    # First iteration, determine number of signals, if several:
                    if isinstance(obj, (list, tuple)):
                        n = len(obj)
                    else:
                        n = 1
                    # Initialize signal 2D list:
                    signals = [[] for j in range(n)]
                else:
                    # Check that number of signals per file doesn't change
                    # for other files:
                    if isinstance(obj, (list, tuple)):
                        if n != len(obj):
                            raise ValueError(
                                "The number of sub-signals per file does not "
                                "match:\n" +
                                (f_error_fmt % (1, n, filenames[0])) +
                                (f_error_fmt % (i, len(obj), filename)))
                    elif n != 1:
                        raise ValueError(
                            "The number of sub-signals per file does not "
                            "match:\n" +
                            (f_error_fmt % (1, n, filenames[0])) +
                            (f_error_fmt % (i, len(obj), filename)))
                # Append loaded signals to 2D list:
                if n == 1:
                    signals[0].append(obj)
                elif n > 1:
                    for j in range(n):
                        signals[j].append(obj[j])
            # Next, merge the signals in the `stack_axis` direction:
            # When each file had N signals, we create N stacks!
            objects = []
            for i in range(n):
                signal = signals[i]   # Sublist, with len = len(filenames)
                signal = stack_method(
                    signal, axis=stack_axis, new_axis_name=new_axis_name,
                    lazy=lazy)
                signal.metadata.General.title = os.path.split(
                    os.path.split(os.path.abspath(filenames[0]))[0])[1]
                _logger.info('Individual files loaded correctly')
                _logger.info(signal._summary())
                objects.append(signal)
        else:
            # No stack, so simply we load all signals in all files separately
            objects = [load_single_file(filename, lazy=lazy,
                                        **kwds)
                       for filename in filenames]

        if preferences.Plot.plot_on_load:
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
    Bruker bcf, FEI ser and emi, EDAX spc and spd, hdf5, and SEMPER unf.

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
    lazy = kwds.get('lazy', False)
    file_data_list = reader.file_reader(filename,
                                        **kwds)
    objects = []

    for signal_dict in file_data_list:
        if 'metadata' in signal_dict:
            if "Signal" not in signal_dict["metadata"]:
                signal_dict["metadata"]["Signal"] = {}
            if signal_type is not None:
                signal_dict['metadata']["Signal"]['signal_type'] = signal_type
            objects.append(dict2signal(signal_dict, lazy=lazy))
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
                           signal_type="",
                           lazy=False):
    """Given record_by and signal_type return the matching Signal subclass.

    Parameters
    ----------
    dtype : :class:`~.numpy.dtype`
    signal_dimension: int
    signal_type : {"EELS", "EDS", "EDS_SEM", "EDS_TEM", "DielectricFunction", "", str}
    lazy: bool

    Returns
    -------
    Signal or subclass

    """
    import hyperspy.signals
    import hyperspy._lazy_signals
    from hyperspy.signal import BaseSignal
    # Check if parameter values are allowed:
    if np.issubdtype(dtype, complex):
        dtype = 'complex'
    elif ('float' in dtype.name or 'int' in dtype.name or
          'void' in dtype.name or 'bool' in dtype.name or
          'object' in dtype.name):
        dtype = 'real'
    else:
        raise ValueError('Data type "{}" not understood!'.format(dtype.name))
    if not isinstance(signal_dimension, int) or signal_dimension < 0:
        raise ValueError("signal_dimension must be a positive interger")
    base_signals = find_subclasses(hyperspy.signals, BaseSignal)
    lazy_signals = find_subclasses(hyperspy._lazy_signals,
                                   hyperspy._lazy_signals.LazySignal)
    if lazy:
        signals = lazy_signals
    else:
        signals = {
            k: v for k,
            v in base_signals.items() if k not in lazy_signals}
    dtype_matches = [s for s in signals.values() if dtype == s._dtype]
    dtype_dim_matches = [s for s in dtype_matches
                         if signal_dimension == s._signal_dimension]
    dtype_dim_type_matches = [s for s in dtype_dim_matches if signal_type == s._signal_type
                              or signal_type in s._alias_signal_types]
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
        return [s for s in dtype_matches if s._signal_dimension == -
                1 and s._signal_type == ""][0]


def dict2signal(signal_dict, lazy=False):
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
    if "attributes" in signal_dict and "_lazy" in signal_dict["attributes"]:
        lazy = signal_dict["attributes"]["_lazy"]
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
                                    dtype=signal_dict['data'].dtype,
                                    lazy=lazy)(**signal_dict)
    if signal._lazy:
        signal._make_lazy()
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
                if value is not None:
                    signal.metadata.set_item(mpattr, value)
    return signal


def save(filename, signal, overwrite=None, **kwds):
    extension = os.path.splitext(filename)[1][1:]
    if extension == '':
        extension = \
            preferences.General.default_file_format
        filename = filename + '.' + \
            preferences.General.default_file_format
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
            raise IOError('This file format cannot write this data. '
                          'The following formats can: %s' %
                          strlist2enumeration(yes_we_can))
        ensure_directory(filename)
        if overwrite is None:
            overwrite = overwrite_method(filename)
        if overwrite is True:
            writer.file_writer(filename, signal, **kwds)
            _logger.info('The %s file was created' % filename)
            folder, filename = os.path.split(os.path.abspath(filename))
            signal.tmp_parameters.set_item('folder', folder)
            signal.tmp_parameters.set_item('filename',
                                           os.path.splitext(filename)[0])
            signal.tmp_parameters.set_item('extension', extension)
