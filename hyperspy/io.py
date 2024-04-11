# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import glob
import importlib
import logging
import os
import warnings
from collections.abc import MutableMapping
from datetime import datetime
from inspect import isgenerator
from pathlib import Path

import numpy as np
from natsort import natsorted
from rsciio import IO_PLUGINS
from rsciio.utils.tools import ensure_directory
from rsciio.utils.tools import overwrite as overwrite_method

from hyperspy.api import __version__ as hs_version
from hyperspy.docstrings.signal import SHOW_PROGRESSBAR_ARG
from hyperspy.docstrings.utils import STACK_METADATA_ARG
from hyperspy.drawing.markers import markers_dict_to_markers
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.extensions import ALL_EXTENSIONS
from hyperspy.misc.utils import get_object_package_info, strlist2enumeration
from hyperspy.misc.utils import stack as stack_method
from hyperspy.ui_registry import get_gui

_logger = logging.getLogger(__name__)


# Utility string:
f_error_fmt = "\tFile %d:\n" "\t\t%d signals\n" "\t\tPath: %s"


def _format_name_to_reader(format_name):
    for reader in IO_PLUGINS:
        if format_name.lower() == reader["name"].lower():
            return reader
        elif reader.get("name_aliases"):
            aliases = [s.lower() for s in reader["name_aliases"]]
            if format_name.lower() in aliases:
                return reader
    raise ValueError("The `format_name` given does not match any format available.")


def _infer_file_reader(string):
    """Return a file reader from the plugins list based on the format name or
    the file extension.

    If the extension is not found or understood, returns
    the Python imaging library as the file reader.

    Parameters
    ----------
    string : str
        File extension, without initial "." separator

    Returns
    -------
    reader : func
        The inferred file reader.

    """
    try:
        reader = _format_name_to_reader(string)
        return reader
    except ValueError:
        pass

    rdrs = [rdr for rdr in IO_PLUGINS if string.lower() in rdr["file_extensions"]]

    if not rdrs:
        # Try to load it with the python imaging library
        _logger.warning(
            f"Unable to infer file type from extension '{string}'. "
            "Will attempt to load the file with the Python imaging library."
        )

        (reader,) = [
            reader for reader in IO_PLUGINS if reader["name"].lower() == "image"
        ]
    elif len(rdrs) > 1:
        names = [rdr["name"] for rdr in rdrs]
        raise ValueError(
            f"There are multiple file readers that could read the file. "
            f"Please select one from the list below with the `reader` keyword. "
            f"File readers for your file: {names}"
        )
    else:
        reader = rdrs[0]

    return reader


def _infer_file_writer(string):
    """Return a file reader from the plugins list based on the file extension.

    If the extension is not found or understood, returns
    the Python imaging library as the file reader.

    Parameters
    ----------
    string : str
        File extension, without initial "." separator

    Returns
    -------
    reader : func
        The inferred file reader.

    """
    plugins = [
        plugin for plugin in IO_PLUGINS if string.lower() in plugin["file_extensions"]
    ]
    writers = [plugin for plugin in plugins if plugin["writes"]]
    if not writers:
        extensions = [
            plugin["file_extensions"][plugin["default_extension"]]
            for plugin in IO_PLUGINS
            if plugin["writes"]
        ]
        if not plugins:
            raise ValueError(
                f"The .{string} extension does not correspond to any supported format. "
                f"Supported file extensions are: {strlist2enumeration(extensions)}."
            )
        else:
            raise ValueError(
                "Writing to this format is not supported. "
                f"Supported file extensions are: {strlist2enumeration(extensions)}."
            )

    elif len(writers) > 1:
        names = [writer["name"] for writer in writers]
        raise ValueError(
            f"There are multiple file formats matching the extension of your file. "
            f"Please select one from the list below with the `format` keyword. "
            f"File formats for your file: {names}"
        )
    else:
        writer = writers[0]

    return writer


def _escape_square_brackets(text):
    """Escapes pairs of square brackets in strings for glob.glob().

    Parameters
    ----------
    text : str
        The text to escape

    Returns
    -------
    str
        The escaped string

    Examples
    --------
    >>> # Say there are two files like this:
    >>> # /home/data/afile[1x1].txt
    >>> # /home/data/afile[1x2].txt
    >>>
    >>> path = "/home/data/afile[*].txt"
    >>> glob.glob(path)
    []
    >>> glob.glob(_escape_square_brackets(path)) # doctest: +SKIP
    ['/home/data/afile[1x2].txt', '/home/data/afile[1x1].txt']

    """
    import re

    rep = dict((re.escape(k), v) for k, v in {"[": "[[]", "]": "[]]"}.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)


def _parse_path(arg):
    """Convenience function to get the path from zarr store or string."""
    # In case of zarr store, get the path
    if isinstance(arg, MutableMapping):
        fname = arg.path
    else:
        fname = arg

    return fname


def load(
    filenames=None,
    signal_type=None,
    stack=False,
    stack_axis=None,
    new_axis_name="stack_element",
    lazy=False,
    convert_units=False,
    escape_square_brackets=False,
    stack_metadata=True,
    load_original_metadata=True,
    show_progressbar=None,
    **kwds,
):
    """Load potentially multiple supported files into HyperSpy.

    Supported formats: hspy (HDF5), msa, Gatan dm3, Ripple (rpl+raw),
    Bruker bcf and spx, FEI ser and emi, SEMPER unf, EMD, EDAX spd/spc, CEOS prz
    tif, and a number of image formats.

    Depending on the number of datasets to load in the file, this function will
    return a HyperSpy signal instance or list of HyperSpy signal instances.

    Any extra keywords are passed to the corresponding reader. For
    available options, see their individual documentation.

    Parameters
    ----------
    filenames :  None, (list of) str or (list of) pathlib.Path, default None
        The filename to be loaded. If None, a window will open to select
        a file to load. If a valid filename is passed, that single
        file is loaded. If multiple file names are passed in
        a list, a list of objects or a single object containing multiple
        datasets, a list of signals or a stack of signals is returned. This
        behaviour is controlled by the `stack` parameter (see below). Multiple
        files can be loaded by using simple shell-style wildcards,
        e.g. 'my_file*.msa' loads all the files that start
        by 'my_file' and have the '.msa' extension. Alternatively, regular
        expression type character classes can be used (e.g. ``[a-z]`` matches
        lowercase letters). See also the `escape_square_brackets` parameter.
    signal_type : None, str, default None
        The acronym that identifies the signal type. May be any signal type
        provided by HyperSpy or by installed extensions as listed by
        `hs.print_known_signal_types()`. The value provided may determines the
        Signal subclass assigned to the data.
        If None (default), the value is read/guessed from the file.
        Any other value would override the value potentially stored in the file.
        For example, for electron energy-loss spectroscopy use 'EELS'.
        If '' (empty string) the value is not read from the file and is
        considered undefined.
    stack : bool, default False
        Default False. If True and multiple filenames are passed, stacking all
        the data into a single object is attempted. All files must match
        in shape. If each file contains multiple (N) signals, N stacks will be
        created, with the requirement that each file contains the same number
        of signals.
    stack_axis : None, int or str, default None
        If None (default), the signals are stacked over a new axis. The data
        must have the same dimensions. Otherwise, the signals are stacked over
        the axis given by its integer index or its name. The data must have the
        same shape, except in the dimension corresponding to `axis`.
    new_axis_name : str, optional
        The name of the new axis (default 'stack_element'), when `axis` is None.
        If an axis with this name already exists, it automatically appends '-i',
        where `i` are integers, until it finds a name that is not yet in use.
    lazy : bool, default False
        Open the data lazily - i.e. without actually reading the data from the
        disk until required. Allows opening arbitrary-sized datasets.
    convert_units : bool, default False
        If True, convert the units using the `convert_to_units` method of
        the `axes_manager`. If False, does nothing.
    escape_square_brackets : bool, default False
        If True, and ``filenames`` is a str containing square brackets,
        then square brackets are escaped before wildcard matching with
        ``glob.glob()``. If False, square brackets are used to represent
        character classes (e.g. ``[a-z]`` matches lowercase letters).
    %s
    %s Only used with ``stack=True``.
    load_original_metadata : bool, default True
        If ``True``, all metadata contained in the input file will be added
        to ``original_metadata``.
        This does not affect parsing the metadata to ``metadata``.
    reader : None, str, module, optional
        Specify the file reader to use when loading the file(s). If None
        (default), will use the file extension to infer the file type and
        appropriate reader. If str, will select the appropriate file reader
        from the list of available readers in HyperSpy. If module, it must
        implement the ``file_reader`` function, which returns
        a dictionary containing the data and metadata for conversion to
        a HyperSpy signal.
    print_info: bool, optional
        For SEMPER unf- and EMD (Berkeley)-files. If True, additional
        information read during loading is printed for a quick overview.
        Default False.
    downsample : int (1–4095), optional
        For Bruker bcf files, if set to integer (>=2) (default 1),
        bcf is parsed into down-sampled size array by given integer factor,
        multiple values from original bcf pixels are summed forming downsampled
        pixel. This allows to improve signal and conserve the memory with the
        cost of lower resolution.
    cutoff_at_kV : None, int, float, optional
        For Bruker bcf files and Jeol, if set to numerical (default is None),
        hypermap is parsed into array with depth cutoff at set energy value.
        This allows to conserve the memory by cutting-off unused spectral
        tails, or force enlargement of the spectra size.
        Bruker bcf reader accepts additional values for semi-automatic cutoff.
        "zealous" value truncates to the last non zero channel (this option
        should not be used for stacks, as low beam current EDS can have different
        last non zero channel per slice).
        "auto" truncates channels to SEM/TEM acceleration voltage or
        energy at last channel, depending which is smaller.
        In case the hv info is not there or hv is off (0 kV) then it fallbacks to
        full channel range.
    select_type : 'spectrum_image', 'image', 'single_spectrum', None, optional
        If None (default), all data are loaded.
        For Bruker bcf and Velox emd files: if one of 'spectrum_image', 'image'
        or 'single_spectrum', the loader returns either only the spectrum image,
        only the images (including EDS map for Velox emd files), or only
        the single spectra (for Velox emd files).
    first_frame : int, optional
        Only for Velox emd files: load only the data acquired after the
        specified fname. Default 0.
    last_frame : None, int, optional
        Only for Velox emd files: load only the data acquired up to specified
        fname. If None (default), load the data up to the end.
    sum_frames : bool, optional
        Only for Velox emd files: if False, load each EDS frame individually.
        Default is True.
    sum_EDS_detectors : bool, optional
        Only for Velox emd files: if True (default), the signals from the
        different detectors are summed. If False, a distinct signal is returned
        for each EDS detectors.
    rebin_energy : int, optional
        Only for Velox emd files: rebin the energy axis by the integer provided
        during loading in order to save memory space. Needs to be a multiple of
        the length of the energy dimension (default 1).
    SI_dtype : numpy.dtype, None, optional
        Only for Velox emd files: set the dtype of the spectrum image data in
        order to save memory space. If None, the default dtype from the Velox emd
        file is used.
    load_SI_image_stack : bool, optional
        Only for Velox emd files: if True, load the stack of STEM images
        acquired simultaneously as the EDS spectrum image. Default is False.
    dataset_path : None, str, list of str, optional
        For filetypes which support several datasets in the same file, this
        will only load the specified dataset. Several datasets can be loaded
        by using a list of strings. Only for EMD (NCEM) and hdf5 (USID) files.
    stack_group : bool, optional
        Only for EMD NCEM. Stack datasets of groups with common name. Relevant
        for emd file version >= 0.5 where groups can be named 'group0000',
        'group0001', etc.
    ignore_non_linear_dims : bool, optional
        Only for HDF5 USID files: if True (default), parameters that were varied
        non-linearly in the desired dataset will result in Exceptions.
        Else, all such non-linearly varied parameters will be treated as
        linearly varied parameters and a Signal object will be generated.
    only_valid_data : bool, optional
        Only for FEI emi/ser files in case of series or linescan with the
        acquisition stopped before the end: if True, load only the acquired
        data. If False, fill empty data with zeros. Default is False and this
        default value will change to True in version 2.0.

    Returns
    -------
    (list of) :class:`~.api.signals.BaseSignal` or subclass

    Examples
    --------
    Loading a single file providing the signal type:

    >>> d = hs.load('file.dm3', signal_type="EDS_TEM") # doctest: +SKIP

    Loading multiple files:

    >>> d = hs.load(['file1.hspy','file2.hspy']) # doctest: +SKIP

    Loading multiple files matching the pattern:

    >>> d = hs.load('file*.hspy') # doctest: +SKIP

    Loading multiple files containing square brackets in the filename:

    >>> d = hs.load('file[*].hspy', escape_square_brackets=True) # doctest: +SKIP

    Loading multiple files containing character classes (regular expression):

    >>> d = hs.load('file[0-9].hspy')  # doctest: +SKIP

    Loading (potentially larger than the available memory) files lazily and
    stacking:

    >>> s = hs.load('file*.blo', lazy=True, stack=True) # doctest: +SKIP

    Specify the file reader to use

    >>> s = hs.load('a_nexus_file.h5', reader='nxs') # doctest: +SKIP

    Loading a file containing several datasets:

    >>> s = hs.load("spameggsandham.nxs") # doctest: +SKIP
    >>> s # doctest: +SKIP
    [<Signal1D, title: spam, dimensions: (32,32|1024)>,
     <Signal1D, title: eggs, dimensions: (32,32|1024)>,
     <Signal1D, title: ham, dimensions: (32,32|1024)>]

    Use list indexation to access single signal

    >>> s[0] # doctest: +SKIP
    <Signal1D, title: spam, dimensions: (32,32|1024)>

    """

    kwds["signal_type"] = signal_type
    kwds["convert_units"] = convert_units
    kwds["load_original_metadata"] = load_original_metadata
    if filenames is None:
        from hyperspy.signal_tools import Load

        load_ui = Load()
        get_gui(load_ui, toolkey="hyperspy.load")
        if load_ui.filename:
            filenames = load_ui.filename
            lazy = load_ui.lazy
        if filenames is None:
            raise ValueError("No file provided to reader")

    pattern = None
    if isinstance(filenames, str):
        pattern = filenames
        if escape_square_brackets:
            filenames = _escape_square_brackets(filenames)

        filenames = natsorted(
            [
                f
                for f in glob.glob(filenames)
                if os.path.isfile(f)
                or (os.path.isdir(f) and os.path.splitext(f)[1] == ".zspy")
            ]
        )

    elif isinstance(filenames, Path):
        pattern = filenames
        # Just convert to list for now, pathlib.Path not
        # fully supported in io_plugins
        filenames = [
            f for f in [filenames] if f.is_file() or (f.is_dir() and ".zspy" in f.name)
        ]

    elif isgenerator(filenames):
        filenames = list(filenames)

    elif not isinstance(filenames, (list, tuple, MutableMapping)):
        raise ValueError(
            "The filenames parameter must be a list, tuple, "
            f"string or None, not {type(filenames)}"
        )

    if not filenames:
        # in case, the file doesn't exist
        raise ValueError(f'No filename matches the pattern "{pattern}"')

    if isinstance(filenames, MutableMapping):
        filenames = [filenames]
    else:
        # pathlib.Path not fully supported in io_plugins,
        # so convert to str here to maintain compatibility
        filenames = [str(f) if isinstance(f, Path) else f for f in filenames]

    if len(filenames) > 1:
        _logger.info("Loading individual files")

    if stack is True:
        # We are loading a stack!
        # Note that while each file might contain several signals, all
        # files are required to contain the same number of signals. We
        # therefore use the first file to determine the number of signals.
        for i, filename in enumerate(filenames):
            obj = load_single_file(filename, lazy=lazy, **kwds)

            if i == 0:
                # First iteration, determine number of signals, if several:
                n = len(obj) if isinstance(obj, (list, tuple)) else 1

                # Initialize signal 2D list:
                signals = [[] for j in range(n)]
            else:
                # Check that number of signals per file doesn't change
                # for other files:
                if isinstance(obj, (list, tuple)):
                    if n != len(obj):
                        raise ValueError(
                            "The number of sub-signals per file does not match:\n"
                            + (f_error_fmt % (1, n, filenames[0]))
                            + (f_error_fmt % (i, len(obj), filename))
                        )
                elif n != 1:
                    raise ValueError(
                        "The number of sub-signals per file does not match:\n"
                        + (f_error_fmt % (1, n, filenames[0]))
                        + (f_error_fmt % (i, len(obj), filename))
                    )

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
            signal = signals[i]  # Sublist, with len = len(filenames)
            signal = stack_method(
                signal,
                axis=stack_axis,
                new_axis_name=new_axis_name,
                lazy=lazy,
                stack_metadata=stack_metadata,
                show_progressbar=show_progressbar,
            )
            signal.metadata.General.title = Path(filenames[0]).parent.stem
            _logger.info("Individual files loaded correctly")
            _logger.info(signal._summary())
            objects.append(signal)
    else:
        # No stack, so simply we load all signals in all files separately
        objects = [
            load_single_file(filename, lazy=lazy, **kwds) for filename in filenames
        ]

    if len(objects) == 1:
        objects = objects[0]

    return objects


load.__doc__ %= (STACK_METADATA_ARG, SHOW_PROGRESSBAR_ARG)


def load_single_file(filename, **kwds):
    """Load any supported file into an HyperSpy structure.

    Supported formats: netCDF, msa, Gatan dm3, Ripple (rpl+raw),
    Bruker bcf, FEI ser and emi, EDAX spc and spd, hspy (HDF5), and SEMPER unf.

    Parameters
    ----------
    filename : string
        File name including the extension.
    **kwds
        Keyword arguments passed to specific file reader.

    Returns
    -------
    object
        Data loaded from the file.

    """
    # in case filename is a zarr store, we want to the path and not the store
    path = _parse_path(filename)

    if not os.path.isfile(path) and not (
        os.path.isdir(path) and os.path.splitext(path)[1] == ".zspy"
    ):
        raise FileNotFoundError(f"File: {path} not found!")

    # File extension without "." separator
    file_ext = os.path.splitext(path)[1][1:]
    reader = kwds.pop("reader", None)

    if reader is None:
        # Infer file reader based on extension
        reader = _infer_file_reader(file_ext)
    elif isinstance(reader, str):
        # Infer file reader based on provided kwarg string
        reader = _infer_file_reader(reader)
    elif hasattr(reader, "file_reader"):
        # Implies the user has passed their own file reader
        pass
    else:
        raise ValueError(
            "`reader` should be one of None, str, " "or a custom file reader object"
        )

    try:
        # Try and load the file
        return load_with_reader(filename=filename, reader=reader, **kwds)

    except BaseException:
        _logger.error(
            "If this file format is supported, please "
            "report this error to the HyperSpy developers."
        )
        raise


def load_with_reader(
    filename,
    reader,
    signal_type=None,
    convert_units=False,
    load_original_metadata=True,
    **kwds,
):
    """Load a supported file with a given reader."""
    lazy = kwds.get("lazy", False)
    if isinstance(reader, dict):
        file_data_list = importlib.import_module(reader["api"]).file_reader(
            filename, **kwds
        )
    else:
        # We assume it is a module
        file_data_list = reader.file_reader(filename, **kwds)
    signal_list = []

    for signal_dict in file_data_list:
        if "metadata" in signal_dict:
            if "Signal" not in signal_dict["metadata"]:
                signal_dict["metadata"]["Signal"] = {}
            if signal_type is not None:
                signal_dict["metadata"]["Signal"]["signal_type"] = signal_type
            signal = dict2signal(signal_dict, lazy=lazy)
            signal = _add_file_load_save_metadata("load", signal, reader)
            path = _parse_path(filename)
            folder, filename = os.path.split(os.path.abspath(path))
            filename, extension = os.path.splitext(filename)
            signal.tmp_parameters.folder = folder
            signal.tmp_parameters.filename = filename
            signal.tmp_parameters.extension = extension.replace(".", "")
            # original_filename and original_file are used to keep track of
            # where is the file which has been open lazily
            signal.tmp_parameters.original_folder = folder
            signal.tmp_parameters.original_filename = filename
            signal.tmp_parameters.original_extension = extension.replace(".", "")
            # test if binned attribute is still in metadata
            if signal.metadata.has_item("Signal.binned"):
                for axis in signal.axes_manager.signal_axes:
                    axis.is_binned = signal.metadata.Signal.binned
                del signal.metadata.Signal.binned
                warnings.warn(
                    "Loading old file version. The binned attribute "
                    "has been moved from metadata.Signal to "
                    "axis.is_binned. Setting this attribute for all "
                    "signal axes instead.",
                    VisibleDeprecationWarning,
                )
            if convert_units:
                signal.axes_manager.convert_units()
            if not load_original_metadata:
                signal._original_metadata = type(signal.original_metadata)()
            signal_list.append(signal)
        else:
            # it's a standalone model
            continue

    if len(signal_list) == 1:
        signal_list = signal_list[0]

    return signal_list


def assign_signal_subclass(dtype, signal_dimension, signal_type="", lazy=False):
    """Given dtype, signal_dimension and signal_type, return the matching
    Signal subclass.

    See `hs.print_known_signal_types()` for a list of known signal_types,
    and the developer guide for details on how to add new signal_types.

    Parameters
    ----------
    dtype : :class:`~.numpy.dtype`
        Signal dtype
    signal_dimension : int
        Signal dimension
    signal_type : str, optional
        Signal type. Default ''. Will log a warning if it is unknown to HyperSpy.
    lazy : bool, optional
        If True, returns the matching LazySignal subclass. Default is False.

    Returns
    -------
    Signal class or subclass

    """
    # Check if parameter values are allowed:
    if np.issubdtype(dtype, np.complexfloating):
        dtype = "complex"
    elif (
        "float" in dtype.name
        or "int" in dtype.name
        or "void" in dtype.name
        or "bool" in dtype.name
        or "object" in dtype.name
    ):
        dtype = "real"
    else:
        raise ValueError(f'Data type "{dtype.name}" not understood!')
    if not isinstance(signal_dimension, int) or signal_dimension < 0:
        raise ValueError("signal_dimension must be a positive integer")

    signals = {
        key: value
        for key, value in ALL_EXTENSIONS["signals"].items()
        if value["lazy"] == lazy
    }
    dtype_matches = {
        key: value for key, value in signals.items() if value["dtype"] == dtype
    }
    dtype_dim_matches = {
        key: value
        for key, value in dtype_matches.items()
        if signal_dimension == value["signal_dimension"]
    }
    dtype_dim_type_matches = {
        key: value
        for key, value in dtype_dim_matches.items()
        if signal_type == value["signal_type"]
        or "signal_type_aliases" in value
        and signal_type in value["signal_type_aliases"]
    }

    valid_signal_types = [v["signal_type"] for v in signals.values()]
    valid_signal_aliases = [
        v["signal_type_aliases"] for v in signals.values() if "signal_type_aliases" in v
    ]
    valid_signal_aliases = [i for j in valid_signal_aliases for i in j]
    valid_signal_types.extend(valid_signal_aliases)

    if dtype_dim_type_matches:
        # Perfect match found
        signal_dict = dtype_dim_type_matches
    else:
        if signal_type not in set(valid_signal_types):
            _logger.warning(
                f"`signal_type='{signal_type}'` not understood. "
                "See `hs.print_known_signal_types()` for a list of installed "
                "signal types or https://github.com/hyperspy/hyperspy-extensions-list "
                "for the list of all hyperspy extensions providing signals."
            )

        # If the following dict is not empty, only signal_dimension and dtype match.
        # The dict should contain a general class for the given signal
        # dimension.
        signal_dict = {
            key: value
            for key, value in dtype_dim_matches.items()
            if value["signal_type"] == ""
        }
        if not signal_dict:
            # no signal_dimension match either, hence select the general subclass for
            # correct dtype
            signal_dict = {
                key: value
                for key, value in dtype_matches.items()
                if value["signal_dimension"] == -1 and value["signal_type"] == ""
            }
    # Sanity check
    if len(signal_dict) > 1:
        _logger.warning(
            "There is more than one kind of signal that matches "
            "the current specifications. This is unexpected behaviour. "
            "Please report this issue to the HyperSpy developers."
        )

    # Regardless of the number of signals in the dict we assign one.
    # The following should only raise an error if the base classes
    # are not correctly registered.
    for key, value in signal_dict.items():
        signal_class = getattr(importlib.import_module(value["module"]), key)

        return signal_class


def dict2signal(signal_dict, lazy=False):
    """Create a signal (or subclass) instance defined by a dictionary.

    Parameters
    ----------
    signal_dict : dictionary

    Returns
    -------
    s : Signal or subclass

    """
    if "package" in signal_dict and signal_dict["package"]:
        try:
            importlib.import_module(signal_dict["package"])
        except ImportError:
            _logger.warning(
                "This file contains a signal provided by the "
                f'{signal_dict["package"]} Python package that is not '
                'currently installed. The signal will be loaded into a '
                'generic HyperSpy signal. Consider installing '
                f'{signal_dict["package"]} to load this dataset into its '
                'original signal class.'
            )
    signal_dimension = -1  # undefined
    signal_type = ""
    mp = signal_dict.get("metadata")

    if mp is not None:
        if "Signal" in mp and "record_by" in mp["Signal"]:
            record_by = mp["Signal"]["record_by"]
            if record_by == "spectrum":
                signal_dimension = 1
            elif record_by == "image":
                signal_dimension = 2
            del mp["Signal"]["record_by"]
        if "Signal" in mp and "signal_type" in mp["Signal"]:
            signal_type = mp["Signal"]["signal_type"]
    if "attributes" in signal_dict and "_lazy" in signal_dict["attributes"]:
        lazy = signal_dict["attributes"]["_lazy"]
    # "Estimate" signal_dimension from axes. It takes precedence over record_by
    if "axes" in signal_dict and len(signal_dict["axes"]) == len(
        [axis for axis in signal_dict["axes"] if "navigate" in axis]
    ):
        # If navigate is defined for all axes
        signal_dimension = len(
            [axis for axis in signal_dict["axes"] if not axis["navigate"]]
        )
    elif signal_dimension == -1:
        # If not defined, all dimension are categorised as signal
        signal_dimension = signal_dict["data"].ndim
    signal = assign_signal_subclass(
        signal_dimension=signal_dimension,
        signal_type=signal_type,
        dtype=signal_dict["data"].dtype,
        lazy=lazy,
    )(**signal_dict)
    if signal._lazy:
        signal._make_lazy()

    # This may happen when the signal dimension couldn't be matched with
    # any specialised subclass
    signal.axes_manager._set_signal_dimension(signal_dimension)

    if "post_process" in signal_dict:
        for f in signal_dict["post_process"]:
            signal = f(signal)
    if "mapping" in signal_dict:
        for opattr, (mpattr, function) in signal_dict["mapping"].items():
            if opattr in signal.original_metadata:
                value = signal.original_metadata.get_item(opattr)
                if function is not None:
                    value = function(value)
                if value is not None:
                    signal.metadata.set_item(mpattr, value)
    if mp is not None and "Markers" in mp:
        for key in mp["Markers"]:
            signal.metadata.Markers[key] = markers_dict_to_markers(mp["Markers"][key])
            signal.metadata.Markers[key]._signal = signal

    return signal


def save(filename, signal, overwrite=None, file_format=None, **kwds):
    """Save hyperspy signal to a file.

    A list of plugins supporting file saving can be found here:
    http://hyperspy.org/hyperspy-doc/current/user_guide/io.html#supported-formats

    Any extra keywords are passed to the corresponding save method in the
    io_plugin. For available options, see their individual documentation.

    Parameters
    ----------
    filename : None, str, pathlib.Path
        The filename to save the signal to.
    signal : Hyperspy signal
        The signal to be saved to the file.
    overwrite : None, bool, optional
        If None (default) and a file exists, the user will be prompted whether
        to overwrite. If False and a file exists, the file will not be written.
        If True and a file exists, the file will be overwritten without
        prompting
    file_format: string
        The file format of choice to save the file. If not given, it is inferred
        from the file extension.

    Returns
    -------
    None

    """
    writer = None
    if isinstance(filename, MutableMapping):
        extension = ".zspy"
        writer = _format_name_to_reader("ZSPY")
    else:
        filename = Path(filename).resolve()
        extension = filename.suffix
        if extension == "":
            if file_format:
                writer = _format_name_to_reader(file_format)
                extension = "." + writer["file_extensions"][writer["default_extension"]]
            else:
                extension = ".hspy"
                writer = _format_name_to_reader("HSPY")
            filename = filename.with_suffix(extension)
        else:
            if file_format:
                writer = _format_name_to_reader(file_format)
            else:
                writer = _infer_file_writer(extension[1:])

    # Check if the writer can write
    sd = signal.axes_manager.signal_dimension
    nd = signal.axes_manager.navigation_dimension

    if writer["writes"] is not True and [sd, nd] not in writer["writes"]:
        compatible_writers = [
            plugin["name"]
            for plugin in IO_PLUGINS
            if plugin["writes"] is True
            or plugin["writes"] is not False
            and [sd, nd] in plugin["writes"]
        ]

        raise TypeError(
            "This file format does not support this data. "
            f"Please try one of {strlist2enumeration(compatible_writers)}"
        )

    if not writer["non_uniform_axis"] and not signal.axes_manager.all_uniform:
        compatible_writers = [
            plugin["name"]
            for plugin in IO_PLUGINS
            if plugin["non_uniform_axis"] is True
        ]
        raise TypeError(
            "Writing to this format is not supported for "
            "non-uniform axes. Use one of the following "
            f"formats: {strlist2enumeration(compatible_writers)}"
        )

    # Create the directory if it does not exist
    if not isinstance(filename, MutableMapping):
        ensure_directory(filename.parent)
        is_file = filename.is_file() or (
            filename.is_dir() and os.path.splitext(filename)[1] == ".zspy"
        )

        if overwrite is None:
            write = overwrite_method(filename)  # Ask what to do
        elif overwrite is True or (overwrite is False and not is_file):
            write = True  # Write the file
        elif overwrite is False and is_file:
            write = False  # Don't write the file
        else:
            raise ValueError(
                "`overwrite` parameter can only be None, True or " "False."
            )
    else:
        write = True  # file does not exist (creating it)
    if write:
        # Pass as a string for now, pathlib.Path not
        # properly supported in io_plugins
        signal = _add_file_load_save_metadata("save", signal, writer)
        signal_dic = signal._to_dictionary(add_models=True)
        signal_dic["package_info"] = get_object_package_info(signal)
        if not isinstance(filename, MutableMapping):
            importlib.import_module(writer["api"]).file_writer(
                str(filename), signal_dic, **kwds
            )
            _logger.info(f"{filename} was created")
            signal.tmp_parameters.set_item("folder", filename.parent)
            signal.tmp_parameters.set_item("filename", filename.stem)
            signal.tmp_parameters.set_item("extension", extension)
        else:
            importlib.import_module(writer["api"]).file_writer(
                filename, signal_dic, **kwds
            )
            if hasattr(filename, "path"):
                file = Path(filename.path).resolve()
                signal.tmp_parameters.set_item("folder", file.parent)
                signal.tmp_parameters.set_item("filename", file.stem)
                signal.tmp_parameters.set_item("extension", extension)


def _add_file_load_save_metadata(operation, signal, io_plugin):
    mdata_dict = {
        "operation": operation,
        "io_plugin": io_plugin["api"]
        if isinstance(io_plugin, dict)
        else io_plugin.__loader__.name,
        "hyperspy_version": hs_version,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    # get the largest integer key present under General.FileIO, returning 0
    # as default if none are present
    largest_index = max(
        [
            int(i.replace("Number_", "")) + 1
            for i in signal.metadata.get_item("General.FileIO", {}).keys()
        ]
        + [0]
    )

    signal.metadata.set_item(f"General.FileIO.{largest_index}", mdata_dict)

    return signal
