.. _metadata_structure:


Metadata structure
******************

The :class:`~.api.signals.BaseSignal` class stores metadata in the
:attr:`~.api.signals.BaseSignal.metadata` attribute, which has a tree structure. By
convention, the node labels are capitalized and the leaves are not
capitalized.

When a leaf contains a quantity that is not dimensionless, the units can be
given in an extra leaf with the same label followed by the "_units" suffix.
For example, an "energy" leaf should be accompanied by an "energy_units" leaf.

The metadata structure is represented in the following tree diagram. The
default units are given in parentheses. Details about the leaves can be found
in the following sections of this chapter.

::

    metadata
    ├── General
    |   |── FileIO
    |   |   ├── 0
    |   |   |   ├── operation
    |   |   |   ├── hyperspy_version
    |   |   |   ├── io_plugin
    |   │   |   └── timestamp
    |   |   ├── 1
    |   |   |   ├── operation
    |   |   |   ├── hyperspy_version
    |   |   |   ├── io_plugin
    |   │   |   └── timestamp
    |   |   └── ...
    │   ├── authors
    │   ├── date
    │   ├── doi
    │   ├── original_filename
    │   ├── notes
    │   ├── time
    │   ├── time_zone
    │   └── title
    ├── Sample
    │   ├── credits
    │   ├── description
    │   └── thickness
    └── Signal
        ├── FFT
        │   └── shifted
        ├── Noise_properties
        │   ├── Variance_linear_model
        │   │   ├── correlation_factor
        │   │   ├── gain_factor
        │   │   ├── gain_offset
        │   │   └── parameters_estimation_method
        │   └── variance
        ├── quantity
        ├── signal_type
        └── signal_origin


.. _general-metadata:

General
=======

title
    type: Str

    A title for the signal, e.g. "Sample overview"

original_filename
    type: Str

    If the signal was loaded from a file this key stores the name of the
    original file.

time_zone
    type: Str

    The time zone as supported by the python-dateutil library, e.g. "UTC",
    "Europe/London", etc. It can also be a time offset, e.g. "+03:00" or
    "-05:00".

time
    type: Str

    The acquisition or creation time in ISO 8601 time format, e.g. '13:29:10'.

date
    type: Str

    The acquisition or creation date in ISO 8601 date format, e.g.
    '2018-01-28'.


authors
    type: Str

    The authors of the data, in Latex format: Surname1, Name1 and Surname2,
    Name2, etc.

doi
    type: Str

    Digital object identifier of the data, e. g. doi:10.5281/zenodo.58841.

notes
    type: Str

    Notes about the data.

.. _general-file-metadata:

FileIO
------

Contains information about the software packages and versions used any time the
Signal was created by reading the original data format (added in HyperSpy
v1.7) or saved by one of HyperSpy's IO tools. If the signal is saved to one
of the ``hspy``, ``zspy`` or ``nxs`` formats, the metadata within the ``FileIO``
node will represent a history of the software configurations used when the
conversion was made from the proprietary/original format to HyperSpy's
format, as well as any time the signal was subsequently loaded from and saved
to disk. Under the ``FileIO`` node will be one or more nodes named ``0``,
``1``, ``2``, etc., each with the following structure:

operation
   type: Str

   This value will be either ``"load"`` or ``"save"`` to indicate whether
   this node represents a load from, or save to disk operation, respectively.

hyperspy_version
    type: Str

    The version number of the HyperSpy software used to extract a Signal from
    this data file or save this Signal to disk

io_plugin
    type: Str

    The specific input/output plugin used to originally extract this data file
    into a HyperSpy Signal or save it to disk -- will be of the form
    ``rsciio.<plugin_name>``.

timestamp
    type: Str

    The timestamp of the computer running the data loading/saving process (in a
    timezone-aware format). The timestamp will be in ISO 8601 format, as
    produced by the :meth:`datetime.date.isoformat`.


.. _sample-metadata:

Sample
======

credits
    type: Str

    Acknowledgment of sample supplier, e.g. Prepared by Putin, Vladimir V.

description
    type: Str

    A brief description of the sample

thickness
    type: Float

    The thickness of the sample in m.


.. _signal-metadata:

Signal
======

signal_type
    type: Str

    A term that describes the signal type, e.g. EDS, PES... This information
    can be used by HyperSpy to load the file as a specific signal class and
    therefore the naming should be standardised. Currently, HyperSpy provides
    special signal class for photoemission spectroscopy, electron energy
    loss spectroscopy and energy dispersive spectroscopy. The signal_type in
    these cases should be respectively PES, EELS and EDS_TEM (EDS_SEM).

signal_origin
    type: Str

    Describes the origin of the signal e.g. 'simulation' or 'experiment'.


record_by
    .. deprecated:: 1.2

    type: Str

    One of 'spectrum' or 'image'. It describes how the data is stored in memory.
    If 'spectrum', the spectral data is stored in the faster index.

quantity
    type: Str

    The name of the quantity of the "intensity axis" with the units in round
    brackets if required, for example Temperature (K).


FFT
---

shifted
    type: bool.

    Specify if the FFT has the zero-frequency component shifted to the center of
    the signal.


Noise_properties
----------------

variance
    type: float or BaseSignal instance.

    The variance of the data. It can be a float when the noise is Gaussian or a
    :class:`~.api.signals.BaseSignal` instance if the noise is heteroscedastic,
    in which case it must have the same dimensions as
    :attr:`~.api.signals.BaseSignal.data`.

Variance_linear_model
^^^^^^^^^^^^^^^^^^^^^

In some cases the variance can be calculated from the data using a simple
linear model: ``variance = (gain_factor * data + gain_offset) *
correlation_factor``.

gain_factor
    type: Float

gain_offset
    type: Float

correlation_factor
    type: Float

parameters_estimation_method
    type: Str


_Internal_parameters
====================

This node is "private" and therefore is not displayed when printing the
:attr:`~.api.signals.BaseSignal.metadata` attribute.

Stacking_history
----------------

Generated when using :func:`~.api.stack`. Used by
:meth:`~.api.signals.BaseSignal.split`, to retrieve the former list of signal.

step_sizes
    type: list of int

    Step sizes used that can be used in split.

axis
    type: int

   The axis index in axes manager on which the dataset were stacked.

Folding
-------

Constains parameters that related to the folding/unfolding of signals.


.. _metadata_handling:

Functions to handle the metadata
================================

Existing nodes can be directly read out or set by adding the path in the
metadata tree:

::

    s.metadata.General.title = 'FlyingCircus'
    s.metadata.General.title


The following functions can operate on the metadata tree. An example with the
same functionality as the above would be:

::

    s.metadata.set_item('General.title', 'FlyingCircus')
    s.metadata.get_item('General.title')


Adding items
------------

:meth:`~.misc.utils.DictionaryTreeBrowser.set_item`
    Given a ``path`` and ``value``, easily set metadata items, creating any
    necessary nodes on the way.

:meth:`~.misc.utils.DictionaryTreeBrowser.add_dictionary`
    Add new items from a given ``dictionary``.


Output metadata
---------------

:meth:`~.misc.utils.DictionaryTreeBrowser.get_item`
    Given an ``item_path``, return the ``value`` of the metadata item.

:meth:`~.misc.utils.DictionaryTreeBrowser.as_dictionary`
    Returns a dictionary representation of the metadata tree.

:meth:`~.misc.utils.DictionaryTreeBrowser.export`
    Saves the metadata tree in pretty tree printing format in a text file.
    Takes ``filename`` as parameter.


Searching for keys
------------------

:meth:`~.misc.utils.DictionaryTreeBrowser.has_item`
    Given an ``item_path``, returns ``True`` if the item exists anywhere
    in the metadata tree.

Using the option ``full_path=False``, the functions
:meth:`~.misc.utils.DictionaryTreeBrowser.has_item` and
:meth:`~.misc.utils.DictionaryTreeBrowser.get_item` can also find items by
their key in the metadata when the exact path is not known. By default, only
an exact match of the search string with the item key counts. The additional
setting ``wild=True`` allows to search for a case-insensitive substring of the
item key. The search functionality also accepts item keys preceded by one or
several nodes of the path (separated by the usual full stop).

:meth:`~.misc.utils.DictionaryTreeBrowser.has_item`
    For ``full_path=False``, given a ``item_key``, returns ``True`` if the item
    exists anywhere in the metadata tree.

:meth:`~.misc.utils.DictionaryTreeBrowser.has_item`
    For ``full_path=False, return_path=True``, returns the path or list of
    paths to any matching item(s).

:meth:`~.misc.utils.DictionaryTreeBrowser.get_item`
    For ``full_path=False``, returns the value or list of values for any
    matching item(s). Setting ``return_path=True``, a tuple (value, path) is
    returned -- or lists of tuples for multiple occurences.
