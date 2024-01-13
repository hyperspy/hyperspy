.. _basic-usage:

Basic Usage
===========


.. _importing_hyperspy-label:

Starting Python in Windows
----------------------------
If you used the bundle installation you should be able to use the context menus
to get started. Right-click on the folder containing the data you wish to
analyse and select "Jupyter notebook here" or "Jupyter qtconsole here". We
recommend the former, since notebooks have many advantages over conventional
consoles, as will be illustrated in later sections. The examples in some later
sections assume Notebook operation. A new tab should appear in your default
browser listing the files in the selected folder. To start a python notebook
choose "Python 3" in the "New" drop-down menu at the top right of the page.
Another new tab will open which is your Notebook.

Starting Python in Linux and MacOS
------------------------------------

You can start IPython by opening a system terminal and executing ``ipython``,
(optionally followed by the "frontend": "qtconsole" for example). However, in
most cases, **the most agreeable way** to work with HyperSpy interactively
is using the `Jupyter Notebook <https://jupyter.org>`_ (previously known as
the IPython Notebook), which can be started as follows:

.. code-block:: bash

    $ jupyter notebook

Linux users may find it more convenient to start Jupyter/IPython from the
`file manager context menu <https://github.com/hyperspy/start_jupyter_cm>`_.
In either OS you can also start by `double-clicking a notebook file
<https://github.com/takluyver/nbopen>`_ if one already exists.

Starting HyperSpy in the notebook (or terminal)
-----------------------------------------------
Typically you will need to `set up IPython for interactive plotting with
matplotlib
<https://ipython.readthedocs.org/en/stable/interactive/plotting.html>`_ using
``%matplotlib`` (which is known as a 'Jupyter magic')
*before executing any plotting command*. So, typically, after starting
IPython, you can import HyperSpy and set up interactive matplotlib plotting by
executing the following two lines in the IPython terminal (In these docs we
normally use the general Python prompt symbol ``>>>`` but you will probably
see ``In [1]:`` etc.):

.. code-block:: python

   >>> %matplotlib qt # doctest: +SKIP
   >>> import hyperspy.api as hs

Note that to execute lines of code in the notebook you must press
``Shift+Return``. (For details about notebooks and their functionality try
the help menu in the notebook). Next, import two useful modules: numpy and
matplotlib.pyplot, as follows:

.. code-block:: python

   >>> import numpy as np
   >>> import matplotlib.pyplot as plt

The rest of the documentation will assume you have done this. It also assumes
that you have installed at least one of HyperSpy's GUI packages:
`jupyter widgets GUI <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`_
and the
`traitsui GUI <https://github.com/hyperspy/hyperspy_gui_traitsui>`_.

Possible warnings when importing HyperSpy?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy supports different GUIs and
`matplotlib backends <https://matplotlib.org/stable/users/explain/backends.html>`_
which in specific cases can lead to warnings when importing HyperSpy. Most of the time
there is nothing to worry about — the warnings simply inform you of several choices you have.
There may be several causes for a warning, for example:

- not all the GUIs packages are installed. If none is installed, we reccomend you to install
  at least the ``hyperspy-gui-ipywidgets`` package is your are planning to perform interactive
  data analysis in the Jupyter Notebook. Otherwise, you can simply disable the warning in
  :ref:`preferences <configuring-hyperspy-label>` as explained below.
- the ``hyperspy-gui-traitsui`` package is installed and you are using an incompatible matplotlib
  backend (e.g. ``notebook``, ``nbagg`` or ``widget``).

   - If you want to use the traitsui GUI, use the ``qt`` matplotlib backend instead.
   - Alternatively, if you prefer to use the ``notebook`` or ``widget`` matplotlib backend,
     and if you don't want to see the (harmless) warning, make sure that you have the
     ``hyperspy-gui-ipywidgets`` installed and disable the traitsui
     GUI in the :ref:`preferences <configuring-hyperspy-label>`.

.. versionchanged:: v1.3
    HyperSpy works with all matplotlib backends, including the ``notebook``
    (also called ``nbAgg``) backend that enables interactive plotting embedded
    in the jupyter notebook.


.. NOTE::

    When running in a  headless system it is necessary to set the matplotlib
    backend appropiately to avoid a `cannot connect to X server` error, for
    example as follows:

    .. code-block:: python

       >>> import matplotlib
       >>> matplotlib.rcParams["backend"] = "Agg"
       >>> import hyperspy.api as hs


Getting help
------------

When using IPython, the documentation (docstring in Python jargon) can be
accessed by adding a question mark to the name of a function. e.g.:

.. ipython::
    :okexcept:

    In [1]: import hyperspy.api as hs
    In [2]: hs?
    In [3]: hs.load?
    In [4]: hs.signals?

This syntax is a shortcut to the standard way one of displaying the help
associated to a given functions (docstring in Python jargon) and it is one of
the many features of `IPython <https://ipython.readthedocs.io/>`_, which is the
interactive python shell that HyperSpy uses under the hood.


Autocompletion
--------------

Another useful IPython feature is the
`autocompletion <https://ipython.readthedocs.io/en/stable/interactive/tutorial.html#tab-completion>`_
of commands and filenames using the tab and arrow keys. It is highly recommended
to read the `Ipython introduction <https://ipython.readthedocs.io/en/stable/interactive/tutorial.html>`_ for many more useful features that will
boost your efficiency when working with HyperSpy/Python interactively.

Creating signal from a numpy array
----------------------------------

HyperSpy can operate on any numpy array by assigning it to a BaseSignal class.
This is useful e.g. for loading data stored in a format that is not yet
supported by HyperSpy—supposing that they can be read with another Python
library—or to explore numpy arrays generated by other Python
libraries. Simply select the most appropriate signal from the
:mod:`~.api.signals` module and create a new instance by passing a numpy array
to the constructor e.g.

.. code-block:: python

    >>> my_np_array = np.random.random((10, 20, 100))
    >>> s = hs.signals.Signal1D(my_np_array)
    >>> s
    <Signal1D, title: , dimensions: (20, 10|100)>

The numpy array is stored in the :attr:`~.api.signals.BaseSignal.data` attribute
of the signal class:

.. code-block:: python

    >>> s.data # doctest: +SKIP


.. _navigation-signal-dimensions:

The navigation and signal dimensions
------------------------------------

In HyperSpy the data is interpreted as a signal array and, therefore, the data
axes are not equivalent. HyperSpy distinguishes between *signal* and
*navigation* axes and most functions operate on the *signal* axes and
iterate on the *navigation* axes. For example, an EELS spectrum image (i.e.
a 2D array of spectra) has three dimensions X, Y and energy-loss. In
HyperSpy, X and Y are the *navigation* dimensions and the energy-loss is the
*signal* dimension. To make this distinction more explicit the
representation of the object includes a separator ``|`` between the
navigation and signal dimensions e.g.

In HyperSpy a spectrum image has signal dimension 1 and navigation dimension 2
and is stored in the Signal1D subclass.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.zeros((10, 20, 30)))
    >>> s
    <Signal1D, title: , dimensions: (20, 10|30)>


An image stack has signal dimension 2 and navigation dimension 1 and is stored
in the Signal2D subclass.

.. code-block:: python

    >>> im = hs.signals.Signal2D(np.zeros((30, 10, 20)))
    >>> im
    <Signal2D, title: , dimensions: (30|20, 10)>

Note that HyperSpy rearranges the axes when compared to the array order. The
following few paragraphs explain how and why it does it.

Depending how the array is arranged, some axes are faster to iterate than
others. Consider an example of a book as the dataset in question. It is
trivially simple to look at letters in a line, and then lines down the page,
and finally pages in the whole book.  However if your words are written
vertically, it can be inconvenient to read top-down (the lines are still
horizontal, it's just the meaning that's vertical!). It's very time-consuming
if every letter is on a different page, and for every word you have to turn 5-6
pages. Exactly the same idea applies here - in order to iterate through the
data (most often for plotting, but applies for any other operation too), you
want to keep it ordered for "fast access".

In Python (more explicitly `numpy`) the "fast axes order" is C order (also
called row-major order). This means that the **last** axis of a numpy array is
fastest to iterate over (i.e. the lines in the book). An alternative ordering
convention is F order (column-major), where it is the reverse - the first axis
of an array is the fastest to iterate over. In both cases, the further an axis
is from the `fast axis` the slower it  is to iterate over it. In the book
analogy you could think, for example, think about reading the first lines of
all pages, then the second and so on.

When data is acquired sequentially it is usually stored in acquisition order.
When a dataset is loaded, HyperSpy generally stores it in memory in the same
order, which is good for the computer. However, HyperSpy will reorder and
classify the axes to make it easier for humans. Let's imagine a single numpy
array that contains pictures of a scene acquired with different exposure times
on different days. In numpy the array dimensions are  ``(D, E, Y, X)``. This
order makes it fast to iterate over the images in the order in which they were
acquired. From a human point of view, this dataset is just a collection of
images, so HyperSpy first classifies the image axes (``X`` and ``Y``) as
`signal axes` and the remaining axes the `navigation axes`. Then it reverses
the order of each sets of axes because many humans are used to get the ``X``
axis first and, more generally the axes in acquisition order from left to
right. So, the same axes in HyperSpy are displayed like this: ``(E, D | X,
Y)``.

Extending this to arbitrary dimensions, by default, we reverse the numpy axes,
chop it into two chunks (signal and navigation), and then swap those chunks, at
least when printing. As an example:

.. code-block:: bash

    (a1, a2, a3, a4, a5, a6) # original (numpy)
    (a6, a5, a4, a3, a2, a1) # reverse
    (a6, a5) (a4, a3, a2, a1) # chop
    (a4, a3, a2, a1) (a6, a5) # swap (HyperSpy)

In the background, HyperSpy also takes care of storing the data in memory in
a "machine-friendly" way, so that iterating over the navigation axes is always
fast.


.. _saving:

Saving Files
------------

The data can be saved to several file formats.  The format is specified by
the extension of the filename.

.. code-block:: python

    >>> # load the data
    >>> d = hs.load("example.tif") # doctest: +SKIP
    >>> # save the data as a tiff
    >>> d.save("example_processed.tif") # doctest: +SKIP
    >>> # save the data as a png
    >>> d.save("example_processed.png") # doctest: +SKIP
    >>> # save the data as an hspy file
    >>> d.save("example_processed.hspy") # doctest: +SKIP

Some file formats are much better at maintaining the information about
how you processed your data. The preferred formats are
:external+rsciio:ref:`hspy <hspy-format>` and :external+rsciio:ref:`zspy <zspy-format>`,
because they are open formats and keep most information possible.

There are optional flags that may be passed to the save function. See
:ref:`saving_files` for more details.

Accessing and setting the metadata
----------------------------------

When loading a file HyperSpy stores all metadata in the BaseSignal
:attr:`~.api.signals.BaseSignal.original_metadata` attribute. In addition,
some of those metadata and any new metadata generated by HyperSpy are stored in
:attr:`~.api.signals.BaseSignal.metadata` attribute.


.. code-block:: python

    >>> import exspy  # doctest: +SKIP
    >>> s = exspy.data.eelsdb(formula="NbO2", edge="M2,3")[0] # doctest: +SKIP
    >>> s.metadata  # doctest: +SKIP
    ├── Acquisition_instrument
    │   └── TEM
    │       ├── Detector
    │       │   └── EELS
    │       │       └── collection_angle = 6.5
    │       ├── beam_energy = 100.0
    │       ├── convergence_angle = 10.0
    │       └── microscope = VG HB501UX
    ├── General
    │   ├── author = Wilfried Sigle
    │   └── title = Niobium oxide NbO2
    ├── Sample
    │   ├── chemical_formula = NbO2
    │   ├── description =  Analyst: David Bach, Wilfried Sigle. Temperature: Room. 
    │   └── elements = ['Nb', 'O']
    └── Signal
        ├── quantity = Electrons ()
        └── signal_type = EELS

    >>> s.original_metadata  # doctest: +SKIP
    ├── emsa
    │   ├── DATATYPE = XY
    │   ├── DATE = 
    │   ├── FORMAT = EMSA/MAS Spectral Data File
    │   ├── NCOLUMNS = 1.0
    │   ├── NPOINTS = 1340.0
    │   ├── OFFSET = 120.0003
    │   ├── OWNER = eelsdatabase.net
    │   ├── SIGNALTYPE = ELS
    │   ├── TIME = 
    │   ├── TITLE = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217
    │   ├── VERSION = 1.0
    │   ├── XPERCHAN = 0.5
    │   ├── XUNITS = eV
    │   └── YUNITS = 
    └── json
        ├── api_permalink = https://api.eelsdb.eu/spectra/niobium-oxide-nbo2-2/
        ├── associated_spectra = [{'name': 'Niobium oxide NbO2', 'link': 'https://eelsdb.eu/spectra/niobium-oxide-nbo2/', 'type': 'Low Loss'}]
        ├── author
        │   ├── name = Wilfried Sigle
        │   ├── profile_api_url = https://api.eelsdb.eu/author/wsigle/
        │   └── profile_url = https://eelsdb.eu/author/wsigle/
        ├── beamenergy = 100 kV
        ├── collection = 6.5 mrad
        ├── comment_count = 0
        ├── convergence = 10 mrad
        ├── darkcurrent = Yes
        ├── description =  Analyst: David Bach, Wilfried Sigle. Temperature: Room. 
        ├── detector = Parallel: Gatan ENFINA
        ├── download_link = https://eelsdb.eu/wp-content/uploads/2015/09/DspecYB7EbW.msa
        ├── edges = ['Nb_M2,3', 'Nb_M4,5', 'O_K']
        ├── elements = ['Nb', 'O']
        ├── formula = NbO2
        ├── gainvariation = Yes
        ├── guntype = cold field emission
        ├── id = 21727
        ├── integratetime = 5 secs
        ├── keywords = ['imported from old site']
        ├── max_energy = 789.5 eV
        ├── microscope = VG HB501UX
        ├── min_energy = 120 eV
        ├── monochromated = No
        ├── other_links = [{'url': 'http://pc-web.cemes.fr/eelsdb/index.php?page=displayspec.php&id=217', 'title': 'Old EELS DB'}]
        ├── permalink = https://eelsdb.eu/spectra/niobium-oxide-nbo2-2/
        ├── published = 2008-02-15 00:00:00
        ├── readouts = 10
        ├── resolution = 1.3 eV
        ├── stepSize = 0.5 eV/pixel
        ├── thickness = 0.58 t/&lambda;
        ├── title = Niobium oxide NbO2
        └── type = Core Loss

    >>> s.metadata.General.title = "NbO2 Nb_M edge"  # doctest: +SKIP
    >>> s.metadata  # doctest: +SKIP
    ├── Acquisition_instrument
    │   └── TEM
    │       ├── Detector
    │       │   └── EELS
    │       │       └── collection_angle = 6.5
    │       ├── beam_energy = 100.0
    │       ├── convergence_angle = 10.0
    │       └── microscope = VG HB501UX
    ├── General
    │   ├── author = Wilfried Sigle
    │   └── title = NbO2 Nb_M edge
    ├── Sample
    │   ├── chemical_formula = NbO2
    │   ├── description =  Analyst: David Bach, Wilfried Sigle. Temperature: Room. 
    │   └── elements = ['Nb', 'O']
    └── Signal
        ├── quantity = Electrons ()
        └── signal_type = EELS


.. _configuring-hyperspy-label:

Configuring HyperSpy
--------------------

The behaviour of HyperSpy can be customised using the
:attr:`~.api.preferences`. The easiest way to do it is by calling
the :meth:`~.api.preferences.gui` method:

.. code-block:: python

    >>> hs.preferences.gui() # doctest: +SKIP

This command should raise the Preferences user interface if one of the
hyperspy gui packages are installed and enabled:

.. _preferences_image:

.. figure::  images/preferences.png
   :align:   center

   Preferences user interface.

.. versionadded:: 1.3
    Possibility to enable/disable GUIs in the preferences.

It is also possible to set the preferences programmatically. For example,
to disable the traitsui GUI elements and save the changes to disk:

.. code-block:: python

    >>> hs.preferences.GUIs.enable_traitsui_gui = False
    >>> hs.preferences.save()
    >>> # if not saved, this setting will be used until the next jupyter kernel shutdown

.. versionchanged:: 1.3

   The following items were removed from preferences:
   ``General.default_export_format``, ``General.lazy``,
   ``Model.default_fitter``, ``Machine_learning.multiple_files``,
   ``Machine_learning.same_window``, ``Plot.default_style_to_compare_spectra``,
   ``Plot.plot_on_load``, ``Plot.pylab_inline``, ``EELS.fine_structure_width``,
   ``EELS.fine_structure_active``, ``EELS.fine_structure_smoothing``,
   ``EELS.synchronize_cl_with_ll``, ``EELS.preedge_safe_window_width``,
   ``EELS.min_distance_between_edges_for_fine_structure``.



.. _logger-label:

Messages log
------------

HyperSpy writes messages to the `Python logger
<https://docs.python.org/3/howto/logging.html#logging-basic-tutorial>`_. The
default log level is "WARNING", meaning that only warnings and more severe
event messages will be displayed. The default can be set in the
:ref:`preferences <configuring-hyperspy-label>`. Alternatively, it can be set
using :func:`~.api.set_log_level` e.g.:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> hs.set_log_level('INFO')
    >>> hs.load('my_file.dm3') # doctest: +SKIP
    INFO:hyperspy.io_plugins.digital_micrograph:DM version: 3
    INFO:hyperspy.io_plugins.digital_micrograph:size 4796607 B
    INFO:hyperspy.io_plugins.digital_micrograph:Is file Little endian? True
    INFO:hyperspy.io_plugins.digital_micrograph:Total tags in root group: 15
    <Signal2D, title: My file, dimensions: (|1024, 1024)
