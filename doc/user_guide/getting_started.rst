Getting started
***************

Starting hyperspy
-----------------

Hyperspy is a Python library to analyze multidimensional. Hyperspy does not
have a GUI. Instead, the most common way of running Hyperspy is interactively
using the wonderful interactive computing package `IPython
<http://ipython.org>`_. In this section we describe the different ways to start
hyperspy in the different operating systems.

Starting hyperspy from the terminal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In all operating systems (OS) you can start Hyperspy
by opening a system terminal and typing hyperspy:

.. code-block:: bash

    $ hyperspy


If Hyperspy is correctly installed it should welcome you with a message similar
to:

.. code-block:: ipython
    
    H y p e r s p y
    Version 0.7
    
    http://www.hyperspy.org	
	

If IPython 0.11 or newer and the Qt libraries are installed in your system it
is also possible to run Hyperspy in `IPython's QtConsole
<http://ipython.org/ipython-doc/stable/interactive/qtconsole.html>`_ by
executing `hyperspy qtconsole` in a terminal:

.. code-block:: bash

    $ hyperspy qtconsole

If IPython 0.12 or newer is installed in your system it is also possible to run
Hyperspy in `IPython's HTML notebook
<http://ipython.org/ipython-doc/stable/interactive/htmlnotebook.html>`_ that
runs inside your browser. The Notebook is probably **the most agreeable way**
to work with Hyperspy interactively. You can start it from a terminal as
follows

.. code-block:: bash

    $ hyperspy notebook 

There are multiple options available when starting from the terminal. To print
these options add the `-h` flag:

.. code-block:: bash

    $ hyperspy -h
    usage: hyperspy [-h] [-v] [--toolkit {qt4,gtk,wx,tk,None}]
                    [--pylab_inline] [--overwrite_profile]
                    [--ipython_args [IPYTHON_ARGS [IPYTHON_ARGS ...]]]
                    [{terminal,console,qtconsole,notebook}]

    Hyperspectral data analysis toolbox
 
    positional arguments:
      {terminal,console,qtconsole,notebook}
                            Selects the IPython environment in which to start
                            Hyperspy. The default is terminal
 
    optional arguments:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit
      --toolkit {qt4,gtk,wx,tk,None}
                            Pre-load matplotlib and traitsui for interactive
                            use,selecting a particular matplotlib backend and loop
                            integration.When using gtk and tk toolkits the user
                            interface elements are not available. None is suitable
                            to run headless.
      --pylab_inline        If True the figure are displayed inline. This option
                            only has effect when using the qtconsole or notebook
      --overwrite_profile   Overwrite the Ipython profile with the default one.
      --ipython_args [IPYTHON_ARGS [IPYTHON_ARGS ...]]
                            Arguments to be passed to IPython. This option must be
                            the last one.Look at the IPython documentation for
                            available options.
 
Starting hyperspy from the context menu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This option is only available for Windows and in Linux when using Gnome.

Windows
"""""""

In Windows it is possible to start Hyperspy from :menuselection:`Start Menu -->
Programs --> Hyperspy`.

Alternatively, one can start Hyperspy in any folder by pressing the :kbd:`right
mouse button` or on a yellow folder icon or (in some cases) on the empty area
of a folder, and choosing :menuselection:`Hyperspy qtconsole here` or
:menuselection:`Hyperspy notebook here` from the context menu.


.. figure::  images/windows_hyperspy_here.png
   :align:   center
   :width:   500    

   Starting hyperspy using the Windows context menu.
   

Linux
"""""

If you are using GNOME in Linux, you can open a terminal in a folder by
choosing :menuselection:`open terminal` in the file menu if
:program:`nautilus-open-terminal` is installed in your system.

Altenatively (and more conviently), if you are using Gnome place `this
<https://github.com/downloads/hyperspy/hyperspy/Hyperspy%20QtConsole%20here.sh>`_
and `this
<https://github.com/downloads/hyperspy/hyperspy/Hyperspy%20Notebook%20here.sh>`_
in the :file:`/.gnome2/nautilus-scripts` folder in your home directory (create
it if it does not exists) and make them executable to get the
:menuselection:`Scripts --> Hyperspy QtConsole Here` and
:menuselection:`Scripts --> Hyperspy Notebook Here` entries in the context
menu. 


.. figure::  images/hyperspy_here_gnome.png
   :align:   center
   :width:   500    

   Starting hyperspy using the Gnome nautilus context menu.

Using Hyperspy as a library
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When starting hyperspy by using the file browser context menu or by running the
`hyperspy` script in the terminal, the `hyperspy` script simply starts and
configures IPython appropiately and imports the contents of the
:py:mod:`~.hspy` module in the user namespace. Of course, It is possible to use
Hyperspy as a library by simply importing its modules. The recommended way to
do it is importing the hspy module as follows:

.. code-block:: python

    import hyperspy.hspy as hp

Choosing a toolkit
------------------
.. versionadded:: 0.7

Hyperspy fully supports the Qt toolkit in addition to the WX toolkit. GTK and
TK are also supported but the user interface elements are not available. The
default toolkit is Qt4/PySide because currently is the one that works best in
most platforms.

The toolkit can be specified on start using the `--toolkit` flag. Also, the
default value can be configured in :ref:`preferences
<configuring-hyperspy-label>`.

Inline plotting
---------------
.. versionadded:: 0.7

Hyperspy can be started using the IPython inline plotting mode by passing the
`--pylab_inline` flag when starting hyperspy. In inline-mode, calling any
plotting function embeds the resulting plots in the Notebook or QtConsole
instead of raising figure windows. The main drawback is that these plots are
not (yet) interactive.

This option only has effect when
running in the *IPython QtConsole* or the *IPython Notebook*, e.g.

.. code-block:: bash
   $ hyperspy qtconsole --pylab_inline

The default value can be configured in :ref:`preferences
<configuring-hyperspy-label>`.

.. _headless-label:

Using Hyperpsy in a headless system
-----------------------------------
.. versionadded:: 0.7

To run Hyperpsy in a headless system select "None" as the toolkit either in 
:ref:`preferences <configuring-hyperspy-label>` or when starting from a
terminal, e.g.:


.. code-block:: bash

    $ hyperspy --toolkit None 


Getting help
------------

The documentation (docstring in Python jargon) can be accessed by adding a
question mark to the name of a function. e.g.:

.. code-block:: python
    
    >>> load?

This syntax is a shortcut to the standard way one of displaying the help
associated to a given functions (docstring in Python jargon) and it is one of
the many features of `IPython <http://ipython.scipy.org/moin/>`_, which is the
interactive python shell that Hyperspy uses under the hood.

Please note that the documentation of the code is a work in progress, so not
all the objects are documented yet.

Up-to-date documentation is always available in `the Hyperspy website.
<http://hyperspy.org/documentation.html>`_


Autocompletion
--------------

Another useful `IPython <http://ipython.scipy.org/moin/>`_ feature is the
autocompletion of commands and filenames using the tab and arrow keys. It is
highly recommended to read the `Ipython documentation
<http://ipython.scipy.org/moin/Documentation>`_ (specially their `Getting
started <http://ipython.org/ipython-doc/stable/interactive/tutorial.html>`_
section) for many more useful features that will boost your efficiency when
working with Hyperspy/Python interactively.


Loading data
------------

Once hyperspy is running, to load from a supported file format (see
:ref:`supported-formats`) simply type:

.. code-block:: python

    >>> s = load("filename")

.. HINT::

   The load function returns an object that contains data read from the file.
   We assign this object to the variable ``s`` but you can choose any (valid)
   variable name you like. for the filename, don't forget to include the
   quotation marks and the file extension.
   
If no argument is passed to the load function, a window will be raised that
allows to select a single file through your OS file manager, e.g.:

.. code-block:: python

    >>> # This raises the load user interface
    >>> s = load()

It is also possible to load multiple files at once or even stack multiple
files. For more details read :ref:`loading_files`

"Loading" zadata from a numpy array
---------------------------------

Hyperspy can operate on any numpy array by assigning it to a Signal class.
This is useful e.g. for loading data stored in a format that is not yet
supported by Hyperspy—supposing that they can be read with another Python
library—or to explore numpy arrays generated by other Python
libraries. Simply select the most appropiate signal from the
:py:mod:`~.signals` module and create a new instance by passing a numpy array
to the constructor e.g.

.. code-block:: python

    >>> my_np_array = np.random.random((10,20,100)) 
    >>> s = signals.Spectrum(my_np_array)
    >>> s
    <Spectrum, title: , dimensions: (20, 10|100)>
   
The numpy array is stored in the :py:attr:`~.signal.Signal.data` attribute 
of the signal class.

The navigation and signal dimensions
------------------------------------

In Hyperspy the data is interpreted as a signal array and, therefore, the data
axes are not equivalent. Hyperspy distiguises between *signal* and *navigation*
axes and most functions operate on the *signal* axes and iterate on the
*navigation* axes. For example, an EELS spectrum image (i.e. a 2D array of
spectra) has three dimensions X, Y and energy-loss. In Hyperspy, X and Y are
the *navigation* dimensions an the energy-loss is the *signal* dimension. To
make this distinction more explicit the representation of the object includes
a separator ``|`` between the navigaton and signal dimensions e.g.

In Hyperpsy a spectrum image has signal dimension 1 and navigation dimension 2.

.. code-block:: python
   
    >>> s = signals.Spectrum(np.zeros((10, 20, 30)))
    >>> s
    <Spectrum, title: , dimensions: (20, 10|30)>


An image stack has signal dimension 2 and navigation dimension 1. 

.. code-block:: python

    >>> im = signals.Image(np.zeros((30, 10, 20)))
    >>> im
    <Image, title: , dimensions: (30|20, 10)>

Note the Hyperspy rearranges the axes position to match the following pattern:
(navigatons axis 0,..., navigation axis n|signal axis 0,..., signal axis n).
This is the order used for :ref:`indexing the Signal class <signal.indexing>`.


Setting axis properties
-----------------------

The axes are managed and stored by the :py:class:`~.axes.AxesManager` class
that is stored in the :py:attr:`~.signal.Signal.axes_manager` attribute of
the signal class. The indidual axes can be accessed by indexing the AxesManager
e.g. 

.. code-block:: python

    >>> s = signals.Spectrum(np.random.random((10, 20 , 100)))
    >>> s
    <Spectrum, title: , dimensions: (20, 10|100)>
    >>> s.axes_manager
    <Axes manager, axes: (<Unnamed 0th axis, size: 20, index: 0>, <Unnamed 1st
    axis, size: 10, index: 0>|<Unnamed 2nd axis, size: 100>)>
    >>> s.axes_manager[0]
    <Unnamed 0th axis, size: 20, index: 0>


The axis properties can be set by setting the :py:class:`~.axes.DataAxis`
attributes e.g. 

.. code-block:: python

    >>> s.axes_manager[0].name = "X"
    >>> s.axes_manager[0]
    <X axis, size: 20, index: 0>
   

Once the name of an axis has been defined it is possible to request it by its
name e.g.:

.. code-block:: python

    >>> s.axes_manager["X"]
    <X axis, size: 20, index: 0>
    >>> s.axes_manager["X"].scale = 0.2
    >>> s.axes_manager["X"].units = nm
    >>> s.axes_manager["X"].offset = 100
    

It is also possible to set the axes properties using a GUI by calling the
:py:meth:`~.axes.AxesManager.gui` method of the :py:class:`~.axes.AxesManager`. 

.. _saving:

Saving Files
------------

The data can be saved to several file formats.  The format is specified by
the extension of the filename.

.. code-block:: python

    >>> # load the data
    >>> d = load("example.tif")
    >>> # save the data as a tiff
    >>> d.save("example_processed.tif")
    >>> # save the data as a png
    >>> d.save("example_processed.png")
    >>> # save the data as an hdf5 file
    >>> d.save("example_processed.hdf5")

Some file formats are much better at maintaining the information about
how you processed your data.  The preferred format in Hyperspy is hdf5,
the hierarchical data format.  This format keeps the most information
possible.

There are optional flags that may be passed to the save function. See
:ref:`saving_files` for more details.

Accessing and setting the metadata
----------------------------------

When loading a file Hyperspy stores all metadata in the Signal 
:py:attr:`~.signal.Signal.original_parameters` attribute. In addition, some of
those metadata and any new metadata generated by Hyperspy are stored in 
:py:attr:`~.signal.Signal.mapped_parameters` attribute. 


.. code-block:: python

   >>> s = load("NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217.msa")
   >>> s.mapped_parameters 
   ├── original_filename = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217.msa
   ├── record_by = spectrum
   ├── signal_origin = 
   ├── signal_type = EELS
   └── title = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217
    
   >>> s.original_parameters 
   ├── DATATYPE = XY
   ├── DATE = 
   ├── FORMAT = EMSA/MAS Spectral Data File
   ├── NCOLUMNS = 1.0
   ├── NPOINTS = 1340.0
   ├── OFFSET = 120.0003
   ├── OWNER = eelsdatabase.net
   ├── SIGNALTYPE = ELS
   ├── TIME = 
   ├── TITLE = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217
   ├── VERSION = 1.0
   ├── XPERCHAN = 0.5
   ├── XUNITS = eV
   └── YUNITS = 
    
   >>> s.set_microscope_parameters(100, 10, 20)
   >>> s.mapped_parameters 
   ├── TEM
   │   ├── EELS
   │   │   └── collection_angle = 20
   │   ├── beam_energy = 100
   │   └── convergence_angle = 10
   ├── original_filename = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217.msa
   ├── record_by = spectrum
   ├── signal_origin = 
   ├── signal_type = EELS
   └── title = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217
   
   >>> s.mapped_parameters.TEM.microscope = "STEM VG"
   >>> s.mapped_parameters
   ├── TEM
   │   ├── EELS
   │   │   └── collection_angle = 20
   │   ├── beam_energy = 100
   │   ├── convergence_angle = 10
   │   └── microscope = STEM VG
   ├── original_filename = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217.msa
   ├── record_by = spectrum
   ├── signal_origin = 
   ├── signal_type = EELS
   └── title = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217


.. _configuring-hyperspy-label:

Configuring hyperspy
--------------------

The behaviour of Hyperspy can be customised using the
:py:class:`~.defaults_parser.Preferences` class. The easiest way to do it is by
calling the :meth:`gui` method:

.. code-block:: python

    >>> preferences.gui()
    
This command should raise the Preferences user interface:

.. _preferences_image:

.. figure::  images/preferences.png
   :align:   center

   Preferences user interface


