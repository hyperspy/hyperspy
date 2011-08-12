Getting started
***************

First steps with eelslab
========================

Starting eelslab
----------------

To start eelslab type in a console:

.. code-block:: bash

    eelslab

.. NOTE::

   If you are using GNOME in Linux, you can open a terminal in a folder by 
   choosing "open terminal" in the file menu if nautilus-open-terminal is 
   installed in your system.
   A similar feature is available in Windows 7 and Windows Vista when pressing 
   the Shift key and the right mouse button. For more comfort in Windows it is 
   recommended to install `Console2 <http://sourceforge.net/projects/console/>`_


Loading and saving data
-----------------------


To load a supported file (e.g. NetCDF, dm3, MSA, MRC, ser, emi or many image 
files) simply type:

.. code-block:: python

    s = load('filename')

.. NOTE::

   We use the variable :guilabel:`s` but you can choose any (valid) variable name

.. NOTE::

   The filename *must* include the extension

If the loading was successful, the variable :guilabel:`s` now contains a python object 
that can be an ``Image`` or ``Spectrum`` instance. EELSLab will try to guess the 
most convenient data type. However, you can force it to read the data as 
a particular data type by providing the keyword ``data_type`` that can take the 
values :guilabel:`SI` or :guilabel:`Image`, e.g.:

.. code-block:: python

    s = load('filename', data_type = 'SI')

.. _configuring-eelslab-label:

Configuring eelslab
-------------------

You can configure some parameters of eelslab by editing the :file:`eelslabrc` 
file. The location of the configuration file depends on the system. 
You can find its path by calling the ```get_configuration_directory_path``` 
function in the eelslab prompt:

.. code-block:: bash

    get_configuration_directory_path()


Alternatively it is possible to change the same parameters at runtime by changing 
the attributes of the defaults class. For example, to plot automatically the 
data when loading it:

.. code-block:: bash

    # First we load some data
    s = load('YourDataFilenameHere')
    # (in the defaults setting nothing is plotted, unless you can changed the 
    # defaults in the eelslabrc file)
    #
    # Now we will change the setting at runtime
    defaults.plot_on_load = True
    s = load('YourDataFilenameHere')
    # The data should have been automatically plotted.



.. _getting-help-label:

Getting help
------------

The documentation can be accessed by adding a question mark to the name of a function. e.g.:

.. code-block:: python
    
    load?

This syntax is one of the many features of `IPython <http://ipython.scipy.org/moin/>`_

Please note that the documentation of the code is a work in progress, so not all the objects are documented yet.

Autocompletion
--------------


Another useful `IPython <http://ipython.scipy.org/moin/>`_ feature is the 
autocompletion of commands and filenames. It is highly recommended to read the 
`Ipython documentation <http://ipython.scipy.org/moin/Documentation>`_.

Data visualisation
==================

The Spectrum and Image objects have a ``plot`` method.

.. code-block:: python
    
    s = load('YourDataFilenameHere')
    s.plot()

if the object is single spectrum or an image one window will appear when calling 
the plot method. If the object is a 2D or 3D SI two figures will appear, 
one containing a plot of a spectrum of the dataset and the other a 2D 
representation of the data. 

To explore an SI drag the cursor present in the 2D data representation 
(it can be a line for 2D SIs or a square for 3D SIs). 
An extra cursor can be added by pressing the ``e`` key. Pressing ``e`` once more will 
disable the extra cursor.

When exploring a 2D SI of high spatial resolution the default size of the
rectangular cursors can be too small to be dragged or even seen. It is possible to change
the size of the cursors by pressing the ``+`` and ``-`` keys  **when the navigator
windows is on focus**.

It is also possible to explore an SI by using the numpad arrows 
**when numlock is on and the spectrum or navigator figure is on focus**. 
When using the numpad arrows the PageUp and PageDown keys change the size of the step.

The same keys can be used to explore an image stack.



=========   =============================
key         function    
=========   =============================
e           Switch second pointer on/off
Arrows      Change coordinates  
PageUp      Increase step size
PageDown    Decrease step size
``+``           Increase pointer size
``-``           Decrease pointer size
=========   =============================


To close all the figures type:

.. code-block:: python

    close('all')


This is a `matplotlib <http://matplotlib.sourceforge.net/>`_ command. 
Matplotlib is the library that eelslab uses to produce the plots. To learn how 
to pan/zoom and more a matplotlib plot 
`check here <http://matplotlib.sourceforge.net/users/navigation_toolbar.html>`_


