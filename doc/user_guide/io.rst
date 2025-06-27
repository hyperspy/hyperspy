.. _io:

Loading and saving data
***********************

.. versionchanged:: 2.0

    File format support in HyperSpy is provided by the separate package 
    :external+rsciio:doc:`RosettaSciIO <index>`. The IO plugins formerly 
    developed within HyperSpy have been moved to RosettaSciIO in order to 
    facilitate a wider use also by other packages. 
    
    **For file format information and documentation**, please refer to:
    
    * :external+rsciio:ref:`Supported File Formats <supported-formats>` - Complete list of supported formats
    * :external+rsciio:doc:`RosettaSciIO Documentation <index>` - Full documentation for file I/O functionality
    * `RosettaSciIO GitHub Repository <https://github.com/hyperspy/rosettasciio>`_ - Source code and development
    
    **Reporting Issues**: File format specific issues should be reported to the 
    `RosettaSciIO issue tracker <https://github.com/hyperspy/rosettasciio/issues>`_. 
    Plugins supporting additional formats or corrections/enhancements to existing 
    plugins should be contributed to the RosettaSciIO repository.

.. _loading_files:

Loading
=======

Basic usage
-----------

HyperSpy can read and write to multiple formats through the 
:external+rsciio:doc:`RosettaSciIO <index>` library (see :external+rsciio:ref:`supported-formats`). 
RosettaSciIO provides the file format support that powers HyperSpy's loading and saving capabilities.

To load data use the :func:`~.load` command. For example, to load the
image ``spam.jpg``, you can type:

.. code-block:: python

    >>> s = hs.load("spam.jpg") # doctest: +SKIP

If loading was successful, the variable ``s`` contains a HyperSpy signal or any
type of signal defined in one of the :ref:`HyperSpy extensions <hyperspy_extensions-label>`, 
see :ref:`load_specify_signal_type-label` for more details.

.. note::

   When the file contains several datasets, the :func:`~.api.load` function
   will return a list of HyperSpy signals, instead of a single HyperSpy signal.
   Each signal can then be accessed using list indexation.

   .. code-block:: python

      >>> s = hs.load("spameggsandham.hspy") # doctest: +SKIP
      >>> s # doctest: +SKIP
      [<Signal1D, title: spam, dimensions: (32,32|1024)>,
       <Signal1D, title: eggs, dimensions: (32,32|1024)>,
       <Signal1D, title: ham, dimensions: (32,32|1024)>]

   Using indexation to access the first signal (index 0):

   .. code-block:: python

      >>> s[0] # doctest: +SKIP
      <Signal1D, title: spam, dimensions: (32,32|1024)>


.. HINT::

   The load function returns an object that contains data read from the file.
   We assign this object to the variable ``s`` but you can choose any (valid)
   variable name you like. for the filename, don\'t forget to include the
   quotation marks and the file extension.

If no argument is passed to the load function, a window will be raised that
allows to select a single file through your OS file manager, e.g.:

.. code-block:: python

    >>> # This raises the load user interface
    >>> s = hs.load() # doctest: +SKIP

It is also possible to load multiple files at once or even stack multiple
files. For more details read :ref:`load-multiple-label`.

Specifying file format
----------------------

HyperSpy (via RosettaSciIO) will attempt to infer the appropriate file reader to use based on
the file extension (for example ``.hspy``, ``.emd`` and so on). You can
override this using the ``file_format`` keyword:

.. code-block:: python

    # Load a .hspy file with an unknown extension
    >>> s = hs.load("filename.some_extension", file_format="hspy") # doctest: +SKIP

For a complete list of supported file formats and their capabilities, see the
:external+rsciio:ref:`supported-formats` documentation in RosettaSciIO.

.. _load_specify_signal_type-label:

Specifying signal type
----------------------

HyperSpy will attempt to infer the most suitable signal type for the data being
loaded. Domain specific signal types are provided by :ref:`extension libraries
<hyperspy_extensions-label>`. To list the signal types
available on your local installation use:

.. code-block:: python

    >>> hs.print_known_signal_types() # doctest: +SKIP

When loading data, the signal type can be specified by providing the ``signal_type``
keyword, which has to correspond to one of the available subclasses of signal
(The ``EELS`` signal type is provided by the extension :external+exspy:ref:`eXSpy <user_guide>`):

.. code-block:: python

    >>> s = hs.load("filename", signal_type="EELS") # doctest: +SKIP

If the loaded file contains several datasets, the :func:`~.api.load`
function will return a list of the corresponding signals:

.. code-block:: python

    >>> s = hs.load("spameggsandham.hspy") # doctest: +SKIP
    >>> s # doctest: +SKIP
    [<Signal1D, title: spam, dimensions: (32,32|1024)>,
    <Signal1D, title: eggs, dimensions: (32,32|1024)>,
    <Signal1D, title: ham, dimensions: (32,32|1024)>]

.. note::

    Note for python programmers: the data is stored in a numpy array
    in the :attr:`~.api.signals.BaseSignal.data` attribute, but you will not
    normally need to access it there.

Metadata
--------

Most scientific file formats store some extra information about the data and the
conditions under which it was acquired (metadata). HyperSpy reads most of them and
stores them in the :attr:`~.api.signals.BaseSignal.original_metadata` attribute.
Also, depending on the file format, a part of this information will be mapped by
HyperSpy to the :attr:`~.api.signals.BaseSignal.metadata` attribute, where it can
for example be used by routines operating on the signal. See the :ref:`metadata structure
<metadata_structure>` for details.

.. note::

    Extensive metadata can slow down loading and processing, and
    loading the :attr:`~.api.signals.BaseSignal.original_metadata` can be disabled
    using the ``load_original_metadata`` argument of the :func:`~.load`
    function. If this argument is set to `False`, the
    :attr:`~.api.signals.BaseSignal.metadata` will still be populated.

To print the content of the attributes simply use:

.. code-block:: python

    >>> s.original_metadata # doctest: +SKIP
    >>> s.metadata # doctest: +SKIP

The :attr:`~.api.signals.BaseSignal.original_metadata` and
:attr:`~.api.signals.BaseSignal.metadata` can be exported to text files
using the :meth:`~.misc.utils.DictionaryTreeBrowser.export` method, e.g.:

.. code-block:: python

    >>> s.original_metadata.export('parameters') # doctest: +SKIP

.. _load_to_memory-label:

Lazy loading of large datasets
------------------------------

.. versionadded:: 1.2
   ``lazy`` keyword argument.

Almost all file readers support `lazy` loading, which means accessing the data
without loading it to memory (see :external+rsciio:ref:`supported-formats` for a
list). This feature can be useful when analysing large files. To use this feature,
set ``lazy`` to ``True`` e.g.:

.. code-block:: python

    >>> s = hs.load("filename.hspy", lazy=True) # doctest: +SKIP

More details on lazy evaluation support can be found in :ref:`big-data-label`.

The units of the navigation and signal axes can be converted automatically
during loading using the ``convert_units`` parameter. If `True`, the
``convert_to_units`` method of the ``axes_manager`` will be used for the conversion
and if set to `False`, the units will not be converted (default).

.. _load-multiple-label:

Loading multiple files
----------------------

Rather than loading files individually, several files can be loaded with a
single command. This can be done by passing a list of filenames to the load
functions, e.g.:

.. code-block:: python

    >>> s = hs.load(["file1.hspy", "file2.hspy"]) # doctest: +SKIP

or by using `shell-style wildcards <https://docs.python.org/library/glob.html>`_:

.. code-block:: python

    >>> s = hs.load("file*.hspy") # doctest: +SKIP

Alternatively, regular expression type character classes can be used such as
``[a-z]`` for lowercase letters or ``[0-9]`` for one digit integers:

.. code-block:: python

    >>> s = hs.load('file[0-9].hspy') # doctest: +SKIP

.. note::

    Wildcards are implemented using ``glob.glob()``, which treats ``*``, ``[``
    and ``]`` as special characters for pattern matching. If your filename or
    path contains square brackets, you may want to set
    ``escape_square_brackets=True``:

    .. code-block:: python

        >>> # Say there are two files like this:
        >>> # /home/data/afile[1x1].hspy
        >>> # /home/data/afile[1x2].hspy

        >>> s = hs.load("/home/data/afile[*].hspy", escape_square_brackets=True) # doctest: +SKIP

HyperSpy also supports ```pathlib.Path`` <https://docs.python.org/3/library/pathlib.html>`_
objects, for example:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> from pathlib import Path

    >>> # Use pathlib.Path
    >>> p = Path("/path/to/a/file.hspy") # doctest: +SKIP
    >>> s = hs.load(p) # doctest: +SKIP

    >>> # Use pathlib.Path.glob
    >>> p = Path("/path/to/some/files/").glob("*.hspy") # doctest: +SKIP
    >>> s = hs.load(p) # doctest: +SKIP

By default HyperSpy will return a list of all the files loaded. Alternatively,
by setting ``stack=True``, HyperSpy can be instructed to stack the data - given
that the files contain data with exactly the same
dimensions. If this is not the case, an error is raised. If each file contains
multiple (N) signals, N stacks will be created. Here, the number of signals
per file must also match, or an error will be raised.

.. code-block:: python

    >>> ls # doctest: +SKIP
    CL1.raw  CL1.rpl  CL2.raw  CL2.rpl  CL3.raw  CL3.rpl  CL4.raw  CL4.rpl
    LL3.raw  LL3.rpl  shift_map-SI3.npy  hdf5/
    >>> s = hs.load('*.rpl') # doctest: +SKIP
    >>> s # doctest: +SKIP
    [<EELSSpectrum, title: CL1, dimensions: (64, 64, 1024)>,
    <EELSSpectrum, title: CL2, dimensions: (64, 64, 1024)>,
    <EELSSpectrum, title: CL3, dimensions: (64, 64, 1024)>,
    <EELSSpectrum, title: CL4, dimensions: (64, 64, 1024)>,
    <EELSSpectrum, title: LL3, dimensions: (64, 64, 1024)>]
    >>> s = hs.load('*.rpl', stack=True) # doctest: +SKIP
    >>> s # doctest: +SKIP
    <EELSSpectrum, title: mva, dimensions: (5, 64, 64, 1024)>

.. _example-data-label:

Loading example data and data from online databases
---------------------------------------------------

HyperSpy is distributed with some example data that can be found in
:mod:`~.api.data`:

.. code-block:: python

    >>> s = hs.data.two_gaussians()
    >>> s.plot()

.. versionadded:: 1.4
    :mod:`~.api.data` (formerly ``hyperspy.api.datasets.artificial_data``)

There are also artificial datasets, which are made to resemble real
experimental data.

.. code-block:: python

    >>> s = hs.data.atomic_resolution_image()
    >>> s.plot()

.. _saving_files:

Saving
======

To save data to a file use the :meth:`~.api.signals.BaseSignal.save` method. The
first argument is the filename and the format is defined by the filename
extension. If the filename does not contain the extension, the default format
(:external+rsciio:ref:`HSpy-HDF5 <hspy-format>`) is used. For example, if the ``s`` variable
contains the :class:`~.api.signals.BaseSignal` that you want to write to a file,
the following will write the data to a file called :file:`spectrum.hspy` in the
default :external+rsciio:ref:`HSpy-HDF5 <hspy-format>` format:

.. code-block:: python

    >>> s.save('spectrum') # doctest: +SKIP

If you want to save to the :external+rsciio:ref:`ripple format <ripple-format>` instead, write:

.. code-block:: python

    >>> s.save('spectrum.rpl') # doctest: +SKIP

Some formats take extra arguments. See the corresponding pages at
:external+rsciio:ref:`supported-formats` for more information.

.. _batch_resaving:

Batch Processing and Re-saving
------------------------------

HyperSpy provides convenient functionality for batch processing and re-saving 
files using the ``tmp_parameters`` that are automatically populated when 
loading files. This is particularly useful when you need to:

* Process multiple files and save them in a different location
* Convert files from one format to another
* Apply the same processing to many files while preserving their original names

The ``tmp_parameters`` contain the original filename, folder, and extension 
information from loaded files, enabling you to save processed data without 
manually specifying filenames.

Basic Re-saving to Different Locations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you load a file with HyperSpy, the ``tmp_parameters`` are automatically 
populated:

.. code-block:: python

    >>> s = hs.load("original_data.hspy")  # doctest: +SKIP
    >>> print(s.tmp_parameters.filename)   # 'original_data'  # doctest: +SKIP
    >>> print(s.tmp_parameters.extension)  # '.hspy'  # doctest: +SKIP
    >>> print(s.tmp_parameters.folder)     # '/path/to/original/'  # doctest: +SKIP

You can then save the signal to a different directory by providing only the 
directory path:

.. code-block:: python

    >>> s.save("/new/output/folder/")  # Saves to /new/output/folder/original_data.hspy  # doctest: +SKIP

Format Conversion
^^^^^^^^^^^^^^^^^

To convert files to different formats, specify the ``file_format`` or 
``extension`` parameter:

.. code-block:: python

    >>> s = hs.load("data.hspy")                       # Load HyperSpy format  # doctest: +SKIP
    >>> s.save("output/", file_format="msa")          # Convert to MSA format  # doctest: +SKIP
    >>> s.save("output/", file_format="rpl")          # Convert to Ripple format  # doctest: +SKIP

Batch Processing Example
^^^^^^^^^^^^^^^^^^^^^^^^

Here's a complete example of batch processing multiple files:

.. code-block:: python

    import hyperspy.api as hs
    from pathlib import Path

    # Define input and output directories
    input_folder = Path("raw_data/")
    output_folder = Path("processed_data/")
    output_folder.mkdir(exist_ok=True)  # Create output directory if it doesn't exist

    # Process all .hspy files in the input directory
    for file_path in input_folder.glob("*.hspy"):
        # Load the signal
        s = hs.load(file_path)
        
        # Apply your processing steps
        s = s.remove_background()
        
        # Save in new location - filename is preserved automatically
        s.save(output_folder)  # Uses original filename from tmp_parameters

    # Convert all files to a different format
    for file_path in input_folder.glob("*.hspy"):
        s = hs.load(file_path)
        # Save as MSA format in output directory
        s.save(output_folder, file_format="msa")

Batch Processing with Different Output Names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need more control over output filenames while still leveraging the 
batch functionality:

.. code-block:: python

    import hyperspy.api as hs
    from pathlib import Path

    input_folder = Path("input_data/")
    output_folder = Path("processed_data/")

    for file_path in input_folder.glob("*.hspy"):
        s = hs.load(file_path)
        
        # Process the signal
        s = s.remove_background()
        
        # Option 1: Use automatic filename with prefix/suffix
        base_name = s.tmp_parameters.filename
        custom_filename = f"processed_{base_name}.hspy"
        s.save(output_folder / custom_filename)
        
        # Option 2: Completely custom filename
        s.save(output_folder / f"{base_name}_cleaned.msa")

.. note::

    The ``tmp_parameters`` are only available for signals that were loaded from 
    files. If you create a signal programmatically, you'll need to provide the 
    full filename when saving.
