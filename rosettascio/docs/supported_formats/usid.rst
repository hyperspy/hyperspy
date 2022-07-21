.. _usid-format:

Universal Spectroscopy and Imaging Data (h5USID)
------------------------------------------------

Background
^^^^^^^^^^

`Universal Spectroscopy and Imaging Data <https://pycroscopy.github.io/USID/about.html>`_
(USID) is an open, community-driven, self-describing, and standardized schema for
representing imaging and spectroscopy data of any size, dimensionality, precision,
instrument of origin, or modality. USID data is typically stored in
Hierarchical Data Format Files (HDF5) and the combination of USID within ``.hdf5``
files is referred to as h5USID.

`pyUSID <https://pycroscopy.github.io/pyUSID/about.html>`_
provides a convenient interface to I/O operations on such h5USID files. USID
(via pyUSID) forms the foundation for other materials microscopy scientific
python package called `pycroscopy <https://pycroscopy.github.io/pycroscopy/about.html>`_.
If you have any questions regarding this module, please consider
`contacting <https://pycroscopy.github.io/pyUSID/contact.html>`_
the developers of pyUSID.

Requirements
^^^^^^^^^^^^

1. Reading and writing h5USID files require the
   `installation of pyUSID <https://pycroscopy.github.io/pyUSID/install.html>`_.
2. Files must use the ``.h5`` file extension in order to use this IO plugin.
   Using the ``.hdf5`` extension will default to the :ref:`HyperSpy plugin <hspy-format>`.

Reading
^^^^^^^

h5USID files can contain multiple USID datasets within the same file.
RosettaSciIO supports reading in one or more USID datasets.

Extra loading arguments
+++++++++++++++++++++++

- ``dataset_path``: str. Absolute path of USID Main HDF5 dataset.
  (default is ``None`` - all USID Main Datasets will be read)
- ``ignore_non_linear_dims``: bool, default is True. If True, parameters that
  were varied non-linearly in the desired dataset will result in Exceptions.
  Else, all such non-linearly varied parameters will be treated as
  linearly varied parameters and a Signal object will be generated.


Reading the sole dataset within a h5USID file:

.. code-block:: python

    >>> hs.load("sample.h5")
    <Signal2D, title: HAADF, dimensions: (|128, 128)>

If multiple datasets are present within the h5USID file and you try the same command again,
**all** available datasets will be loaded.

.. note::

    Given that HDF5 files can accommodate very large datasets, setting ``lazy=True``
    is strongly recommended if the contents of the HDF5 file are not known apriori.
    This prevents issues with regard to loading datasets far larger than memory.

    Also note that setting ``lazy=True`` leaves the file handle to the HDF5 file open.
    If it is important that the files be closed after reading, set ``lazy=False``.

.. code-block:: python

    >>> hs.load("sample.h5")
    [<Signal2D, title: HAADF, dimensions: (|128, 128)>,
    <Signal1D, title: EELS, dimensions: (|64, 64, 1024)>]

We can load a specific dataset using the ``dataset_path`` keyword argument. Setting it to the
absolute path of the desired dataset will cause the single dataset to be loaded.

.. code-block:: python

    >>> # Loading a specific dataset
    >>> hs.load("sample.h5", dataset_path='/Measurement_004/Channel_003/Main_Data')
    <Signal2D, title: HAADF, dimensions: (|128, 128)>

h5USID files support the storage of HDF5 dataset with
`compound data types <https://pycroscopy.github.io/USID/usid_model.html#compound-datasets>`_.
As an (*oversimplified*) example, one could store a color image using a compound data type that allows
each color channel to be accessed by name rather than an index.
Naturally, reading in such a compound dataset into HyperSpy will result in a separate
signal for each named component in the dataset:

.. code-block:: python

    >>> hs.load("file_with_a_compound_dataset.h5")
    [<Signal2D, title: red, dimensions: (|128, 128)>,
    Signal2D, title: blue, dimensions: (|128, 128)>,
    Signal2D, title: green, dimensions: (|128, 128)>]

h5USID files also support parameters or dimensions that have been varied non-uniformly.
This capability is important in several spectroscopy techniques where the bias is varied as a
`bi-polar triangular waveform <https://pycroscopy.github.io/pyUSID/auto_examples/beginner/plot_usi_dataset.html#values-for-each-dimension>`_
rather than uniformly from the minimum value to the maximum value.
So far, the reading of non-uniform axes is not implemented in RosettaSciIO and as the non-uniform information
would be lost in the axes manager, the USID plugin will default to a warning
when it encounters a parameter that has been varied non-uniformly:

.. code-block:: python

    >>> hs.load("sample.h5")
    UserWarning: Ignoring non-uniformity of dimension: Bias
    <BaseSignal, title: , dimensions: (|7, 3, 5, 2)>

In order to prevent accidental misinterpretation of information downstream, the keyword argument
``ignore_non_uniform_dims`` can be set to ``False`` which will result in a ``ValueError`` instead.

.. code-block:: python

    >>> hs.load("sample.h5")
    ValueError: Cannot load provided dataset. Parameter: Bias was varied non-uniformly.
    Supply keyword argument "ignore_non_uniform_dims=True" to ignore this error

Writing
^^^^^^^

Signals can be written to new h5USID files using the standard :py:meth:`~.signal.BaseSignal.save` function.
Setting the ``overwrite`` keyword argument to ``True`` will append to the specified
HDF5 file. All other keyword arguments will be passed to
`pyUSID.hdf_utils.write_main_dataset() <https://pycroscopy.github.io/pyUSID/_autosummary/_autosummary/pyUSID.io.hdf_utils.html#pyUSID.io.hdf_utils.write_main_dataset>`_

.. code-block:: python

    >>> sig.save("USID.h5")

Note that the model and other secondary data artifacts linked to the signal are not
written to the file but these can be implemented at a later stage.
