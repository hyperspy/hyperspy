.. _emd-format:

Electron Microscopy Dataset (EMD)
---------------------------------

EMD stands for “Electron Microscopy Dataset”. It is a subset of the open source
HDF5 wrapper format. N-dimensional data arrays of any standard type can be
stored in an HDF5 file, as well as tags and other metadata.

.. _emd_ncem-format:

EMD (NCEM)
^^^^^^^^^^

This `EMD format <https://emdatasets.com>`_ was developed by Colin Ophus at the
National Center for Electron Microscopy (NCEM).
This format is used by the `prismatic software <https://prism-em.com/docs-outputs/>`_
to save the simulation outputs.

Extra loading arguments
+++++++++++++++++++++++

- ``dataset_path`` : None, str or list of str. Path of the dataset. If None,
  load all supported datasets, otherwise the specified dataset(s).
- ``stack_group`` : bool, default is True. Stack datasets of groups with common
  path. Relevant for emd file version >= 0.5 where groups can be named
  'group0000', 'group0001', etc.
- ``chunks`` : None, True or tuple. Determine the chunking of the dataset to save.
  See the ``chunks`` arguments of the ``hspy`` file format for more details.


For files containing several datasets, the `dataset_name` argument can be
used to select a specific one:

.. code-block:: python

    >>> s = hs.load("adatafile.emd", dataset_name="/experimental/science_data_1/data")


Or several by using a list:

.. code-block:: python

    >>> s = hs.load("adatafile.emd",
    ...             dataset_name=[
    ...                 "/experimental/science_data_1/data",
    ...                 "/experimental/science_data_2/data"])


.. _emd_fei-format:

EMD (Velox)
^^^^^^^^^^^

This is a non-compliant variant of the standard EMD format developed by
Thermo-Fisher (former FEI). RosettaSciIO supports importing images, EDS spectrum and EDS
spectrum streams (spectrum images stored in a sparse format). For spectrum
streams, there are several loading options (described below) to control the frames
and detectors to load and if to sum them on loading.  The default is
to import the sum over all frames and over all detectors in order to decrease
the data size in memory.


.. note::

    Pruned Velox EMD files only contain the spectrum image in a proprietary
    format that RosettaSciIO cannot read. Therefore, don't prune Velox EMD files
    if you intend to read them with RosettaSciIO.

.. code-block:: python

    >>> hs.load("sample.emd")
    [<Signal2D, title: HAADF, dimensions: (|179, 161)>,
    <EDSSEMSpectrum, title: EDS, dimensions: (179, 161|4096)>]

.. note::

    When using `HyperSpy <https://hyperspy.org>`_, FFTs made in Velox are loaded
    in as-is as a HyperSpy ComplexSignal2D object.
    The FFT is not centered and only positive frequencies are stored in the file.
    Making FFTs with HyperSpy from the respective image datasets is recommended.

.. note::

    When using `HyperSpy <https://hyperspy.org>`_, DPC data is loaded in as a HyperSpy ComplexSignal2D object.

.. note::

    Currently only lazy uncompression rather than lazy loading is implemented.
    This means that it is not currently possible to read EDS SI Velox EMD files
    with size bigger than the available memory.


.. warning::

   This format is still not stable and files generated with the most recent
   version of Velox may not be supported. If you experience issues loading
   a file, please report it  to the RosettaSciIO developers so that they can
   add support for newer versions of the format.


.. _Extra-loading-arguments-fei-emd:

Extra loading arguments
+++++++++++++++++++++++

- ``select_type`` : one of {None, 'image', 'single_spectrum', 'spectrum_image'} (default is None).
- ``first_frame`` : integer (default is 0).
- ``last_frame`` : integer (default is None)
- ``sum_frames`` : boolean (default is True)
- ``sum_EDS_detectors`` : boolean (default is True)
- ``rebin_energy`` : integer (default is 1)
- ``SI_dtype`` : numpy dtype (default is None)
- ``load_SI_image_stack`` : boolean (default is False)

The ``select_type`` parameter specifies the type of data to load: if `image` is selected,
only images (including EDS maps) are loaded, if `single_spectrum` is selected, only
single spectra are loaded and if `spectrum_image` is selected, only the spectrum
image will be loaded. The ``first_frame`` and ``last_frame`` parameters can be used
to select the frame range of the EDS spectrum image to load. To load each individual
EDS frame, use ``sum_frames=False`` and the EDS spectrum image will be loaded
with an extra navigation dimension corresponding to the frame index
(time axis). Use the ``sum_EDS_detectors=True`` parameter to load the signal of
each individual EDS detector. In such a case, a corresponding number of distinct
EDS signal is returned. The default is ``sum_EDS_detectors=True``, which loads the
EDS signal as a sum over the signals from each EDS detectors.  The ``rebin_energy``
and ``SI_dtype`` parameters are particularly useful in combination with
``sum_frames=False`` to reduce the data size when one want to read the
individual frames of the spectrum image. If ``SI_dtype=None`` (default), the dtype
of the data in the emd file is used. The ``load_SI_image_stack`` parameter allows
loading the stack of STEM images acquired simultaneously as the EDS spectrum image.
This can be useful to monitor any specimen changes during the acquisition or to
correct the spatial drift in the spectrum image by using the STEM images.

.. code-block:: python

    >>> hs.load("sample.emd", sum_EDS_detectors=False)
    [<Signal2D, title: HAADF, dimensions: (|179, 161)>,
    <EDSSEMSpectrum, title: EDS - SuperXG21, dimensions: (179, 161|4096)>,
    <EDSSEMSpectrum, title: EDS - SuperXG22, dimensions: (179, 161|4096)>,
    <EDSSEMSpectrum, title: EDS - SuperXG23, dimensions: (179, 161|4096)>,
    <EDSSEMSpectrum, title: EDS - SuperXG24, dimensions: (179, 161|4096)>]

    >>> hs.load("sample.emd", sum_frames=False, load_SI_image_stack=True, SI_dtype=np.int8, rebin_energy=4)
    [<Signal2D, title: HAADF, dimensions: (50|179, 161)>,
    <EDSSEMSpectrum, title: EDS, dimensions: (50, 179, 161|1024)>]
