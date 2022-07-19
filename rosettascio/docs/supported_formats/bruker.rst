.. _bruker-format:

Bruker formats
--------------

.. _bcf-format:

Bruker composite file (BCF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy can read "hypermaps" saved with Bruker's Esprit v1.x or v2.x in ``.bcf``
hybrid (virtual file system/container with xml and binary data, optionally
compressed) format. Most ``.bcf``` import functionality is implemented. Both
high-resolution 16-bit SEM images and hyperspectral EDX data can be retrieved
simultaneously.

BCF can look as all inclusive format, however it does not save some key EDX
parameters: any of dead/live/real times, FWHM at Mn_Ka line. However, real time
for whole map is calculated from pixelAverage, lineAverage, pixelTime,
lineCounter and map height parameters.

Note that Bruker Esprit uses a similar format for EBSD data, but it is not
currently supported by HyperSpy.

Extra loading arguments
+++++++++++++++++++++++

- ``select_type`` : one of (None, 'spectrum', 'image'). If specified, only the
  corresponding type of data, either spectrum or image, is returned.
  By default (None), all data are loaded.
- ``index`` : one of (None, int, "all"). Allow to select the index of the dataset
  in the ``.bcf`` file, which can contains several datasets. Default None value
  result in loading the first dataset. When set to 'all', all available datasets
  will be loaded and returned as separate signals.
- ``downsample`` : the downsample ratio of hyperspectral array (height and width
  only), can be integer >=1, where '1' results in no downsampling (default 1).
  The underlying method of downsampling is unchangeable: sum. Differently than
  ``block_reduce`` from skimage.measure it is memory efficient (does not creates
  intermediate arrays, works inplace).
- ``cutoff_at_kV`` : if set (can be None, int, float (kV), one of 'zealous'
  or 'auto') can be used either to crop or enlarge energy (or number of
  channels) range at max values. It can be used to conserve memory or enlarge
  the range if needed to mach the size of other file. Default value is None
  (which does not influence size). Numerical values should be in kV.
  'zealous' truncates to the last non zero channel (this option
  should not be used for stacks, as low beam current EDS can have different
  last non zero channel per slice). 'auto' truncates channels to SEM/TEM
  acceleration voltage or energy at last channel, depending which is smaller.
  In case the hv info is not there or hv is off (0 kV) then it fallbacks to
  full channel range.

Example of loading reduced (downsampled, and with energy range cropped)
"spectrum only" data from ``.bcf`` (original shape: 80keV EDS range (4096 channels),
100x75 pixels; SEM acceleration voltage: 20kV):

.. code-block:: python

    >>> hs.load("sample80kv.bcf", select_type='spectrum', downsample=2, cutoff_at_kV=10)
    <EDSSEMSpectrum, title: EDX, dimensions: (50, 38|595)>

load the same file with limiting array size to SEM acceleration voltage:

.. code-block:: python

    >>> hs.load("sample80kv.bcf", cutoff_at_kV='auto')
    [<Signal2D, title: BSE, dimensions: (|100, 75)>,
    <Signal2D, title: SE, dimensions: (|100, 75)>,
    <EDSSEMSpectrum, title: EDX, dimensions: (100, 75|1024)>]

The loaded array energy dimension can by forced to be larger than the data
recorded by setting the 'cutoff_at_kV' kwarg to higher value:

.. code-block:: python

    >>> hs.load("sample80kv.bcf", cutoff_at_kV=60)
    [<Signal2D, title: BSE, dimensions: (|100, 75)>,
    <Signal2D, title: SE, dimensions: (|100, 75)>,
    <EDSSEMSpectrum, title: EDX, dimensions: (100, 75|3072)>]

loading without setting cutoff_at_kV value would return data with all 4096
channels. Note that setting downsample to >1 currently locks out using SEM
images for navigation in the plotting.

.. _spx-format:

SPX (xml) format
^^^^^^^^^^^^^^^^

Hyperspy can read Bruker's ``.spx`` format (single spectra format based on XML).
The format contains extensive list of details and parameters of EDS analyses
which are mapped in hyperspy to metadata and original_metadata dictionaries.
