.. _jeol-format:

JEOL Analyst Station (ASW, ...)
-------------------------------

This is the file format used by the `JEOL Analysist Station software` for which
hyperspy can read the ``.asw``, ``.pts``, ``.map`` and ``.eds`` format. To read the
calibration, it is required to load the ``.asw`` file, which will load all others
files automatically.

Extra loading arguments
^^^^^^^^^^^^^^^^^^^^^^^

- ``rebin_energy`` : int, default 1.
  Factor used to rebin the energy dimension. It must be a
  factor of the number of channels, typically 4096.
- ``sum_frames`` : bool, default True.
  If False, each individual frame (sweep in JEOL software jargon)
  is loaded. Be aware that loading each individual will use a lot of memory,
  however, it can be used in combination with ``rebin_energy``, ``cutoff_at_kV``
  and ``downsample`` to reduce memory usage.
- ``SI_dtype`` : dtype, default np.uint8.
  set dtype of the eds dataset. Useful to adjust memory usage
  and maximum number of X-rays per channel.
- ``cutoff_at_kV`` : int, float, or None, default None.
  if set (>= 0), use to crop the energy range up the specified energy.
  If ``None``, the whole energy range is loaded.
  Useful to reduce memory usage.
- ``downsample`` : int, default 1.
  the downsample ratio of the navigation dimension of EDS
  dataset, it can be integer or a tuple of length 2 to define ``x`` and ``y``
  separetely and it must be a mutiple of the size of the navigation dimension.
- ``only_valid_data`` : bool, default True.
  for ``.pts`` file only, ignore incomplete and partly
  acquired last frame, which typically occurs when the acquisition was
  interrupted. When loading incomplete data (``only_valid_data=False``),
  the missing data are filled with zeros. If ``sum_frames=True``, this argument
  will be ignored to enforce consistent sum over the mapped area. 
- ``read_em_image`` : bool, default False.
  for ``.pts`` file only, If ``read_em_image=True``,
  read SEM/STEM image from ``.pts`` file if available. In this case, both
  spectrum Image and SEM/STEM Image will be returned as list.
- ``frame_list`` : list of integer or None, default None
  for ``.pts`` file only, frames in frame_list will be loaded.
  for example, ``frame_list=[1,3]`` means second and forth frame will be loaded.
  If ``None``, all frames are loaded.
- ``frame_shifts`` : list of [int, int], list of [int, int, int], or None, default None
  for ``.pts`` file only, each frame will be loaded with offset of
  [dy, dx (, and optionary dEnergy)]. Units are pixels/channels.
  The result of estimate_shift2D() can be used as a parameter of frame_shifts.
  This is useful for express drift correction. Not suitable for accurate analysis.
- ``lazy`` : bool, default False
  for ``.pts`` file only, spectrum image is loaded as a dask.array if lazy == true.
  This is useful to reduce memory usage, with cost of cpu time for calculation.


Example of loading data downsampled, and with energy range cropped with the
original navigation dimension 512 x 512 and the EDS range 40 keV over 4096
channels:

.. code-block:: python

    >>> hs.load("sample40kv.asw", downsample=8, cutoff_at_kV=10)
    [<Signal2D, title: IMG1, dimensions: (|512, 512)>,
     <Signal2D, title: C K, dimensions: (|512, 512)>,
     <Signal2D, title: O K, dimensions: (|512, 512)>,
     <EDSTEMSpectrum, title: EDX, dimensions: (64, 64|1096)>]

load the same file without extra arguments:

.. code-block:: python

    >>> hs.load("sample40kv.asw")
    [<Signal2D, title: IMG1, dimensions: (|512, 512)>,
     <Signal2D, title: C K, dimensions: (|512, 512)>,
     <Signal2D, title: O K, dimensions: (|512, 512)>,
     <EDSTEMSpectrum, title: EDX, dimensions: (512, 512|4096)>]
