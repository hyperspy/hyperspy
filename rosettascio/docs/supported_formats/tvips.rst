.. _tvips-format:

TVIPS format
------------

The TVIPS format is the default format for image series collected by pixelated
cameras from the TVIPS company. Typically individual images captured by these
cameras are stored in the :ref:`TIFF format<tiff-format>` which can also be 
loaded by Hyperspy. This format instead serves to store image streams from 
in-situ and 4D-STEM experiments. During collection, the maximum file size is
typically capped meaning the dataset is typically split over multiple files
ending in `_xyz.tvips`. The `_000.tvips` will contain the main header and
it is essential for loading the data. If a filename is provided for loading
or saving without a `_000` suffix, this will automatically be added. Loading
will not work if no such file is found.

.. warning::

   While TVIPS files are supported, it is a proprietary format, and future
   versions of the format might therefore not be readable. Complete
   interoperability with the official software can neither be guaranteed.

.. warning::
    
   The TVIPS format currently stores very limited amount of metadata about
   scanning experiments. To reconstruct scan data, e.g. 4D-STEM datasets,
   parameters like the shape and scales of the scan dimensions should be
   manually recorded.

Extra loading arguments
^^^^^^^^^^^^^^^^^^^^^^^
- ``scan_shape``: a tuple of integers to indicate the shape of the navigation
  axes. For example, `(3, 4)` will have 3 scan points in the y direction and 4
  in the x direction. If this argument is not provided, the data will be loaded
  as a 1D stack of images. `auto` is also an option which aims to reconstruct
  the scan based on the `rotidx` indices in frame headers. Since this only
  works for square scan grids and is prone to failure, this option is not 
  recommended.
- ``scan_start_frame``: index of the first frame of the dataset to consider,
  mainly relevant for 4D-STEM datasets. If `scan_shape="auto"` this parameter
  is ignored.
- ``winding_scan_axis``: if the acquisition software collected data without
  beam flyback but with a winding "snake" scan, then every second scan row
  or column needs to be reversed to make sense of the data. This can be
  indicated with values `"x"` or `"y"`, depending on whether winding happened
  along the primary or secondary axis. By default, flyback scan without winding
  is assumed with `x` the fast scan and `y` the slow scan direction.
- ``hysteresis``: if winding scan was active it is likely there is an overshoot
  of a few pixels (2-5) every second scan row. This parameter shifts every
  second row by the indicated number of scan points to align even and odd scan
  rows. Default is 0, no hysteresis.
- ``rechunking``: only relevant when using lazy loading. If set to `False`
  each TVIPS file is loaded as a single chunk. For a better experience, with the
  default setting of `auto` rechunking is performed such that the navigation axes
  are optimally chunked and the signal axes are not chunked. If set to anything else, the
  value will be passed to the `chunks` argument in `dask.array.rechunk`.
  
Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``max_file_size``: approximate maximum size of individual files in bytes. 
  In this way a dataset can be split into multiple files. A file needs to be
  at least the size of the main header in the first file plus one frame and its
  frame header. By default there is no maximum and the entire dataset is saved
  to one file.
- ``version``: TVIPS file format version, defaults to version 2. Only version
  1 and 2 are currently supported.
- ``frame_header_extra_bytes``: the number of bytes to pad the frame headers
  with. By default this is 0.
- ``mode``: `1` for imaging, `2` for diffraction. By default the mode is
  guessed from the signal type and signal unites.
