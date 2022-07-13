.. _mrcz-format:

MRCZ
----

MRCZ is an extension of the CCP-EM MRC2014 file format. `CCP-EM MRC2014
<http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ file format.  It uses the
`blosc` meta-compression library to bitshuffle and compress files in a blocked,
multi-threaded environment. The supported data types are:

[`float32`,`int8`,`uint16`,`int16`,`complex64`]

It supports arbitrary meta-data, which is serialized into JSON.

MRCZ also supports asynchronous reads and writes.

Repository: https://github.com/em-MRCZ
PyPI:       https://pypi.python.org/pypi/mrcz
Citation:   Submitted.
Preprint:   http://www.biorxiv.org/content/early/2017/03/13/116533

Support for this format is not enabled by default. In order to enable it
install the `mrcz` and optionally the `blosc` Python packages.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``do_async``: currently supported within HyperSpy for writing only, this will
  save  the file in a background thread and return immediately. Defaults
  to `False`.

.. Warning::

    There is no method currently implemented within Hyperspy to tell if an
    asychronous write has finished.


- ``compressor``: The compression codec, one of [`None`,`'zlib`',`'zstd'`, `'lz4'`].
  Defaults to `None`.
- ``clevel``: The compression level, an `int` from 1 to 9. Defaults to 1.
- ``n_threads``: The number of threads to use for 'blosc' compression. Defaults to
  the maximum number of virtual cores (including Intel Hyperthreading)
  on your system, which is recommended for best performance. If \
  ``do_async = True`` you may wish to leave one thread free for the
  Python GIL.

The recommended compression codec is 'zstd' (zStandard) with `clevel=1` for
general use. If speed is critical, use 'lz4' (LZ4) with `clevel=9`. Integer data
compresses more redably than floating-point data, and in general the histogram
of values in the data reflects how compressible it is.

To save files that are compatible with other programs that can use MRC such as
GMS, IMOD, Relion, MotionCorr, etc. save with `compressor=None`, extension `.mrc`.
JSON metadata will not be recognized by other MRC-supporting software but should
not cause crashes.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

    >>> s.save('file.mrcz', do_async=True, compressor='zstd', clevel=1)

    >>> new_signal = hs.load('file.mrcz')
