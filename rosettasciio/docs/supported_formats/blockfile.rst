.. _blockfile-format:

Blockfile
---------

RosettaSciIO can read and write the blockfile format from NanoMegas ASTAR software.
It is used to store a series of diffraction patterns from scanning precession
electron diffraction (SPED) measurements, with a limited set of metadata. The
header of the blockfile contains information about centering and distortions
of the diffraction patterns, but is not applied to the signal during reading.
Blockfiles only support data values of type
`np.uint8 <http://docs.scipy.org/doc/numpy/user/basics.types.html>`_ (integers
in range 0-255).

.. warning::

   While Blockfiles are supported, it is a proprietary format, and future
   versions of the format might therefore not be readable. Complete
   interoperability with the official software can neither be guaranteed.

Blockfiles are by default loaded in a "copy-on-write" manner using
`numpy.memmap
<http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html>`_ .
For blockfiles ``load`` takes the ``mmap_mode`` keyword argument enabling
loading the file using a different mode. However, note that lazy loading
does not support in-place writing (i.e lazy loading and the "r+" mode
are incompatible).

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``intensity_scaling`` : in case the dataset that needs to be saved does not
  have the `np.uint8` data type, casting to this datatype without intensity
  rescaling results in overflow errors (default behavior). This option allows
  you to perform linear intensity scaling of the images prior to saving the
  data. The options are:

  - `'dtype'`: the limits of the datatype of the dataset, e.g. 0-65535 for
    `np.uint16`, are mapped onto 0-255 respectively. Does not work for `float`
    data types.
  - `'minmax'`: the minimum and maximum in the dataset are mapped to 0-255.
  - `'crop'`: everything below 0 and above 255 is set to 0 and 255 respectively
  - 2-tuple of `floats` or `ints`: the intensities between these values are
    scaled between 0-255, everything below is 0 and everything above is 255.
- ``navigator_signal``: the BLO file also stores a virtual bright field (VBF) image which
  behaves like a navigation signal in the ASTAR software. By default this is
  set to `'navigator'`, which results in the default :py:attr:`navigator` signal to be used.
  If this signal was not calculated before (e.g. by calling :py:meth:`~.signal.BaseSignal.plot`), it is
  calculated when :py:meth:`~.signal.BaseSignal.save` is called, which can be time consuming.
  Alternatively, setting the argument to `None` will result in a correctly sized
  zero array to be used. Finally, a custom ``Signal2D`` object can be passed,
  but the shape must match the navigation dimensions.
