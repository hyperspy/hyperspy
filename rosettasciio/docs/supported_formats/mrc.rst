.. _mrc-format:

MRC format
----------

The ``.mrc`` format is widely used for tomographic data. Our implementation is based
on `this specification
<https://www2.mrc-lmb.cam.ac.uk/research/locally-developed-software/image-processing-software/>`_. We also
partly support FEI's custom header. We do not provide writing features for this
format, but, as it is an open format, we may implement this feature in the
future on demand.

For mrc files ``load`` takes the ``mmap_mode`` keyword argument enabling
loading the file using a different mode (default is copy-on-write) . However,
note that lazy loading does not support in-place writing (i.e lazy loading and
the "r+" mode are incompatible).
