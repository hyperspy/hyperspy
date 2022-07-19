.. _msa-format:

EMSA/MSA format
---------------

The ``.msa`` format is an `open standard format
<https://www.microscopy.org/resources/scientific_data/index.cfm>`_
widely used to exchange single spectrum data, but does not support
multidimensional data. It can for example be used to exchange single spectra
with Gatan's Digital Micrograph.

.. WARNING::
    If several spectra are loaded and stacked (``hs.load('pattern', stack_signals=True``)
    the calibration read from the first spectrum and applied to all other spectra.

Extra saving arguments
^^^^^^^^^^^^^^^^^^^^^^

For the ``.msa`` format the ``format`` argument is used to specify whether the
energy axis should also be saved with the data.  The default, 'Y' omits the
energy axis in the file.  The alternative, 'XY', saves a second column with the
calibrated energy data. It is possible to personalise the separator with the
`separator` keyword.

.. Warning::

    However, if a different separator is chosen the resulting file will not
    comply with the MSA/EMSA standard and HyperSpy and other software may not
    be able to read it.

The default encoding is `latin-1`. It is possible to set a different encoding
using the `encoding` argument, e.g.:

.. code-block:: python

    >>> s.save('file.msa', encoding = 'utf8')
