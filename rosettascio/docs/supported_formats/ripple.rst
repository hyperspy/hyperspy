.. _ripple-format:

Ripple
------

This *open standard format* developed at NIST as native format for
`Lispix <http://www.nist.gov/lispix/>`_ is widely used to exchange
multidimensional data. However, it only supports data of up to three
dimensions. It can also be used to exchange data with Bruker and used in
combination with the :ref:`import-rpl` it is very useful for exporting data
to Gatan's Digital Micrograph.

The default encoding is latin-1. It is possible to set a different encoding
using the encoding argument, e.g.:

.. code-block:: python

    >>> s.save('file.rpl', encoding = 'utf8')


For mrc files ``load`` takes the ``mmap_mode`` keyword argument enabling
loading the file using a different mode (default is copy-on-write) . However,
note that lazy loading does not support in-place writing (i.e lazy loading and
the "r+" mode are incompatible).
