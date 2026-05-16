.. _lazy_import-label:

Lazy import
===========

HyperSpy uses lazy imports to reduce the initial import time and memory
footprint, as described in `SPEC <https://scientific-python.org/specs/spec-0001/>`_.

The following patterns are used in HyperSpy for lazy imports:

- Libraries such as ``scipy``, ``scikit-image`` are imported using, for example,
  ``import scipy`` instead of ``from scipy import ndimage`` to avoid loading
  submodules until they are needed.
- Deferring expensive imports, such as ``dask``, ``matplotlib`` or ``pint`` until
  they are needed inside functions or methods.
- ``__getattr__`` and ``__dir__`` are used to implement lazy imports of HyperSpy
  objects, as described in `PEP 562 <https://peps.python.org/pep-0562/>`_.
- Lazily the import of signals and define lazy signals in separate modules from
  their non-signal counterparts to import dask only when needed, for example:

  In the module ``my_new_signal.py``

  .. code:: python

    from hyperspy import signals 

    class MyNewSignal(signals.BaseSignal):
        # all of my great functions. 

  In the module ``lazy_my_new_signal.py``
  
  .. code:: python

    from hyperspy import signals
    from my_package.my_new_signal import MyNewSignal

    class LazyMyNewSignal(MyNewSignal, signals.LazySignal):
        """ Lazy signal for my new signal class"""
