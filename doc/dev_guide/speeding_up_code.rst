
Speeding up code
================

Python is not the fastest language, but this is not usually an issue because
most scientific Python software uses libraries written in compiled languages
such as Numpy for data processing, hence running at close to C-speed.
Nevertheless, sometimes it is necessary to improve the speed of some parts of
the code by writing some functions
in compiled languages or by using Just-in-time (JIT) compilation. Before taking
this approach, please make
sure that the extra complexity is worth it by writing a first implementation of
the functionality using Python and Numpy and profiling your code.

Writing Numba code
------------------

If you need to improve the speed of a given part of the code your first choice
should be `Numba <https://numba.pydata.org/>`_. The motivation is that Numba
code is very similar (when not identical) to Python code, and therefore, it is
a lot easier to maintain than Cython code (see below).

Numba is also a required dependency for HyperSpy, unlike Cython which
is only an optional dependency.

Writing Cython code
-------------------

Cython code should only be considered if:

1. It is not possible to speed up the function using Numba, and instead,
2. it is accompanied by a pure Python
   version of the same code that behaves exactly in the same way when the
   compiled C extension is not present. This extra version is required because
   we may not be able to provide binaries for all platforms and not all users
   will be able to compile C code in their platforms.

Please read through the official Cython recommendations
(http://docs.cython.org/) before writing Cython code.

To help troubleshoot potential deprecations in future Cython releases, add a
comment in the header of your .pyx files stating the Cython version you used
when writing the code.

Note that the "cythonized" .c or .cpp files are not welcome in the git source
repository because they are typically very large.

Once you have written your Cython files, add them to ``raw_extensions`` in
``setup.py``.

Compiling Cython code
^^^^^^^^^^^^^^^^^^^^^

If Cython is present in
the build environment and any cythonized c/c++ file is missing, then
``setup.py`` tries to cythonize all extensions automatically.

To make the development easier ``setup.py`` provides a ``recythonize`` command
that can be used in conjunction with default commands.  For
example

.. code-block:: bash

   python setup.py recythonize build_ext --inplace

will recythonize all Cython code and compile it.

Cythonization and compilation will also take place during continous
integration (CI).
