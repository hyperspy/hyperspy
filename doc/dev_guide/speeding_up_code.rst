
Contributing cython code
========================

Python is not the fastest language, and can be particularly slow in loops.
Performance can sometimes be significantly improved by implementing optional
cython code alongside the pure Python versions. While developing cython code,
make use of the official cython recommendations (http://docs.cython.org/).  Add
your cython extensions to the setup.py, in the existing list of
``raw_extensions``.

Unlike the cython recommendation, the cythonized .c or .cpp files are not
welcome in the git source repository (except original c or c++ files), since
they are typically quite large. Cythonization will take place during Travis
CI and Appveyor building. The cythonized code will be generated and included
in source or binary distributions for end users. To help troubleshoot
potential deprecation with future cython releases, add a comment with in the
header of your .pyx files with the cython version. If cython is present in
the build environment and any cythonized c/c++ file is missing, then setup
.py tries to cythonize all extensions automatically.

To make the development easier the new command ``recythonize`` has been added
to setup.py.  It can be used in conjunction with other default commands.  For
example ``python setup.py recythonize build_ext --inplace`` will recythonize
all changed (and described in setup.py!) cython code and compile.

When developing on git branches, the first time you call setup.py in
conjunction with or without any other command - it will generate a
post-checkout hook, which will include a potential cythonization and
compilation product list (.c/.cpp/.so/.pyd). With your next ``git checkout``
the hook will remove them and automatically run ``python setup.py build_ext
--inplace`` to cythonize and compile the code if available.  If an older
version of HyperSpy (<= 0.8.4.x) is checked out this should have no side
effects.

If another custom post-checkout hook is detected on PR, then setup.py tries to
append or update the relevant part. To prevent unwanted hook generation or
update you can create the empty file ``.hook_ignore`` in source directory (same
level as setup.py).