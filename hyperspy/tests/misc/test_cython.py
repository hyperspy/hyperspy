import nose.tools as nt

import os

from nose.plugins.skip import SkipTest

my_path = os.path.dirname(__file__)


def test_cythonization():
    try:
        import Cython
    except ImportError:
        raise SkipTest

    if not os.path.exists(my_path + '/cython/test_cython_integration.c'):
        raise RuntimeError("""Cython library presence but cythonized c extension absence
indicate that either test is run on clean source where any setup.py commands were not called,
or setup.py deprecated automatic cythonization.
Try to run 'python setup.py recythonize', then rerun the test""")


def test_extensions_built():
    try:
        import hyperspy.tests.misc.cython.test_cython_integration as test_cy
        nt.assert_equal(7, test_cy.testing_cython())
    except ImportError:
        raise RuntimeError("""Compiled code is missing in source directory:
1. Check out that testing environment compiles code in the source directory
  by calling 'python setup.py build_ext --inplace', before running the test.
2. Check out you have compiling environment (gcc/clang/msvc..)
  same as used python version on your OS.
3. Check out that cythonization of code works. I.e. try running
  'python setup.py recythonize' """)