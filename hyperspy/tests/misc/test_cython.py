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

