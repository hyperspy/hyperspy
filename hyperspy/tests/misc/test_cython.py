import nose.tools as nt
import warnings

import os

my_path = os.path.dirname(__file__)

def test_cythonization():
    try:
        import Cython
    except ImportError:
        warnings.warn("""no cython library were found, aborting testing of cythonization""")
    else:
        if not os.path.exists(my_path + '/cython/test_cython_integration.c'):
            raise DeprecationWarning('deprecation of cythonization of pyx extensions detected!')

def test_extensions_built():
    try:
        import hyperspy.tests.misc.cython.test_cython_integration as test_cy
        nt.assert_equal(7, test_cy.testing_cython())
    except ImportError:
        raise DeprecationWarning('deprecation of building extensions detected!')