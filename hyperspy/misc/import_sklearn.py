"""Import sklearn, fast_svd and randomized_svd from scikits-learn
with support for multiple versions

"""

import warnings
from distutils.version import StrictVersion

try:
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore")
        try:
            import sklearn
        except:
            import scikits.learn as sklearn
        sklearn_version = StrictVersion(sklearn.__version__)
        if  sklearn_version < StrictVersion("0.9"):
            import scikits.learn.decomposition
            from scikits.learn.utils.extmath import fast_svd
        elif sklearn_version == StrictVersion("0.9"):
            from sklearn.utils.extmath import fast_svd
            import sklearn.decomposition
        else:
            from sklearn.utils.extmath import randomized_svd as fast_svd
            import sklearn.decomposition
        from sklearn.decomposition import FastICA
        sklearn_installed = True
except ImportError:
    sklearn_installed = False
    from hyperspy.misc.fastica import FastICA
