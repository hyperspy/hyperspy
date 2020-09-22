"""Import sklearn, fast_svd and randomized_svd from scikits-learn
with support for multiple versions

"""

import warnings
from distutils.version import LooseVersion

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import sklearn
        except:
            import scikits.learn as sklearn
        sklearn_version = LooseVersion(sklearn.__version__)
        if sklearn_version < LooseVersion("0.9"):
            import scikits.learn.decomposition
            from scikits.learn.utils.extmath import fast_svd
        elif sklearn_version == LooseVersion("0.9"):
            from sklearn.utils.extmath import fast_svd
            import sklearn.decomposition
        else:
            from sklearn.utils.extmath import randomized_svd as fast_svd
            import sklearn.decomposition
        from sklearn.decomposition import FastICA
        sklearn_installed = True
except ImportError:
    fast_svd = None
    sklearn_installed = False
