"""Import sklearn if installed API changes in 0.12:
The old scikits.learn package has disappeared;
all code should import from sklearn instead, which was introduced in 0.9.

"""

import warnings


try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.utils.extmath import randomized_svd as fast_svd
        import sklearn.decomposition
        import sklearn.cluster
        import sklearn.preprocessing        
        import sklearn.metrics
        from sklearn.decomposition import FastICA
        sklearn_installed = True
except ImportError:
    fast_svd = None
    sklearn_installed = False
