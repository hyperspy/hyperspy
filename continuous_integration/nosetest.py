#!/usr/bin/python
"""Run nosetests after setting ETS toolkit to "null"."""

if __name__ == '__main__':
    import sys
    from nose import run_exit
    import warnings
    from traits.etsconfig.api import ETSConfig

    ETSConfig.toolkit = "null"
    warnings.simplefilter('error')
    warnings.filterwarnings(
        'ignore', "Failed to import the optional scikit image package",
        UserWarning)
    warnings.filterwarnings(
        'ignore',
        "BaseException\.message has been deprecated as of Python 2\.6",
        DeprecationWarning)
    sys.exit(run_exit())
