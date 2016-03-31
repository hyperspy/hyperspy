#!/usr/bin/python
"""Run nosetests after setting ETS toolkit to "null"."""

if __name__ == '__main__':
    import sys
    from nose import run_exit
    from traits.etsconfig.api import ETSConfig
    import os
    import warnings
    ETSConfig.toolkit = "null"
    import matplotlib
    matplotlib.use("Agg")

    # Check if we should fail on external deprecation messages
    fail_on_external = os.environ.pop('FAIL_ON_EXTERNAL_DEPRECATION', False)
    if isinstance(fail_on_external, str):
        fail_on_external = (fail_on_external.lower() in
                            ['true', 't', '1', 'yes', 'y', 'set'])

    if fail_on_external:
        warnings.filterwarnings(
            'error', category=DeprecationWarning)
        # Travis setup has these warnings, so ignore:
        warnings.filterwarnings(
            'ignore',
            "BaseException\.message has been deprecated as of Python 2\.6",
            DeprecationWarning)
        # Don't care about warnings in hyperspy in this mode!
        warnings.filterwarnings('default', module="hyperspy")
    else:
        # Fall-back filter: Error
        warnings.simplefilter('error')
        warnings.filterwarnings(
            'ignore', "Failed to import the optional scikit image package",
            UserWarning)
        # We allow extrernal warnings:
        warnings.filterwarnings('default',
                                module="(?!hyperspy)")

    sys.exit(run_exit())
