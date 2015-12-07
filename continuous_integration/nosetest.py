#!/usr/bin/python
"""Run nosetests after setting ETS toolkit to "null"."""

if __name__ == '__main__':
    import sys
    from nose import run_exit
    from traits.etsconfig.api import ETSConfig
    import os
    import warnings
    ETSConfig.toolkit = "null"

    # Check if we should fail on external deprecation messages
    fail_on_external = os.environ.pop('FAIL_ON_EXTERNAL_DEPRECATION', False)
    if isinstance(fail_on_external, basestring):
        fail_on_external = (fail_on_external.lower() in
            ['true', 't', '1', 'yes', 'y', 'set'])

    # Fall-back filter: Error
    warnings.simplefilter('error')
    warnings.filterwarnings(
        'ignore',"Failed to import the optional scikit image package", 
        UserWarning)

    if fail_on_external:
        # Travis setup has this:
        warnings.filterwarnings(
            'ignore',
            "BaseException\.message has been deprecated as of Python 2\.6",
            DeprecationWarning)
        # Don't care about warnings in hyperspy
        warnings.filterwarnings('default', module="hyperspy")
    else:
        # Default behavior for non-hyperspy deprecations
        warnings.filterwarnings('default', category=DeprecationWarning,
                                module="(?!hyperspy)")

    sys.exit(run_exit())
