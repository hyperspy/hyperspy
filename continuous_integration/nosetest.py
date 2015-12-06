#!/usr/bin/python
"""Run nosetests after setting ETS toolkit to "null"."""

if __name__ == '__main__':
    import sys
    import os
    from nose import run_exit
    from traits.etsconfig.api import ETSConfig

    ETSConfig.toolkit = "null"
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'error,ignore::ImportWarning'
    sys.exit(run_exit(env=env))
