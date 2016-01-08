#!/usr/bin/python
"""Run nosetests after setting ETS toolkit to "null"."""

if __name__ == '__main__':
    import sys
    from nose import run_exit
    from traits.etsconfig.api import ETSConfig

    ETSConfig.toolkit = "null"
    import matplotlib
    matplotlib.use("Agg")
    sys.exit(run_exit())
