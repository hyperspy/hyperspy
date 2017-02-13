# Configure mpl and traits to work in a headless system
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = "null"
import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = np
    doctest_namespace['plt'] = plt
    doctest_namespace['hs'] = hs

def setup_module(mod):
    if pytest.config.getoption("--pdb"):
        import dask
        dask.set_options(get=dask.async.get_sync)
