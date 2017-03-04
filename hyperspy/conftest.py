# Configure mpl and traits to work in a headless system
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = "null"
import matplotlib.pyplot as plt
plt.rcParams.clear()
plt.rcParams.update(plt.rcParamsDefault)

import pytest
import numpy as np
import matplotlib
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

@pytest.fixture
def mpl_cleanup():
    from matplotlib.testing.decorators import _do_cleanup

    original_units_registry = matplotlib.units.registry.copy()
    original_settings = matplotlib.rcParams.copy()
    _do_cleanup(original_units_registry, original_settings)