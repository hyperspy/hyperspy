# Configure mpl and traits to work in a headless system
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = "null"

# pytest-mpl 0.7 already import pyplot, so setting the matplotlib backend to
# 'agg' as early as we can is useless for testing.
import matplotlib.pyplot as plt

import pytest
import numpy as np
import matplotlib
import hyperspy.api as hs


matplotlib.rcParams['figure.max_open_warning'] = 25
matplotlib.rcParams['interactive'] = False
hs.preferences.Plot.saturated_pixels = 0.0


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = np
    doctest_namespace['plt'] = plt
    doctest_namespace['hs'] = hs

@pytest.fixture
def pdb_cmdopt(request):
    return request.config.getoption("--pdb")

def setup_module(mod, pdb_cmdopt):
    if pdb_cmdopt:
        import dask
        dask.set_options(get=dask.local.get_sync)

from matplotlib.testing.conftest import mpl_test_settings
