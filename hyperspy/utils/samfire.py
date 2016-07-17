"""SAMFire modules


The :mod:`~hyperspy.api.samfire` module contains the following submodules:

fit_tests
    Tests to check fit convergence when running SAMFire

global_strategies
    Available global strategies to use in SAMFire

local_strategies
    Available global strategies to use in SAMFire

SamfirePool
    The parallel pool, customized to run SAMFire.

"""
from hyperspy.samfire_utils import fit_tests
from hyperspy.samfire_utils import global_strategies
from hyperspy.samfire_utils import local_strategies
