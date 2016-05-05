__author__ = 'to266'

import numpy as np

from hyperspy._samfire_utils.strategy import DiffusionStrategy
from hyperspy._samfire_utils._weights.red_chisq import ReducedChiSquaredWeight


def exp_decay(distances):
    return np.exp(-distances)


class ReducedChiSquaredStrategy(DiffusionStrategy):

    def __init__(self):
        DiffusionStrategy.__init__(self, 'Reduced chi squared strategy')
        self.weight = ReducedChiSquaredWeight()
        self.radii = 3.
        self.decay_function = exp_decay
