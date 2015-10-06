__author__ = 'to266'

import numpy as np

from hyperspy._samfire_utils.strategy import diffusion_strategy
from hyperspy._samfire_utils._weights.red_chisq import Reduced_chi_squared_weight


def exp_decay(distances):
    return np.exp(-distances)


class reduced_chi_squared_strategy(diffusion_strategy):

    def __init__(self):
        diffusion_strategy.__init__(self, 'Reduced chi squared strategy')
        self.weight = Reduced_chi_squared_weight()
        self.radii = 3.
        self.decay_function = exp_decay
