import numpy as np

from hyperspy.component import Component

class Gaussian2D(Component):

    def __init__(self,
                 amplitude=1.,
                 sigma_x=1.,
                 sigma_y=1.,
                 centre_x=0.,
                 centre_y=0,
                ):
        Component.__init__(self, ['amplitude',
                                  'sigma_x',
                                  'sigma_y',
                                  'centre_x',
                                  'centre_y',
                                 ])
        self.amplitude.value = amplitude
        self.sigma_x.value = sigma_x
        self.sigma_y.value = sigma_y
        self.centre_x.value = centre_x
        self.centre_y.value = centre_y

    def function(self, x, y):
        amp = self.amplitude.value
        sx = self.sigma_x.value
        sy = self.sigma_y.value
        x0 = self.centre_x.value
        y0 = self.centre_y.value

        return amp *  np.exp(
            -((x - x0) ** 2 / (2 * sx ** 2) +
              (y - y0) ** 2 / (2 * sy ** 2)))
