
import numpy as np
from nose.tools import assert_almost_equal

import hyperspy.hspy as hs
from hyperspy.misc.elements import elements_db


class TestWeightToFromAtomic():

    def setUp(self):
        # TiO2
        self.elements = (("Ti", "O"))
        natoms = (1, 2)
        self.at = [100 * nat / float(sum(natoms)) for nat in natoms]
        atomic_weight = np.array(
            [elements_db[element].General_properties.atomic_weight for element in self.elements])
        mol_weight = atomic_weight * natoms
        self.wt = [100 * w / mol_weight.sum() for w in mol_weight]

    def test_weight_to_atomic(self):
        cwt = hs.utils.material.weight_to_atomic(self.elements, self.wt)
        assert_almost_equal(cwt[0], self.at[0])
        assert_almost_equal(cwt[1], self.at[1])

    def test_atomic_to_weight(self):
        cat = hs.utils.material.atomic_to_weight(self.elements, self.at)
        assert_almost_equal(cat[0], self.wt[0])
        assert_almost_equal(cat[1], self.wt[1])


def test_density_of_mixture():
    # Bronze
    elements = ("Cu", "Sn")
    wt = (88., 12.)
    densities = np.array(
        [elements_db[element].Physical_properties.density_gcm3 for element in elements])

    volumes = wt / densities
    density = 100. / volumes.sum()
    assert_almost_equal(density,
                        hs.utils.material.density_of_mixture_of_pure_elements(
                            elements, wt))
