
import numpy as np
from nose.tools import assert_almost_equal

from hyperspy.hspy import *
from hyperspy.misc.eds.elements import elements as elements_db

class TestWeightToFromAtomic():
    def setUp(self):
        # TiO2
        self.elements = (("Ti", "O"))
        natoms = (1, 2)
        self.at = [100 * nat / float(sum(natoms)) for nat in natoms]
        atomic_weight = np.array(
            [elements_db[element]['A'] for element in self.elements])
        mol_weight = atomic_weight * natoms
        self.wt = [100 * w / mol_weight.sum() for w in mol_weight]

    def test_weight_to_atomic(self):
        cwt = utils.material.weight_to_atomic(self.elements, self.wt)
        assert_almost_equal(cwt[0], self.at[0])
        assert_almost_equal(cwt[1], self.at[1])

    def test_atomic_to_weight(self):
        cat = utils.material.atomic_to_weight(self.elements, self.at)
        assert_almost_equal(cat[0], self.wt[0])
        assert_almost_equal(cat[1], self.wt[1])
