
import numpy as np
from nose.tools import assert_almost_equal
import nose.tools

import hyperspy.api as hs
from hyperspy.misc.elements import elements_db


class TestWeightToFromAtomic:

    def setUp(self):
        # TiO2
        self.elements = ("Ti", "O")
        natoms = (1, 2)
        self.at = [100 * nat / float(sum(natoms)) for nat in natoms]
        atomic_weight = np.array(
            [elements_db[element].General_properties.atomic_weight for element
                in self.elements])
        mol_weight = atomic_weight * natoms
        self.wt = [100 * w / mol_weight.sum() for w in mol_weight]

    def test_weight_to_atomic(self):
        cwt = hs.material.weight_to_atomic(self.wt, self.elements)
        assert_almost_equal(cwt[0], self.at[0])
        assert_almost_equal(cwt[1], self.at[1])

    def test_atomic_to_weight(self):
        cat = hs.material.atomic_to_weight(self.at, self.elements)
        assert_almost_equal(cat[0], self.wt[0])
        assert_almost_equal(cat[1], self.wt[1])

    def test_multi_dim(self):
        elements = ("Cu", "Sn")
        wt = np.array([[[88] * 2] * 3, [[12] * 2] * 3])
        at = hs.material.weight_to_atomic(wt, elements)
        nose.tools.assert_true(np.allclose(
            at[:, 0, 0], np.array([93.196986, 6.803013]), atol=1e-3))
        wt2 = hs.material.atomic_to_weight(at, elements)
        nose.tools.assert_true(np.allclose(wt, wt2))


def test_density_of_mixture():
    # Bronze
    elements = ("Cu", "Sn")
    wt = (88., 12.)
    densities = np.array(
        [elements_db[element].Physical_properties.density_gcm3 for element in
            elements])

    volumes = wt * densities
    density = volumes.sum() / 100.
    assert_almost_equal(
        density, hs.material.density_of_mixture_of_pure_elements(
            wt, elements, mean='weighted'))

    volumes = wt / densities
    density = 100. / volumes.sum()
    assert_almost_equal(
        density, hs.material.density_of_mixture_of_pure_elements(
            wt, elements))

    wt = np.array([[[88] * 2] * 3, [[12] * 2] * 3])
    assert_almost_equal(
        density, hs.material.density_of_mixture_of_pure_elements(
            wt, elements)[0, 0])
