
import numpy as np

from numpy.testing import assert_allclose

import hyperspy.api as hs
from hyperspy.misc.elements import elements_db


class TestWeightToFromAtomic:

    def setup_method(self, method):
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
        assert_allclose(cwt[0], self.at[0])
        assert_allclose(cwt[1], self.at[1])

    def test_atomic_to_weight(self):
        cat = hs.material.atomic_to_weight(self.at, self.elements)
        assert_allclose(cat[0], self.wt[0])
        assert_allclose(cat[1], self.wt[1])

    def test_multi_dim(self):
        elements = ("Cu", "Sn")
        wt = np.array([[[88] * 2] * 3, [[12] * 2] * 3])
        at = hs.material.weight_to_atomic(wt, elements)
        assert np.allclose(
            at[:, 0, 0], np.array([93.196986, 6.803013]), atol=1e-3)
        wt2 = hs.material.atomic_to_weight(at, elements)
        assert np.allclose(wt, wt2)


def test_density_of_mixture():
    # Bronze
    elements = ("Cu", "Sn")
    wt = (88., 12.)
    densities = np.array(
        [elements_db[element].Physical_properties.density_gcm3 for element in
            elements])

    volumes = wt * densities
    density = volumes.sum() / 100.
    assert_allclose(
        density, hs.material.density_of_mixture(wt, elements, mean='weighted'))

    volumes = wt / densities
    density = 100. / volumes.sum()
    assert_allclose(
        density, hs.material.density_of_mixture(wt, elements))

    wt = np.array([[[88] * 2] * 3, [[12] * 2] * 3])
    assert_allclose(
        density, hs.material.density_of_mixture(wt, elements)[0, 0])


def test_mac():
    assert_allclose(
        hs.material.mass_absorption_coefficient('Al', 3.5), 506.0153356472)
    assert np.allclose(
        hs.material.mass_absorption_coefficient('Ta', [1, 3.2, 2.3]),
        [3343.7083701143229, 1540.0819991890, 3011.264941118])
    assert_allclose(
        hs.material.mass_absorption_coefficient('Zn', 'Zn_La'),
        1413.291119134)
    assert np.allclose(
        hs.material.mass_absorption_coefficient(
            'Zn', ['Cu_La', 'Nb_La']), [1704.7912903000029,
                                        1881.2081950943339])


def test_mixture_mac():
    assert_allclose(hs.material.mass_absorption_mixture([50, 50],
                                                        ['Al', 'Zn'],
                                                        'Al_Ka'),
                    2587.4161643905127)
    elements = ("Cu", "Sn")
    lines = [0.5, 'Al_Ka']
    wt = np.array([[[88.] * 2] * 3, [[12.] * 2] * 3])
    np.testing.assert_array_almost_equal(
        hs.material.mass_absorption_mixture(wt, elements, lines)[:, 0, 0],
        np.array([8003.05391481, 4213.4235561]))
    wt = hs.signals.Signal2D(wt).split()
    mac = hs.material.mass_absorption_mixture(wt, elements, lines)
    np.testing.assert_array_almost_equal(mac[0].data[0, 0], 8003.053914)
