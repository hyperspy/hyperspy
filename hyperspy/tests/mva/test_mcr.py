import numpy as np

from hyperspy._signals.signal1d import Signal1D


def are_mcr_components_equivalent(c1_list, c2_list, atol=1e-4):
    """Check if two list of components are equivalent.

    To be equivalent they must differ by a max of `atol` except
    for an arbitraty -1 factor.

    Parameters
    ----------
    c1_list, c2_list: list of Signal instances.
        The components to check.
    atol: float
        Absolute tolerance for the amount that they can differ.

    Returns
    -------
    bool

    """
    matches = 0
    for c1 in c1_list:
        for c2 in c2_list:
            if (np.allclose(c2.data, c1.data, atol=atol) or
                    np.allclose(c2.data, -c1.data, atol=atol)):
                matches += 1
    return matches == len(c1_list)


class TestMCR1D:

    def setup_method(self, method):
        orig_factors = np.random.random(size=(3, 1000))
        np.random.seed(1)
        orig_loadings = np.random.random((100, 3))
        self.s = Signal1D(np.dot(orig_loadings, orig_factors))
        self.s.decomposition()

    def test_spatial_consistency(self):
        self.s.decomposition(algorithm='MCR', output_dimension=3,
                             simplicity='spatial')
        s2 = self.s.as_signal1D(1)
        s2.decomposition()
        s2.decomposition(algorithm='MCR', output_dimension=3,
                         simplicity='spatial')
        assert are_mcr_components_equivalent(
            self.s.get_decomposition_factors(), s2.get_decomposition_factors())

    def test_spectral_consistency(self):
        self.s.decomposition(algorithm='MCR', output_dimension=3,
                             simplicity='spectral')
        s2 = self.s.as_signal1D(1)
        s2.decomposition()
        s2.decomposition(algorithm='MCR', output_dimension=3,
                         simplicity='spectral')
        assert are_mcr_components_equivalent(
            self.s.get_decomposition_factors(), s2.get_decomposition_factors())

    def test_spatial_dimensions(self):
        self.s.decomposition(algorithm='MCR', output_dimension=3,
                             simplicity='spatial')
        factors = self.s.get_decomposition_factors()

        assert (factors.data.shape[0] == 3)
        assert(factors.data.shape[1] == self.s.data.shape[1])

    def test_spectral_dimensions(self):
        self.s.decomposition(algorithm='MCR', output_dimension=3,
                             simplicity='spectral')
        factors = self.s.get_decomposition_factors()

        assert (factors.data.shape[0] == 3)
        assert(factors.data.shape[1] == self.s.data.shape[1])


class TestMCR2D:

    def setup_method(self, method):
        orig_factors = np.random.random(size=(3, 1000))
        np.random.seed(1)
        orig_loadings = np.random.random((100, 3))
        self.s = Signal1D(np.dot(orig_loadings,
                          orig_factors).reshape([10, 10, 1000]))
        self.s.decomposition()

    def test_spatial_consistency(self):
        self.s.decomposition(algorithm='MCR', output_dimension=3,
                             simplicity='spatial')
        s2 = self.s.as_signal1D(2)
        s2.decomposition()
        s2.decomposition(algorithm='MCR', output_dimension=3,
                         simplicity='spatial')
        assert are_mcr_components_equivalent(
            self.s.get_decomposition_factors(), s2.get_decomposition_factors())

    def test_spectral_consistency(self):
        self.s.decomposition(algorithm='MCR', output_dimension=3,
                             simplicity='spectral')
        s2 = self.s.as_signal1D(2)
        s2.decomposition()
        s2.decomposition(algorithm='MCR', output_dimension=3,
                         simplicity='spectral')
        assert are_mcr_components_equivalent(
            self.s.get_decomposition_factors(), s2.get_decomposition_factors())

    def test_spatial_dimensions(self):
        self.s.decomposition(algorithm='MCR', output_dimension=3,
                             simplicity='spatial')
        factors = self.s.get_decomposition_factors()

        assert (factors.data.shape[0] == 3)
        assert(factors.data.shape[1] == self.s.data.shape[2])

    def test_spectral_dimensions(self):
        self.s.decomposition(algorithm='MCR', output_dimension=3,
                             simplicity='spectral')
        factors = self.s.get_decomposition_factors()

        assert (factors.data.shape[0] == 3)
        assert(factors.data.shape[1] == self.s.data.shape[2])
