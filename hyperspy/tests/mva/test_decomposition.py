import numpy as np

from nose.tools import assert_true, raises
from hyperspy import signals


class TestNdAxes:

    def setUp(self):
        # Create three signals with dimensions:
        # s1 : <Signal, title: , dimensions: (4, 3, 2|2, 3)>
        # s2 : <Signal, title: , dimensions: (2, 3|4, 3, 2)>
        # s12 : <Signal, title: , dimensions: (2, 3|4, 3, 2)>
        # Where s12 data is transposed in respect to s2
        dc1 = np.random.random((2, 3, 4, 3, 2))
        dc2 = np.rollaxis(np.rollaxis(dc1, -1), -1)
        s1 = signals.Signal(dc1.copy())
        s2 = signals.Signal(dc2)
        s12 = signals.Signal(dc1.copy())
        for i, axis in enumerate(s1.axes_manager._axes):
            if i < 3:
                axis.navigate = True
            else:
                axis.navigate = False
        for i, axis in enumerate(s2.axes_manager._axes):
            if i < 2:
                axis.navigate = True
            else:
                axis.navigate = False
        for i, axis in enumerate(s12.axes_manager._axes):
            if i < 3:
                axis.navigate = False
            else:
                axis.navigate = True
        self.s1 = s1
        self.s2 = s2
        self.s12 = s12

    def test_consistensy(self):
        s1 = self.s1
        s2 = self.s2
        s12 = self.s12
        s1.decomposition()
        s2.decomposition()
        s12.decomposition()
        assert_true((s2.learning_results.loadings ==
                     s12.learning_results.loadings).all())
        assert_true((s2.learning_results.factors ==
                     s12.learning_results.factors).all())
        assert_true((s1.learning_results.loadings ==
                     s2.learning_results.factors).all())
        assert_true((s1.learning_results.factors ==
                     s2.learning_results.loadings).all())

    def test_consistensy_poissonian(self):
        s1 = self.s1
        s2 = self.s2
        s12 = self.s12
        s1.decomposition(normalize_poissonian_noise=True)
        s2.decomposition(normalize_poissonian_noise=True)
        s12.decomposition(normalize_poissonian_noise=True)
        assert_true((s2.learning_results.loadings ==
                     s12.learning_results.loadings).all())
        assert_true((s2.learning_results.factors ==
                     s12.learning_results.factors).all())
        assert_true((s1.learning_results.loadings ==
                     s2.learning_results.factors).all())
        assert_true((s1.learning_results.factors ==
                     s2.learning_results.loadings).all())


class TestGetExplainedVarinaceRation():

    def setUp(self):
        s = signals.Signal(np.empty(1))
        s.learning_results.explained_variance_ratio = np.empty(10)
        self.s = s

    def test_data(self):
        assert_true((self.s.get_explained_variance_ratio().data ==
                     self.s.learning_results.explained_variance_ratio).all())

    @raises(AttributeError)
    def test_no_evr(self):
        self.s.get_explained_variance_ration()


class TestReverseDecompositionComponent():

    def setUp(self):
        s = signals.Signal(np.random.random((10, 10, 100)))
        self.s = s
        self.s_rev = s.deepcopy()

    def test_reversal_factors(self):
        # Reverse one component:
        self.s.decomposition(output_dimension=5)
        self.s_rev.decomposition(output_dimension=5)
        self.s_rev.reverse_decomposition_component(0)
        assert_true((self.s.learning_results.factors[:, 0] ==
                     self.s_rev.learning_results.factors[:, 0] * -1).all())

        # Reverse two components:
        self.s.decomposition(output_dimension=5)
        self.s_rev.decomposition(output_dimension=5)
        self.s_rev.reverse_decomposition_component([0,1])
        assert_true((self.s.learning_results.factors[:, 0] ==
                     self.s_rev.learning_results.factors[:, 0] * -1).all())
        assert_true((self.s.learning_results.factors[:, 1] ==
                     self.s_rev.learning_results.factors[:, 1] * -1).all())

        # Reverse all components:
        self.s.decomposition(output_dimension=5)
        self.s_rev.decomposition(output_dimension=5)
        self.s_rev.reverse_decomposition_component([0, 1, 2, 3, 4])
        assert_true((self.s.learning_results.factors[:, 0] ==
                     self.s_rev.learning_results.factors[:, 0] * -1).all())
        assert_true((self.s.learning_results.factors[:, 1] ==
                     self.s_rev.learning_results.factors[:, 1] * -1).all())
        assert_true((self.s.learning_results.factors[:, 2] ==
                     self.s_rev.learning_results.factors[:, 2] * -1).all())
        assert_true((self.s.learning_results.factors[:, 3] ==
                     self.s_rev.learning_results.factors[:, 3] * -1).all())
        assert_true((self.s.learning_results.factors[:, 4] ==
                     self.s_rev.learning_results.factors[:, 4] * -1).all())

    def test_reversal_loadings(self):
        # Reverse one component:
        self.s.decomposition(output_dimension=5)
        self.s_rev.decomposition(output_dimension=5)
        self.s_rev.reverse_decomposition_component(0)
        assert_true((self.s.learning_results.loadings[:, 0] ==
                     self.s_rev.learning_results.loadings[:, 0] * -1).all())

        # Reverse two components:
        self.s.decomposition(output_dimension=5)
        self.s_rev.decomposition(output_dimension=5)
        self.s_rev.reverse_decomposition_component([0,1])
        assert_true((self.s.learning_results.loadings[:, 0] ==
                     self.s_rev.learning_results.loadings[:, 0] * -1).all())
        assert_true((self.s.learning_results.loadings[:, 1] ==
                     self.s_rev.learning_results.loadings[:, 1] * -1).all())

        # Reverse all components:
        self.s.decomposition(output_dimension=5)
        self.s_rev.decomposition(output_dimension=5)
        self.s_rev.reverse_decomposition_component([0, 1, 2, 3, 4])
        assert_true((self.s.learning_results.loadings[:, 0] ==
                     self.s_rev.learning_results.loadings[:, 0] * -1).all())
        assert_true((self.s.learning_results.loadings[:, 1] ==
                     self.s_rev.learning_results.loadings[:, 1] * -1).all())
        assert_true((self.s.learning_results.loadings[:, 2] ==
                     self.s_rev.learning_results.loadings[:, 2] * -1).all())
        assert_true((self.s.learning_results.loadings[:, 3] ==
                     self.s_rev.learning_results.loadings[:, 3] * -1).all())
        assert_true((self.s.learning_results.loadings[:, 4] ==
                     self.s_rev.learning_results.loadings[:, 4] * -1).all())