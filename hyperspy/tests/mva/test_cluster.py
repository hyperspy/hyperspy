import numpy as np

from hyperspy import signals


class TestCluster1d:

    def setUp(self):
        # Use prime numbers to avoid fluke equivalences
        self.signal = signals.Signal1D(np.random.rand(11, 5, 7))
        self.signal.decomposition(output_dimension=2)
        self.navigation_mask = np.zeros((11, 5)).astype(bool)
        self.navigation_mask[4:6, 1:4] = True
        self.signal_mask = np.zeros((7,)).astype(bool)
        self.signal_mask[2:6] = True

    def test_shapes(self):
        self.signal.cluster(3, use_decomposition_results=False)
        np.testing.assert_array_equal(
            self.signal.learning_results.memberships.shape, (55, 3))
        np.testing.assert_array_equal(
            self.signal.learning_results.centers.shape, (3, 7))

    def test_shapes_decomposed(self):
        self.signal.cluster(3, reproject=False)
        np.testing.assert_array_equal(
            self.signal.learning_results.memberships.shape, (55, 3))
        np.testing.assert_array_equal(
            self.signal.learning_results.centers.shape, (3, 2))

    def test_shapes_decomposed_reprojected(self):
        self.signal.cluster(3)
        np.testing.assert_array_equal(
            self.signal.learning_results.memberships.shape, (55, 3))
        np.testing.assert_array_equal(
            self.signal.learning_results.centers.shape, (3, 7))

    def test_shapes_masked(self):
        self.signal.cluster(3, use_decomposition_results=False,
                            navigation_mask=self.navigation_mask,
                            signal_mask=self.signal_mask)
        np.testing.assert_array_equal(
            self.signal.learning_results.memberships.shape, (55, 3))
        np.testing.assert_array_equal(
            self.signal.learning_results.centers.shape, (3, 7))


class TestCluster2d:

    def setUp(self):
        self.signal = signals.Signal2D(np.random.rand(11, 5, 7))
        self.signal.decomposition(output_dimension=2)
        self.navigation_mask = np.zeros((11,)).astype(bool)
        self.navigation_mask[4:6] = True
        self.signal_mask = np.zeros((5, 7)).astype(bool)
        self.signal_mask[1:4, 2:6] = True

    def test_shapes(self):
        self.signal.cluster(3, use_decomposition_results=False)
        np.testing.assert_array_equal(
            self.signal.learning_results.memberships.shape,
            (11, 3))
        np.testing.assert_array_equal(
            self.signal.learning_results.centers.shape,
            (3, 35))

    def test_shapes_decomposed(self):
        self.signal.cluster(3, reproject=False)
        np.testing.assert_array_equal(
            self.signal.learning_results.memberships.shape, (11, 3))
        np.testing.assert_array_equal(
            self.signal.learning_results.centers.shape, (3, 2))

    def test_shapes_decomposed_reprojected(self):
        self.signal.cluster(3)
        np.testing.assert_array_equal(
            self.signal.learning_results.memberships.shape, (11, 3))
        np.testing.assert_array_equal(
            self.signal.learning_results.centers.shape, (3, 35))

    def test_shapes_masked(self):
        self.signal.cluster(3, use_decomposition_results=False,
                            navigation_mask=self.navigation_mask,
                            signal_mask=self.signal_mask)
        np.testing.assert_array_equal(
            self.signal.learning_results.memberships.shape, (11, 3))
        np.testing.assert_array_equal(
            self.signal.learning_results.centers.shape, (3, 35))
