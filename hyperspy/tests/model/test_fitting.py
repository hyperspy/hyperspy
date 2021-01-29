# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import logging

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass
from hyperspy.exceptions import VisibleDeprecationWarning

TOL = 1e-5


def _create_toy_1d_gaussian_model(binned=True, weights=False, noise=False):
    """Toy dataset for 1D fitting

    Parameters
    ----------
    binned : bool, default True
        Is the signal binned?
    weights : bool, default False
        If True, set an arbitrary noise variance for weighted fitting
    noise : bool, default False
        If True, add Poisson noise to the signal

    Returns
    -------
    m
        Model1D for fitting

    """
    np.random.seed(1)
    v = 2.0 * np.exp(-((np.arange(10, 100, 0.1) - 50) ** 2) / (2 * 5.0 ** 2))
    s = hs.signals.Signal1D(v)
    s.axes_manager[0].scale = 0.1
    s.axes_manager[0].offset = 10
    s.metadata.Signal.binned = binned

    if weights:
        s_var = hs.signals.Signal1D(np.arange(10, 100, 0.01))
        s.set_noise_variance(s_var)

    if noise:
        s.add_poissonian_noise()

    g = hs.model.components1D.Gaussian()
    g.centre.value = 56.0
    g.A.value = 250.0
    g.sigma.value = 4.9
    m = s.create_model()
    m.append(g)

    return m


@lazifyTestClass
class TestModelFitBinnedLeastSquares:
    def setup_method(self, method):
        self.m = _create_toy_1d_gaussian_model()

    def _check_model_values(self, model, expected, **kwargs):
        np.testing.assert_allclose(model.A.value, expected[0], **kwargs)
        np.testing.assert_allclose(model.centre.value, expected[1], **kwargs)
        np.testing.assert_allclose(model.sigma.value, expected[2], **kwargs)

    @pytest.mark.parametrize("grad", ["fd", "analytical"])
    @pytest.mark.parametrize(
        "bounded, expected",
        [(False, (250.66282746, 50.0, 5.0)), (True, (257.48162397, 55.0, 7.76886132))],
    )
    def test_fit_lm(self, grad, bounded, expected):
        if bounded:
            self.m[0].centre.bmin = 55.0

        self.m.fit(optimizer="lm", bounded=bounded, grad=grad)
        self._check_model_values(self.m[0], expected, rtol=TOL)

        assert isinstance(self.m.fit_output, OptimizeResult)
        assert self.m.p_std is not None
        assert len(self.m.p_std) == 3
        assert np.all(~np.isnan(self.m.p_std))

    @pytest.mark.parametrize(
        "grad, expected",
        [("fd", (250.66282746, 50.0, 5.0)), ("analytical", (250.66282746, 50.0, 5.0))],
    )
    def test_fit_trf(self, grad, expected):
        self.m.fit(optimizer="trf", grad=grad)
        self._check_model_values(self.m[0], expected, rtol=TOL)

        assert isinstance(self.m.fit_output, OptimizeResult)
        assert self.m.p_std is not None
        assert len(self.m.p_std) == 3
        assert np.all(~np.isnan(self.m.p_std))

    @pytest.mark.parametrize(
        "grad, expected",
        [
            (None, (250.66282746, 50.0, 5.0)),
            ("fd", (250.66282746, 50.0, 5.0)),
            ("analytical", (250.66282746, 50.0, 5.0)),
        ],
    )
    def test_fit_odr(self, grad, expected):
        self.m.fit(optimizer="odr", grad=grad)
        self._check_model_values(self.m[0], expected, rtol=TOL)

        assert isinstance(self.m.fit_output, OptimizeResult)
        assert self.m.p_std is not None
        assert len(self.m.p_std) == 3
        assert np.all(~np.isnan(self.m.p_std))

    def test_fit_bounded_bad_starting_values(self):
        self.m[0].centre.bmin = 0.5
        self.m[0].centre.value = -1
        self.m.fit(optimizer="lm", bounded=True)
        expected = (0.0, 0.5, 4.90000050)
        self._check_model_values(self.m[0], expected, rtol=TOL)

    def test_fit_ext_bounding(self):
        self.m[0].A.bmin = 200.0
        self.m[0].A.bmax = 300.0
        self.m[0].centre.bmin = 51.0
        self.m[0].centre.bmax = 60.0
        self.m[0].sigma.bmin = 3.5
        self.m[0].sigma.bmax = 4.9

        with pytest.warns(
            VisibleDeprecationWarning, match="`ext_bounding=True` has been deprecated",
        ):
            self.m.fit(optimizer="lm", ext_bounding=True)

        expected = (200.0, 51.0, 4.9)
        self._check_model_values(self.m[0], expected, rtol=TOL)


class TestModelFitBinnedScipyMinimize:
    def setup_method(self, method):
        self.m = _create_toy_1d_gaussian_model()

    def _check_model_values(self, model, expected, **kwargs):
        np.testing.assert_allclose(model.A.value, expected[0], **kwargs)
        np.testing.assert_allclose(model.centre.value, expected[1], **kwargs)
        np.testing.assert_allclose(model.sigma.value, expected[2], **kwargs)

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    @pytest.mark.parametrize(
        "loss_function, expected",
        [
            ("ls", (250.66280759, 49.99999971, 5.00000122)),
            ("ML-poisson", (250.66282637, 49.99999927, 4.99999881)),
            ("huber", (250.66280759, 49.99999971, 5.00000122)),
        ],
    )
    def test_fit_scipy_minimize_gradient_free(self, loss_function, expected):
        self.m.fit(optimizer="Nelder-Mead", loss_function=loss_function)
        self._check_model_values(self.m[0], expected, rtol=TOL)
        assert isinstance(self.m.fit_output, OptimizeResult)

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    @pytest.mark.parametrize("grad", ["fd", "analytical"])
    @pytest.mark.parametrize(
        "loss_function, bounded, expected",
        [
            ("ls", False, (250.66284342, 50.00000045, 4.99999983)),
            ("ls", True, (257.48175956, 55.0, 7.76887330)),
            ("ML-poisson", True, (250.66296821, 55.0, 7.07106541)),
            ("huber", True, (257.48175678, 55.0, 7.76886929)),
        ],
    )
    def test_fit_scipy_minimize_gradients(self, grad, loss_function, bounded, expected):
        if bounded:
            self.m[0].centre.bmin = 55.0

        self.m.fit(
            optimizer="L-BFGS-B",
            loss_function=loss_function,
            grad=grad,
            bounded=bounded,
        )
        self._check_model_values(self.m[0], expected, rtol=TOL)
        assert isinstance(self.m.fit_output, OptimizeResult)

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    @pytest.mark.parametrize("grad", ["fd", "analytical"])
    @pytest.mark.parametrize(
        "delta, expected",
        [
            (None, (250.6628443, 49.9999987, 4.9999999)),
            (1.0, (250.6628443, 49.9999987, 4.9999999)),
            (10.0, (250.6628702, 50.0000011, 5.0000002)),
        ],
    )
    def test_fit_huber_delta(self, grad, delta, expected):
        self.m.fit(
            optimizer="L-BFGS-B", loss_function="huber", grad=grad, huber_delta=delta,
        )
        print(self.m.p0)
        self._check_model_values(self.m[0], expected, rtol=TOL)
        assert isinstance(self.m.fit_output, OptimizeResult)

    def test_constraints(self):
        # Primarily checks that constraints are passed correctly,
        # even though the end result is a bad fit
        cons = {"type": "ineq", "fun": lambda x: x[0] - x[1]}
        self.m.fit(optimizer="SLSQP", constraints=cons)
        expected = (250.69857440, 49.99996610, 5.00034370)
        self._check_model_values(self.m[0], expected, rtol=TOL)

    def test_fit_scipy_minimize_no_success(self, caplog):
        # Set bad starting values, no bounds,
        # max iteration of 1 to deliberately fail
        self.m[0].A.value = 0.0
        self.m[0].centre.value = -50.0
        self.m[0].sigma.value = 1000.0
        with caplog.at_level(logging.WARNING):
            self.m.fit(optimizer="Nelder-Mead", options={"maxiter": 1})

        expected = (0.00025, -50.0, 1000.0)
        self._check_model_values(self.m[0], expected, rtol=TOL)
        assert isinstance(self.m.fit_output, OptimizeResult)
        assert "Maximum number of iterations has been exceeded" in caplog.text
        assert not self.m.fit_output.success
        assert self.m.fit_output.nit == 1


class TestModelFitBinnedGlobal:
    def setup_method(self, method):
        self.m = _create_toy_1d_gaussian_model()

        # Add bounds for all free parameters
        # (needed for global optimization)
        self.m[0].A.bmin = 200.0
        self.m[0].A.bmax = 300.0
        self.m[0].centre.bmin = 40.0
        self.m[0].centre.bmax = 60.0
        self.m[0].sigma.bmin = 4.5
        self.m[0].sigma.bmax = 5.5

    def _check_model_values(self, model, expected, **kwargs):
        np.testing.assert_allclose(model.A.value, expected[0], **kwargs)
        np.testing.assert_allclose(model.centre.value, expected[1], **kwargs)
        np.testing.assert_allclose(model.sigma.value, expected[2], **kwargs)

    @pytest.mark.parametrize(
        "loss_function, expected",
        [
            ("ls", (250.66282746, 50.0, 5.0)),
            ("ML-poisson", (250.66337106, 50.00001755, 4.99997114)),
            ("huber", (250.66282746, 50.0, 5.0)),
        ],
    )
    def test_fit_differential_evolution(self, loss_function, expected):
        self.m.fit(
            optimizer="Differential Evolution",
            loss_function=loss_function,
            bounded=True,
            seed=1,
        )
        self._check_model_values(self.m[0], expected, rtol=TOL)
        assert isinstance(self.m.fit_output, OptimizeResult)

    def test_fit_dual_annealing(self):
        pytest.importorskip("scipy", minversion="1.2.0")
        self.m.fit(optimizer="Dual Annealing", loss_function="ls", bounded=True, seed=1)
        expected = (250.66282750, 50.0, 5.0)
        self._check_model_values(self.m[0], expected, rtol=TOL)
        assert isinstance(self.m.fit_output, OptimizeResult)

    def test_fit_shgo(self):
        pytest.importorskip("scipy", minversion="1.2.0")
        self.m.fit(optimizer="SHGO", loss_function="ls", bounded=True)
        expected = (250.66282750, 50.0, 5.0)
        self._check_model_values(self.m[0], expected, rtol=TOL)
        assert isinstance(self.m.fit_output, OptimizeResult)


@lazifyTestClass
class TestModelWeighted:
    def setup_method(self, method):
        self.m = _create_toy_1d_gaussian_model(binned=True, weights=True, noise=True)

    def _check_model_values(self, model, expected, **kwargs):
        np.testing.assert_allclose(model.A.value, expected[0], **kwargs)
        np.testing.assert_allclose(model.centre.value, expected[1], **kwargs)
        np.testing.assert_allclose(model.sigma.value, expected[2], **kwargs)

    @pytest.mark.parametrize("grad", ["fd", "analytical"])
    def test_chisq(self, grad):
        self.m.signal.metadata.Signal.binned = True
        self.m.fit(grad=grad)
        np.testing.assert_allclose(self.m.chisq.data, 18.81652763)

    @pytest.mark.parametrize("grad", ["fd", "analytical"])
    def test_red_chisq(self, grad):
        self.m.fit(grad=grad)
        np.testing.assert_allclose(self.m.red_chisq.data, 0.02100059)

    @pytest.mark.parametrize(
        "optimizer, binned, expected",
        [
            ("lm", True, (256.7752411, 49.9770694, 5.3008397)),
            ("odr", True, (256.7752604, 49.9770693, 5.3008397)),
            ("lm", False, (25.6775426, 49.9770509, 5.3008481)),
            ("odr", False, (25.6775411, 49.9770507, 5.3008476)),
        ],
    )
    def test_fit(self, optimizer, binned, expected):
        self.m.signal.metadata.Signal.binned = binned
        self.m.fit(optimizer=optimizer)
        self._check_model_values(self.m[0], expected, rtol=TOL)


class TestModelScalarVariance:
    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.ones(100))
        self.m = self.s.create_model()
        self.m.append(hs.model.components1D.Offset())

    @pytest.mark.parametrize("std, expected", [(1, 78.35015229), (10, 78.35015229)])
    def test_std1_chisq(self, std, expected):
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.set_noise_variance(std ** 2)
        self.m.fit()
        np.testing.assert_allclose(self.m.chisq.data, expected)

    @pytest.mark.parametrize("std, expected", [(1, 0.79949135), (10, 0.79949135)])
    def test_std1_red_chisq(self, std, expected):
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.set_noise_variance(std ** 2)
        self.m.fit()
        np.testing.assert_allclose(self.m.red_chisq.data, expected)

    @pytest.mark.parametrize("std, expected", [(1, 0.84233497), (10, 0.84233497)])
    def test_std1_red_chisq_in_range(self, std, expected):
        self.m.set_signal_range(10, 50)
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.set_noise_variance(std ** 2)
        self.m.fit()
        np.testing.assert_allclose(self.m.red_chisq.data, expected)


class TestFitPrintReturnInfo:
    def setup_method(self, method):
        np.random.seed(1)
        s = hs.signals.Signal1D(np.random.normal(scale=2, size=10000)).get_histogram()
        s.metadata.Signal.binned = True
        g = hs.model.components1D.Gaussian()
        self.m = s.create_model()
        self.m.append(g)
        g.sigma.value = 1
        g.centre.value = 0.5
        g.A.value = 1000

    @pytest.mark.parametrize("optimizer", ["odr", "Nelder-Mead", "L-BFGS-B"])
    def test_print_info(self, optimizer, capfd):
        self.m.fit(optimizer=optimizer, print_info=True)
        captured = capfd.readouterr()
        assert "Fit info:" in captured.out

    @pytest.mark.parametrize("bounded", [True, False])
    def test_print_info_lm(self, bounded, capfd):
        if bounded:
            self.m[0].centre.bmin = 0.5

        self.m.fit(optimizer="lm", bounded=bounded, print_info=True)
        captured = capfd.readouterr()
        assert "Fit info:" in captured.out

    def test_no_print_info(self, capfd):
        # Default is print_info=False
        self.m.fit(optimizer="lm")
        captured = capfd.readouterr()
        assert "Fit info:" not in captured.out

    @pytest.mark.parametrize("optimizer", ["odr", "Nelder-Mead", "L-BFGS-B"])
    def test_return_info(self, optimizer):
        # Default is return_info=True
        res = self.m.fit(optimizer=optimizer)
        assert isinstance(res, OptimizeResult)

    def test_no_return_info(self):
        # Default is return_info=True
        res = self.m.fit(optimizer="lm", return_info=False)
        assert res is None


class TestFitErrorsAndWarnings:
    def setup_method(self, method):
        np.random.seed(1)
        s = hs.signals.Signal1D(np.random.normal(scale=2, size=10000)).get_histogram()
        s.metadata.Signal.binned = True
        g = hs.model.components1D.Gaussian()
        m = s.create_model()
        m.append(g)
        g.sigma.value = 1
        g.centre.value = 0.5
        g.A.value = 1000
        self.m = m

    @pytest.mark.parametrize("optimizer", ["fmin", "mpfit", "leastsq"])
    def test_deprecated_optimizers(self, optimizer):
        with pytest.warns(
            VisibleDeprecationWarning,
            match=r".* has been deprecated and will be removed",
        ):
            self.m.fit(optimizer=optimizer)

    def test_deprecated_fitter(self):
        with pytest.warns(
            VisibleDeprecationWarning,
            match=r"fitter=.* has been deprecated and will be removed",
        ):
            self.m.fit(fitter="lm")

    def test_wrong_loss_function(self):
        with pytest.raises(ValueError, match="loss_function must be one of"):
            self.m.fit(loss_function="dummy")

    def test_not_support_loss_function(self):
        with pytest.raises(
            NotImplementedError, match=r".* only supports least-squares fitting"
        ):
            self.m.fit(loss_function="ML-poisson", optimizer="lm")

    def test_not_support_bounds(self):
        with pytest.raises(ValueError, match="Bounded optimization is only supported"):
            self.m.fit(optimizer="odr", bounded=True)

    def test_wrong_grad(self):
        with pytest.raises(ValueError, match="`grad` must be one of"):
            self.m.fit(grad="random")

    def test_wrong_fd_scheme(self):
        with pytest.raises(ValueError, match="`fd_scheme` must be one of"):
            self.m.fit(optimizer="L-BFGS-B", grad="fd", fd_scheme="random")

    @pytest.mark.parametrize("some_bounds", [True, False])
    def test_global_optimizer_wrong_bounds(self, some_bounds):
        if some_bounds:
            self.m[0].centre.bmin = 0.5
            self.m[0].centre.bmax = np.inf

        with pytest.raises(ValueError, match="Finite upper and lower bounds"):
            self.m.fit(optimizer="Differential Evolution", bounded=True)


class TestCustomOptimization:
    def setup_method(self, method):
        # data that should fit with A=49, centre=5.13, sigma=2.0
        s = hs.signals.Signal1D([1.0, 2, 3, 5, 7, 12, 8, 6, 3, 2, 2])
        self.m = s.create_model()
        self.m.append(hs.model.components1D.Gaussian())

        def sets_second_parameter_to_two(model, parameters, data, weights=None):
            return np.abs(parameters[1] - 2)

        self.fmin = sets_second_parameter_to_two

    def test_custom_function(self):
        self.m.fit(loss_function=self.fmin, optimizer="TNC")
        np.testing.assert_allclose(self.m[0].centre.value, 2.0)

    def test_custom_gradient_function(self):
        from unittest import mock

        gradf = mock.Mock(return_value=[10, 1, 10])
        self.m.fit(loss_function=self.fmin, optimizer="BFGS", grad=gradf)
        assert gradf.called
        assert all([args[0] is self.m for args, kwargs in gradf.call_args_list])


@lazifyTestClass
class TestMultifit:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros((2, 200)))
        s.axes_manager[-1].offset = 1
        s.data[:] = 2 * s.axes_manager[-1].axis ** (-3)
        m = s.create_model()
        m.append(hs.model.components1D.PowerLaw())
        m[0].A.value = 2
        m[0].r.value = 2
        m.store_current_values()
        m.axes_manager.indices = (1,)
        m[0].r.value = 100
        m[0].A.value = 2
        m.store_current_values()
        m[0].A.free = False
        self.m = m
        m.axes_manager.indices = (0,)
        m[0].A.value = 100

    def test_fetch_only_fixed_false(self):
        # HyperSpy 2.0: remove setting iterpath='serpentine'
        self.m.multifit(fetch_only_fixed=False, iterpath="serpentine", optimizer="trf")
        np.testing.assert_array_almost_equal(self.m[0].r.map["values"], [3.0, 100.0])
        np.testing.assert_array_almost_equal(self.m[0].A.map["values"], [2.0, 2.0])

    def test_fetch_only_fixed_true(self):
        # HyperSpy 2.0: remove setting iterpath='serpentine'
        self.m.multifit(fetch_only_fixed=True, iterpath="serpentine", optimizer="trf")
        np.testing.assert_array_almost_equal(self.m[0].r.map["values"], [3.0, 3.0])
        np.testing.assert_array_almost_equal(self.m[0].A.map["values"], [2.0, 2.0])

    def test_parameter_as_signal_values(self):
        # There are more as_signal tests in test_parameters.py
        rs = self.m[0].r.as_signal(field="values")
        np.testing.assert_allclose(rs.data, np.array([2.0, 100.0]))
        assert rs.get_noise_variance() is None
        # HyperSpy 2.0: remove setting iterpath='serpentine'
        self.m.multifit(fetch_only_fixed=True, iterpath="serpentine")
        rs = self.m[0].r.as_signal(field="values")
        assert rs.get_noise_variance() is not None
        assert isinstance(rs.get_noise_variance(), hs.signals.Signal1D)

    @pytest.mark.parametrize("optimizer", ["lm", "L-BFGS-B"])
    def test_bounded_snapping(self, optimizer):
        m = self.m
        m[0].A.free = True
        m.signal.data *= 2.0
        m[0].A.value = 2.0
        m[0].A.bmin = 3.0
        # HyperSpy 2.0: remove setting iterpath='serpentine'
        m.multifit(optimizer=optimizer, bounded=True, iterpath="serpentine")
        np.testing.assert_allclose(self.m[0].r.map["values"], [3.0, 3.0], rtol=TOL)
        np.testing.assert_allclose(self.m[0].A.map["values"], [4.0, 4.0], rtol=TOL)

    @pytest.mark.parametrize("iterpath", ["flyback", "serpentine"])
    def test_iterpaths(self, iterpath):
        self.m.multifit(iterpath=iterpath)

    def test_iterpath_none(self):
        with pytest.warns(
            VisibleDeprecationWarning,
            match="'iterpath' default will change from 'flyback' to 'serpentine'",
        ):
            self.m.multifit()  # iterpath = None by default

        with pytest.warns(
            VisibleDeprecationWarning,
            match="'iterpath' default will change from 'flyback' to 'serpentine'",
        ):
            self.m.multifit(iterpath=None)


@lazifyTestClass
class TestMultiFitSignalVariance:
    def setup_method(self, method):
        variance = hs.signals.Signal1D(
            np.arange(100, 300, dtype="float64").reshape((2, 100))
        )
        s = variance.deepcopy()
        np.random.seed(1)
        std = 10
        np.random.seed(1)
        s.add_gaussian_noise(std)
        np.random.seed(1)
        s.add_poissonian_noise()
        s.set_noise_variance(variance + std ** 2)
        m = s.create_model()
        m.append(hs.model.components1D.Polynomial(order=1, legacy=False))
        self.s = s
        self.m = m
        self.var = (variance + std ** 2).data

    def test_std1_red_chisq(self):
        # HyperSpy 2.0: remove setting iterpath='serpentine'
        self.m.multifit(iterpath="serpentine")
        np.testing.assert_allclose(self.m.red_chisq.data[0], 0.813109, rtol=TOL)
        np.testing.assert_allclose(self.m.red_chisq.data[1], 0.697727, rtol=TOL)


def test_missing_analytical_gradient():
    """Tests the error in gh-1388.

    In particular:

    > "The issue is that EELSCLEdge doesn't provide an analytical gradient for
       onset_energy. That's because it's not trivial since shifting the
       energy requires recomputing the XS."

    This creates an arbitrary dataset that closely mimics the one
    referenced in that issue.

    """
    metadata_dict = {
        "Acquisition_instrument": {
            "TEM": {
                "Detector": {"EELS": {"aperture_size": 2.5, "collection_angle": 41.0}},
                "beam_current": 0.0,
                "beam_energy": 200,
                "camera_length": 20.0,
                "convergence_angle": 31.48,
                "magnification": 400000.0,
            }
        }
    }

    np.random.seed(1)
    s = hs.signals.Signal1D(np.arange(1000).astype(float), metadata=metadata_dict)
    s.set_signal_type("EELS")
    s.add_gaussian_noise(10)
    m = s.create_model(auto_add_edges=False)

    e1 = hs.model.components1D.EELSCLEdge("Zr_L3")
    e1.intensity.bmin = 0
    e1.intensity.bmax = 0.1

    m.append(e1)

    e2 = hs.model.components1D.Gaussian()
    e2.centre.value = 2230.0
    e2.centre.bmin = 2218.0
    e2.centre.bmax = 2240.0
    e2.sigma.bmin = 0
    e2.sigma.bmax = 3
    e2.A.bmin = 0
    e2.A.bmax = 1e10
    m.append(e2)

    e1.onset_energy.twin = e2.centre

    with pytest.raises(ValueError, match=r"Analytical gradient not available for .*"):
        m.fit(grad="analytical", optimizer="L-BFGS-B", bounded=True)
