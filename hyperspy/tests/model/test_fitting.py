# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

import numpy as np
import pytest
import scipy.odr as odr
from scipy.optimize import OptimizeResult

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.external.mpfit.mpfit import mpfit


@lazifyTestClass
class TestModelFitBinned:
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

    def _check_model_values(self, model, expected, **kwargs):
        np.testing.assert_allclose(model.A.value, expected[0], **kwargs)
        np.testing.assert_allclose(model.centre.value, expected[1], **kwargs)
        np.testing.assert_allclose(model.sigma.value, expected[2], **kwargs)

    @pytest.mark.parametrize(
        "grad, bounded, expected",
        [
            (False, False, (9976.145261, -0.110611, 1.983807)),
            (True, False, (9976.145261, -0.110611, 1.983807)),
            (False, True, (9991.654220, 0.5, 2.083982)),
            (True, True, (9991.654220, 0.5, 2.083982)),
        ],
    )
    def test_fit_leastsq(self, grad, bounded, expected):
        if bounded:
            self.m[0].centre.bmin = 0.5

        self.m.fit(fitter="leastsq", grad=grad, bounded=bounded)
        self._check_model_values(self.m[0], expected, rtol=1e-5)

        if bounded:
            assert isinstance(self.m.fit_output, OptimizeResult)
        else:
            assert isinstance(self.m.fit_output, tuple)

        assert self.m.p_std is not None
        assert len(self.m.p_std) == 3
        assert np.all(~np.isinf(self.m.p_std))

    @pytest.mark.parametrize(
        "grad, expected",
        [
            (False, (9976.145320, -0.110611, 1.983807)),
            (True, (9976.145320, -0.110611, 1.983807)),
        ],
    )
    def test_fit_odr(self, grad, expected):
        self.m.fit(fitter="odr", grad=grad)
        self._check_model_values(self.m[0], expected, rtol=1e-5)
        assert isinstance(self.m.fit_output, odr.Output)

        assert self.m.p_std is not None
        assert len(self.m.p_std) == 3
        assert np.all(~np.isinf(self.m.p_std))

    @pytest.mark.parametrize(
        "grad, bounded, expected",
        [
            (False, False, (9976.145263, -0.110611, 1.983807)),
            (True, False, (9976.145261, -0.110611, 1.983807)),
            (False, True, (9991.654220, 0.5, 2.083982)),
            (True, True, (9991.6542201, 0.5, 2.083982)),
        ],
    )
    def test_fit_mpfit(self, grad, bounded, expected):
        if bounded:
            self.m[0].centre.bmin = 0.5

        with pytest.warns(
            VisibleDeprecationWarning,
            match=r"The method .* has been deprecated and will be removed",
        ):
            self.m.fit(fitter="mpfit", grad=grad, bounded=bounded)

        self._check_model_values(self.m[0], expected, rtol=1e-5)
        assert isinstance(self.m.fit_output, mpfit)

        assert self.m.p_std is not None
        assert len(self.m.p_std) == 3
        assert np.all(~np.isinf(self.m.p_std))

    @pytest.mark.parametrize(
        "fitter, expected", [("leastsq", (9991.654220, 0.5, 2.083982))],
    )
    def test_fit_bounded_bad_starting_values(self, fitter, expected):
        self.m[0].centre.bmin = 0.5
        self.m[0].centre.value = -1
        self.m.fit(fitter=fitter, bounded=True)
        self._check_model_values(self.m[0], expected, rtol=1e-5)

    @pytest.mark.parametrize(
        "fitter, expected", [("leastsq", (9950.0, 0.5, 2.078154))],
    )
    def test_fit_ext_bounding(self, fitter, expected):
        self.m[0].A.bmin = 9950.0
        self.m[0].A.bmax = 10050.0
        self.m[0].centre.bmin = 0.5
        self.m[0].centre.bmax = 5.0
        self.m[0].sigma.bmin = 0.5
        self.m[0].sigma.bmax = 2.5

        with pytest.warns(
            VisibleDeprecationWarning, match="`ext_bounding=True` has been deprecated",
        ):
            self.m.fit(fitter=fitter, ext_bounding=True)

        self._check_model_values(self.m[0], expected, rtol=1e-5)


class TestModelFitBinnedLocal:
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

    def _check_model_values(self, model, expected, **kwargs):
        np.testing.assert_allclose(model.A.value, expected[0], **kwargs)
        np.testing.assert_allclose(model.centre.value, expected[1], **kwargs)
        np.testing.assert_allclose(model.sigma.value, expected[2], **kwargs)

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    @pytest.mark.parametrize(
        "fitter, method, grad, bounded, expected",
        [
            ("Nelder-Mead", "ls", False, False, (9976.145193, -0.110611, 1.983807)),
            ("Nelder-Mead", "ml", False, False, (10001.396139, -0.104151, 2.000536)),
            ("Nelder-Mead", "huber", False, False, (10032.953495, -0.110309, 1.987885)),
            ("L-BFGS-B", "ls", True, False, (9976.145193, -0.110611, 1.983807)),
            ("L-BFGS-B", "ml", True, False, (10001.396139, -0.104148, 2.000536)),
            ("L-BFGS-B", "huber", True, False, (10032.953495, -0.110309, 1.987885)),
            ("L-BFGS-B", "ls", False, True, (9991.627563, 0.5, 2.083987)),
            ("L-BFGS-B", "ls", True, True, (9991.627563, 0.5, 2.083987)),
        ],
    )
    def test_fit_scipy_minimize(self, fitter, method, grad, bounded, expected):
        if bounded:
            self.m[0].centre.bmin = 0.5

        self.m.fit(fitter=fitter, method=method, grad=grad, bounded=bounded)
        self._check_model_values(self.m[0], expected, rtol=1e-5)
        assert isinstance(self.m.fit_output, OptimizeResult)

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    @pytest.mark.parametrize(
        "grad, expected",
        [
            (False, (9976.145193, -0.110611, 1.983807)),
            (True, (9976.145193, -0.110611, 1.983807)),
            ("2-point", (9976.145193, -0.110611, 1.983807)),
            ("3-point", (9976.145193, -0.110611, 1.983807)),
        ],
    )
    def test_fit_scipy_minimize_gradients(self, grad, expected):
        self.m.fit(fitter="L-BFGS-B", method="ls", grad=grad)
        self._check_model_values(self.m[0], expected, rtol=1e-5)
        assert isinstance(self.m.fit_output, OptimizeResult)

    @pytest.mark.parametrize(
        "fitter, expected",
        [
            ("Powell", (9991.464524, 0.500064, 2.083900)),
            ("L-BFGS-B", (9991.654220, 0.5, 2.083982)),
        ],
    )
    def test_fit_bounded_bad_starting_values(self, fitter, expected):
        self.m[0].centre.bmin = 0.5
        self.m[0].centre.value = -1
        self.m.fit(fitter=fitter, bounded=True)
        self._check_model_values(self.m[0], expected, rtol=1e-5)

    @pytest.mark.parametrize(
        "fitter, expected",
        [("Powell", (9950.0, 0.5, 2.078154)), ("L-BFGS-B", (9950.0, 0.5, 2.078154))],
    )
    def test_fit_ext_bounding(self, fitter, expected):
        self.m[0].A.bmin = 9950.0
        self.m[0].A.bmax = 10050.0
        self.m[0].centre.bmin = 0.5
        self.m[0].centre.bmax = 5.0
        self.m[0].sigma.bmin = 0.5
        self.m[0].sigma.bmax = 2.5

        with pytest.warns(
            VisibleDeprecationWarning, match="`ext_bounding=True` has been deprecated",
        ):
            self.m.fit(fitter=fitter, ext_bounding=True)

        self._check_model_values(self.m[0], expected, rtol=1e-5)

    @pytest.mark.parametrize(
        "fitter, expected", [("SLSQP", (988.401164, -177.122887, -10.100562))]
    )
    def test_constraints(self, fitter, expected):
        # Primarily checks that constraints are passed correctly,
        # even though the end result is a bad fit
        cons = {"type": "ineq", "fun": lambda x: x[0] - x[1]}
        self.m.fit(fitter=fitter, constraints=cons)
        self._check_model_values(self.m[0], expected, rtol=1e-5)


class TestModelFitBinnedGlobal:
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

    def _check_model_values(self, model, expected, **kwargs):
        np.testing.assert_allclose(model.A.value, expected[0], **kwargs)
        np.testing.assert_allclose(model.centre.value, expected[1], **kwargs)
        np.testing.assert_allclose(model.sigma.value, expected[2], **kwargs)

    @pytest.mark.parametrize(
        "method, expected",
        [
            ("ls", (9972.351479, -0.110612, 1.983298)),
            ("ml", (10046.513541, -0.104155, 2.000547)),
            ("huber", (10032.952811, -0.110309, 1.987885)),
        ],
    )
    def test_fit_differential_evolution(self, method, expected):
        self.m[0].A.bmin = 9950.0
        self.m[0].A.bmax = 10050.0
        self.m[0].centre.bmin = -5.0
        self.m[0].centre.bmax = 5.0
        self.m[0].sigma.bmin = 0.5
        self.m[0].sigma.bmax = 2.5

        self.m.fit(fitter="Differential Evolution", method=method, bounded=True, seed=1)
        self._check_model_values(self.m[0], expected, rtol=1e-5)
        assert isinstance(self.m.fit_output, OptimizeResult)

    @pytest.mark.parametrize(
        "method, expected",
        [
            ("ls", (9976.145304, -0.110611, 1.983807)),
            ("ml", (10001.395614, -0.104151, 2.000536)),
            ("huber", (10032.952811, -0.110309, 1.987885)),
        ],
    )
    def test_fit_dual_annealing(self, method, expected):
        pytest.importorskip("scipy", minversion="1.2.0")
        self.m[0].A.bmin = 9950.0
        self.m[0].A.bmax = 10050.0
        self.m[0].centre.bmin = -5.0
        self.m[0].centre.bmax = 5.0
        self.m[0].sigma.bmin = 0.5
        self.m[0].sigma.bmax = 2.5

        self.m.fit(fitter="Dual Annealing", method=method, bounded=True, seed=1)
        self._check_model_values(self.m[0], expected, rtol=1e-5)
        assert isinstance(self.m.fit_output, OptimizeResult)

    @pytest.mark.parametrize(
        "method, expected",
        [
            ("ls", (9997.107685, -0.289231, 1.557846)),
            ("ml", (9999.999922, -0.104151, 2.000536)),
            ("huber", (10032.952811, -0.110309, 1.987885)),
        ],
    )
    def test_fit_shgo(self, method, expected):
        pytest.importorskip("scipy", minversion="1.2.0")
        self.m[0].A.bmin = 9950.0
        self.m[0].A.bmax = 10050.0
        self.m[0].centre.bmin = -5.0
        self.m[0].centre.bmax = 5.0
        self.m[0].sigma.bmin = 0.5
        self.m[0].sigma.bmax = 2.5

        self.m.fit(fitter="SHGO", method=method, bounded=True)
        self._check_model_values(self.m[0], expected, rtol=1e-5)
        assert isinstance(self.m.fit_output, OptimizeResult)


@lazifyTestClass
class TestModelWeighted:
    def setup_method(self, method):
        np.random.seed(1)
        s = hs.signals.Signal1D(np.arange(10, 100, 0.1))
        self.s_var = hs.signals.Signal1D(np.arange(10, 100, 0.01))
        s.set_noise_variance(self.s_var)
        s.axes_manager[0].scale = 0.1
        s.axes_manager[0].offset = 10
        s.add_poissonian_noise()
        self.m = s.create_model()
        self.m.append(hs.model.components1D.Polynomial(1, legacy=False))

    def _check_model_values(self, model, expected, **kwargs):
        np.testing.assert_allclose(model.a1.value, expected[0], **kwargs)
        np.testing.assert_allclose(model.a0.value, expected[1], **kwargs)

    def test_chisq(self):
        self.m.signal.metadata.Signal.binned = True
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.chisq.data, 3029.16949561)

    def test_red_chisq(self):
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.red_chisq.data, 3.37700055)

    @pytest.mark.parametrize(
        "fitter, binned, expected",
        [
            ("leastsq", True, [9.916560, 1.662824]),
            ("odr", True, [9.916560, 1.662825]),
            ("Nelder-Mead", True, [9.913729, 1.844601]),
            ("leastsq", False, [0.991656, 0.166283]),
            ("odr", False, [0.991656, 0.166282]),
            ("Nelder-Mead", False, [0.991362, 0.184831]),
        ],
    )
    def test_fit(self, fitter, binned, expected):
        self.m.signal.metadata.Signal.binned = binned
        self.m.fit(fitter=fitter, method="ls")
        self._check_model_values(self.m[0], expected, rtol=1e-5)


@lazifyTestClass
class TestModelWeightedOptions:
    def setup_method(self, method):
        np.random.seed(1)
        self.s = hs.signals.Signal1D(np.arange(10, 100, 0.1))
        self.s.set_noise_variance(hs.signals.Signal1D(np.arange(10, 100, 0.01)))
        self.s.axes_manager[0].scale = 0.1
        self.s.axes_manager[0].offset = 10
        self.s.add_poissonian_noise()
        self.m = self.s.create_model()
        self.m.append(hs.model.components1D.Polynomial(1, legacy=False))

    def test_red_chisq_default(self):
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.red_chisq.data, 3.377001, rtol=1e-5)

    def test_red_chisq_inv_var(self):
        self.m.fit(fitter="leastsq", method="ls", weights="inverse_variance")
        np.testing.assert_allclose(self.m.red_chisq.data, 3.377001, rtol=1e-5)

    def test_red_chisq_unweighted(self):
        self.m.fit(fitter="leastsq", method="ls", weights=None)
        np.testing.assert_allclose(self.m.red_chisq.data, 3.377004, rtol=1e-5)


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
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.chisq.data, expected)

    @pytest.mark.parametrize("std, expected", [(1, 0.79949135), (10, 0.79949135)])
    def test_std1_red_chisq(self, std, expected):
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.set_noise_variance(std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.red_chisq.data, expected)

    @pytest.mark.parametrize("std, expected", [(1, 0.86206965), (10, 0.86206965)])
    def test_std1_red_chisq_in_range(self, std, expected):
        self.m.set_signal_range(10, 50)
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.set_noise_variance(std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.red_chisq.data, expected)


@pytest.mark.filterwarnings("ignore:The API of the `Polynomial`")
@lazifyTestClass
class TestModelSignalVariance:
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
        m.append(hs.model.components1D.Polynomial(order=1))
        self.s = s
        self.m = m
        self.var = (variance + std ** 2).data

    def test_std1_red_chisq(self):
        # HyperSpy 2.0: remove setting iterpath='serpentine'
        self.m.multifit(fitter="leastsq", method="ls", iterpath="serpentine")
        np.testing.assert_allclose(self.m.red_chisq.data[0], 0.813109, atol=1e-5)
        np.testing.assert_allclose(self.m.red_chisq.data[1], 0.697727, atol=1e-5)


class TestFitPrintInfo:
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

    @pytest.mark.parametrize("fitter", ["odr", "Nelder-Mead", "L-BFGS-B"])
    def test_print_info(self, fitter, capfd):
        self.m.fit(fitter=fitter, print_info=True)
        captured = capfd.readouterr()
        assert "Fit info:" in captured.out

    @pytest.mark.parametrize("bounded", [True, False])
    def test_print_info_leastsq(self, bounded, capfd):
        if bounded:
            self.m[0].centre.bmin = 0.5

        self.m.fit(fitter="leastsq", bounded=bounded, print_info=True)
        captured = capfd.readouterr()
        assert "Fit info:" in captured.out

    def test_print_info_mpfit(self, capfd):
        with pytest.warns(
            VisibleDeprecationWarning,
            match=r"The method .* has been deprecated and will be removed",
        ):
            self.m.fit(fitter="mpfit", print_info=True)

        captured = capfd.readouterr()
        assert "Fit info:" in captured.out

    def test_no_print_info(self, capfd):
        self.m.fit(fitter="leastsq")  # Default is print_info=False
        captured = capfd.readouterr()
        assert "Fit info:" not in captured.out


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

    @pytest.mark.parametrize("fitter", ["fmin", "mpfit"])
    def test_deprecated_fitters(self, fitter):
        with pytest.warns(
            VisibleDeprecationWarning,
            match=r"The method .* has been deprecated and will be removed",
        ):
            self.m.fit(fitter=fitter)

    def test_wrong_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            self.m.fit(method="dummy")

    def test_not_support_method(self):
        with pytest.raises(
            NotImplementedError, match="optimizers only support least-squares fitting"
        ):
            self.m.fit(method="ml", fitter="leastsq")

    def test_not_support_bounds(self):
        with pytest.raises(ValueError, match="Bounded optimization is only supported"):
            self.m.fit(fitter="odr", bounded=True)

    def test_wrong_grad(self):
        with pytest.raises(ValueError, match="grad must be one of"):
            self.m.fit(grad="random")

    @pytest.mark.parametrize(
        "fitter", ["Differential Evolution", "Dual Annealing", "SHGO"]
    )
    def test_global_optimizer_no_bounds(self, fitter):
        with pytest.raises(
            ValueError, match=r"Bounds must be specified for fitter=.*",
        ):
            self.m.fit(fitter=fitter, bounded=False)

    @pytest.mark.parametrize(
        "fitter", ["Differential Evolution", "Dual Annealing", "SHGO"]
    )
    def test_global_optimizer_wrong_bounds(self, fitter):
        self.m[0].centre.bmin = 0.5
        self.m[0].centre.bmax = np.inf

        with pytest.raises(
            ValueError,
            match="Finite upper and lower bounds must be specified for every free parameter",
        ):
            self.m.fit(fitter=fitter, bounded=True)


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
        self.m.fit(method="custom", min_function=self.fmin, fitter="TNC")
        assert self.m[0].centre.value == 2.0

    def test_no_function(self):
        with pytest.raises(ValueError, match="Custom minimization requires"):
            self.m.fit(method="custom", fitter="TNC")

    def test_no_gradient(self):
        with pytest.raises(ValueError, match="Custom minimization with gradients"):
            self.m.fit(
                method="custom", min_function=lambda *args: 1, grad=True, fitter="BFGS"
            )

    def test_custom_gradient_function(self):
        from unittest import mock

        gradf = mock.Mock(return_value=[10, 1, 10])
        self.m.fit(
            method="custom",
            fitter="BFGS",
            min_function=self.fmin,
            grad=True,
            min_function_grad=gradf,
        )
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
        self.m.multifit(fetch_only_fixed=False, iterpath="serpentine")
        np.testing.assert_array_almost_equal(self.m[0].r.map["values"], [3.0, 100.0])
        np.testing.assert_array_almost_equal(self.m[0].A.map["values"], [2.0, 2.0])

    def test_fetch_only_fixed_true(self):
        # HyperSpy 2.0: remove setting iterpath='serpentine'
        self.m.multifit(fetch_only_fixed=True, iterpath="serpentine")
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

    @pytest.mark.parametrize("fitter", ["leastsq", "L-BFGS-B"])
    def test_bounded_snapping(self, fitter):
        m = self.m
        m[0].A.free = True
        m.signal.data *= 2.0
        m[0].A.value = 2.0
        m[0].A.bmin = 3.0
        # HyperSpy 2.0: remove setting iterpath='serpentine'
        m.multifit(fitter=fitter, bounded=True, iterpath="serpentine")
        np.testing.assert_allclose(self.m[0].r.map["values"], [3.0, 3.0], rtol=1e-5)
        np.testing.assert_allclose(self.m[0].A.map["values"], [4.0, 4.0], rtol=1e-5)

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
