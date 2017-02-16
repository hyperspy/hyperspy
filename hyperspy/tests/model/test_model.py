from distutils.version import LooseVersion
from unittest import mock

import numpy as np
import scipy

import pytest
from numpy.testing import assert_allclose

import hyperspy.api as hs
from hyperspy.misc.utils import slugify


class TestModelJacobians:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros(1))
        m = s.create_model()
        self.low_loss = 7.
        self.weights = 0.3
        m.axis.axis = np.array([1, 0])
        m.channel_switches = np.array([0, 1], dtype=bool)
        m.append(hs.model.components1D.Gaussian())
        m[0].A.value = 1
        m[0].centre.value = 2.
        m[0].sigma.twin = m[0].centre
        m._low_loss = mock.MagicMock()
        m.low_loss.return_value = self.low_loss
        self.model = m
        m.convolution_axis = np.zeros(2)

    def test_jacobian_not_convolved(self):
        m = self.model
        m.convolved = False
        jac = m._jacobian((1, 2, 3), None, weights=self.weights)
        np.testing.assert_array_almost_equal(jac.squeeze(), self.weights *
                                             np.array([m[0].A.grad(0),
                                                       m[0].sigma.grad(0) +
                                                       m[0].centre.grad(0)]))
        assert m[0].A.value == 1
        assert m[0].centre.value == 2
        assert m[0].sigma.value == 2

    def test_jacobian_convolved(self):
        m = self.model
        m.convolved = True
        m.append(hs.model.components1D.Gaussian())
        m[0].convolved = False
        m[1].convolved = True
        jac = m._jacobian((1, 2, 3, 4, 5), None, weights=self.weights)
        np.testing.assert_array_almost_equal(jac.squeeze(), self.weights *
                                             np.array([m[0].A.grad(0),
                                                       m[0].sigma.grad(0) +
                                                       m[0].centre.grad(0),
                                                       m[1].A.grad(0) *
                                                       self.low_loss,
                                                       m[1].centre.grad(0) *
                                                       self.low_loss,
                                                       m[1].sigma.grad(0) *
                                                       self.low_loss,
                                                       ]))
        assert m[0].A.value == 1
        assert m[0].centre.value == 2
        assert m[0].sigma.value == 2
        assert m[1].A.value == 3
        assert m[1].centre.value == 4
        assert m[1].sigma.value == 5


class TestModelCallMethod:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.empty(1))
        m = s.create_model()
        m.append(hs.model.components1D.Gaussian())
        m.append(hs.model.components1D.Gaussian())
        self.model = m

    def test_call_method_no_convolutions(self):
        m = self.model
        m.convolved = False

        m[1].active = False
        r1 = m()
        r2 = m(onlyactive=True)
        np.testing.assert_allclose(m[0].function(0) * 2, r1)
        np.testing.assert_allclose(m[0].function(0), r2)

        m.convolved = True
        r1 = m(non_convolved=True)
        r2 = m(non_convolved=True, onlyactive=True)
        np.testing.assert_allclose(m[0].function(0) * 2, r1)
        np.testing.assert_allclose(m[0].function(0), r2)

    def test_call_method_with_convolutions(self):
        m = self.model
        m._low_loss = mock.MagicMock()
        m.low_loss.return_value = 0.3
        m.convolved = True

        m.append(hs.model.components1D.Gaussian())
        m[1].active = False
        m[0].convolved = True
        m[1].convolved = False
        m[2].convolved = False
        m.convolution_axis = np.array([0., ])

        r1 = m()
        r2 = m(onlyactive=True)
        np.testing.assert_allclose(m[0].function(0) * 2.3, r1)
        np.testing.assert_allclose(m[0].function(0) * 1.3, r2)

    def test_call_method_binned(self):
        m = self.model
        m.convolved = False
        m.remove(1)
        m.signal.metadata.Signal.binned = True
        m.signal.axes_manager[-1].scale = 0.3
        r1 = m()
        np.testing.assert_allclose(m[0].function(0) * 0.3, r1)


class TestModelPlotCall:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.empty(1))
        m = s.create_model()
        m.__call__ = mock.MagicMock()
        m.__call__.return_value = np.array([0.5, 0.25])
        m.axis = mock.MagicMock()
        m.fetch_stored_values = mock.MagicMock()
        m.channel_switches = np.array([0, 1, 1, 0, 0], dtype=bool)
        self.model = m

    def test_model2plot_own_am(self):
        m = self.model
        m.axis.axis.shape = (5,)
        res = m._model2plot(m.axes_manager)
        np.testing.assert_array_equal(
            res, np.array([np.nan, 0.5, 0.25, np.nan, np.nan]))
        assert m.__call__.called
        assert (
            m.__call__.call_args[1] == {
                'non_convolved': False, 'onlyactive': True})
        assert not m.fetch_stored_values.called

    def test_model2plot_other_am(self):
        m = self.model
        res = m._model2plot(m.axes_manager.deepcopy(), out_of_range2nans=False)
        np.testing.assert_array_equal(res, np.array([0.5, 0.25]))
        assert m.__call__.called
        assert (
            m.__call__.call_args[1] == {
                'non_convolved': False, 'onlyactive': True})
        assert 2 == m.fetch_stored_values.call_count


class TestModelSettingPZero:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.empty(1))
        m = s.create_model()
        m.append(hs.model.components1D.Gaussian())

        m[0].A.value = 1.1
        m[0].centre._number_of_elements = 2
        m[0].centre.value = (2.2, 3.3)
        m[0].sigma.value = 4.4
        m[0].sigma.free = False

        m[0].A._bounds = (0.1, 0.11)
        m[0].centre._bounds = ((0.2, 0.21), (0.3, 0.31))
        m[0].sigma._bounds = (0.4, 0.41)

        self.model = m

    def test_setting_p0(self):
        m = self.model
        m.append(hs.model.components1D.Gaussian())
        m[-1].active = False
        m.p0 = None
        m._set_p0()
        assert m.p0 == (1.1, 2.2, 3.3)

    def test_fetching_from_p0(self):
        m = self.model

        m.append(hs.model.components1D.Gaussian())
        m[-1].active = False
        m[-1].A.value = 100
        m[-1].sigma.value = 200
        m[-1].centre.value = 300

        m.p0 = (1.2, 2.3, 3.4, 5.6, 6.7, 7.8)
        m._fetch_values_from_p0()
        assert m[0].A.value == 1.2
        assert m[0].centre.value == (2.3, 3.4)
        assert m[0].sigma.value == 4.4
        assert m[1].A.value == 100
        assert m[1].sigma.value == 200
        assert m[1].centre.value == 300

    def test_setting_boundaries(self):
        m = self.model
        m.append(hs.model.components1D.Gaussian())
        m[-1].active = False
        m.set_boundaries()
        assert (m.free_parameters_boundaries ==
                [(0.1, 0.11), (0.2, 0.21), (0.3, 0.31)])

    def test_setting_mpfit_parameters_info(self):
        m = self.model
        m[0].A.bmax = None
        m[0].centre.bmin = None
        m[0].centre.bmax = 0.31
        m.append(hs.model.components1D.Gaussian())
        m[-1].active = False
        m.set_mpfit_parameters_info()
        assert (m.mpfit_parinfo ==
                [{'limited': [True, False],
                  'limits': [0.1, 0]},
                 {'limited': [False, True],
                  'limits': [0, 0.31]},
                 {'limited': [False, True],
                  'limits': [0, 0.31]},
                 ])


class TestModel1D:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.empty(1))
        m = s.create_model()
        self.model = m

    def test_errfunc(self):
        m = self.model
        m._model_function = mock.MagicMock()
        m._model_function.return_value = 3.
        np.testing.assert_equal(m._errfunc(None, 1., None), 2.)
        np.testing.assert_equal(m._errfunc(None, 1., 0.3), 0.6)

    def test_errfunc2(self):
        m = self.model
        m._model_function = mock.MagicMock()
        m._model_function.return_value = 3. * np.ones(2)
        np.testing.assert_equal(m._errfunc2(None, np.ones(2), None), 2 * 4.)
        np.testing.assert_equal(m._errfunc2(None, np.ones(2), 0.3), 2 * 0.36)

    def test_gradient_ls(self):
        m = self.model
        m._errfunc = mock.MagicMock()
        m._errfunc.return_value = 0.1
        m._jacobian = mock.MagicMock()
        m._jacobian.return_value = np.ones((1, 2)) * 7.
        np.testing.assert_equal(m._gradient_ls(None, None), 2 * 0.1 * 7 * 2)

    def test_gradient_ml(self):
        m = self.model
        m._model_function = mock.MagicMock()
        m._model_function.return_value = 3. * np.ones(2)
        m._jacobian = mock.MagicMock()
        m._jacobian.return_value = np.ones((1, 2)) * 7.
        np.testing.assert_equal(
            m._gradient_ml(None, 1.2), -2 * 7 * (1.2 / 3 - 1))

    def test_model_function(self):
        m = self.model
        m.append(hs.model.components1D.Gaussian())
        m[0].A.value = 1.3
        m[0].centre.value = 0.003
        m[0].sigma.value = 0.1
        param = (100, 0.1, 0.2)
        np.testing.assert_array_almost_equal(176.03266338,
                                             m._model_function(param))
        assert m[0].A.value == 100
        assert m[0].centre.value == 0.1
        assert m[0].sigma.value == 0.2

    def test_append_existing_component(self):
        g = hs.model.components1D.Gaussian()
        m = self.model
        m.append(g)
        with pytest.raises(ValueError):
            m.append(g)

    def test_append_component(self):
        g = hs.model.components1D.Gaussian()
        m = self.model
        m.append(g)
        assert g in m
        assert g.model is m
        assert g._axes_manager is m.axes_manager
        assert all([hasattr(p, 'map') for p in g.parameters])

    def test_calculating_convolution_axis(self):
        m = self.model
        # setup
        m.axis.offset = 10
        m.axis.size = 10
        ll_axis = mock.MagicMock()
        ll_axis.size = 7
        ll_axis.value2index.return_value = 3
        m._low_loss = mock.MagicMock()
        m.low_loss.axes_manager.signal_axes = [ll_axis, ]

        # calculation
        m.set_convolution_axis()

        # tests
        np.testing.assert_array_equal(m.convolution_axis, np.arange(7, 23))
        np.testing.assert_equal(ll_axis.value2index.call_args[0][0], 0)

    @pytest.mark.parallel
    def test_notebook_interactions(self):
        ipywidgets = pytest.importorskip("ipywidgets", minversion="5.0")
        ipython = pytest.importorskip("IPython")
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None or not getattr(ip, 'kernel', None):
            pytest.skip("Not attached to notebook")
        m = self.model
        m.notebook_interaction()
        m.append(hs.model.components1D.Offset())
        m[0].notebook_interaction()
        m[0].offset.notebook_interaction()

    def test_access_component_by_name(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        g2 = hs.model.components1D.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        assert m["test"] is g2

    def test_access_component_by_index(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        g2 = hs.model.components1D.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        assert m[1] is g2

    def test_component_name_when_append(self):
        m = self.model
        gs = [
            hs.model.components1D.Gaussian(),
            hs.model.components1D.Gaussian(),
            hs.model.components1D.Gaussian()]
        m.extend(gs)
        assert m['Gaussian'] is gs[0]
        assert m['Gaussian_0'] is gs[1]
        assert m['Gaussian_1'] is gs[2]

    def test_several_component_with_same_name(self):
        m = self.model
        gs = [
            hs.model.components1D.Gaussian(),
            hs.model.components1D.Gaussian(),
            hs.model.components1D.Gaussian()]
        m.extend(gs)
        m[0]._name = "hs.model.components1D.Gaussian"
        m[1]._name = "hs.model.components1D.Gaussian"
        m[2]._name = "hs.model.components1D.Gaussian"
        with pytest.raises(ValueError):
            m['Gaussian']

    def test_no_component_with_that_name(self):
        m = self.model
        with pytest.raises(ValueError):
            m['Voigt']

    def test_component_already_in_model(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        with pytest.raises(ValueError):
            m.extend((g1, g1))

    def test_remove_component(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        m.remove(g1)
        assert len(m) == 0

    def test_remove_component_by_index(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        m.remove(0)
        assert len(m) == 0

    def test_remove_component_by_name(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        m.remove(g1.name)
        assert len(m) == 0

    def test_delete_component_by_index(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        del m[0]
        assert g1 not in m

    def test_delete_component_by_name(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        del m[g1.name]
        assert g1 not in m

    def test_delete_slice(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        g2 = hs.model.components1D.Gaussian()
        g3 = hs.model.components1D.Gaussian()
        m.extend([g1, g2, g3])
        del m[:2]
        assert g1 not in m
        assert g2 not in m
        assert g3 in m

    def test_get_component_by_name(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        g2 = hs.model.components1D.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        assert m._get_component("test") is g2

    def test_get_component_by_index(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        g2 = hs.model.components1D.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        assert m._get_component(1) is g2

    def test_get_component_by_component(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        g2 = hs.model.components1D.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        assert m._get_component(g2) is g2

    def test_get_component_wrong(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        g2 = hs.model.components1D.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        with pytest.raises(ValueError):
            m._get_component(1.2)

    def test_components_class_default(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        assert getattr(m.components, g1.name) is g1

    def test_components_class_change_name(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        g1.name = "test"
        assert getattr(m.components, g1.name) is g1

    def test_components_class_change_name_del_default(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        g1.name = "test"
        with pytest.raises(AttributeError):
            getattr(m.components, "Gaussian")

    def test_components_class_change_invalid_name(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        g1.name = "1, Test This!"
        assert (
            getattr(m.components,
                    slugify(g1.name, valid_variable_name=True)) is g1)

    def test_components_class_change_name_del_default(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        invalid_name = "1, Test This!"
        g1.name = invalid_name
        g1.name = "test"
        with pytest.raises(AttributeError):
            getattr(m.components, slugify(invalid_name))

    def test_snap_parameter_bounds(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        g2 = hs.model.components1D.Gaussian()
        m.append(g2)
        g3 = hs.model.components1D.Gaussian()
        m.append(g3)
        g4 = hs.model.components1D.Gaussian()
        m.append(g4)
        p = hs.model.components1D.Polynomial(3)
        m.append(p)

        g1.A.value = 3.
        g1.centre.bmin = 300.
        g1.centre.value = 1.
        g1.sigma.bmax = 15.
        g1.sigma.value = 30

        g2.A.value = 1
        g2.A.bmin = 0.
        g2.A.bmax = 3.
        g2.centre.value = 0
        g2.centre.bmin = 1
        g2.centre.bmax = 3.
        g2.sigma.value = 4
        g2.sigma.bmin = 1
        g2.sigma.bmax = 3.

        g3.A.bmin = 0
        g3.A.value = -3
        g3.A.free = False

        g3.centre.value = 15
        g3.centre.bmax = 10
        g3.centre.free = False

        g3.sigma.value = 1
        g3.sigma.bmin = 0
        g3.sigma.bmax = 0

        g4.active = False
        g4.A.value = 300
        g4.A.bmin = 500
        g4.centre.value = 0
        g4.centre.bmax = -1
        g4.sigma.value = 1
        g4.sigma.bmin = 10

        p.coefficients.value = (1, 2, 3, 4)
        p.coefficients.bmin = 2
        p.coefficients.bmax = 3
        m.ensure_parameters_in_bounds()
        np.testing.assert_allclose(g1.A.value, 3.)
        np.testing.assert_allclose(g2.A.value, 1.)
        np.testing.assert_allclose(g3.A.value, -3.)
        np.testing.assert_allclose(g4.A.value, 300.)

        np.testing.assert_allclose(g1.centre.value, 300.)
        np.testing.assert_allclose(g2.centre.value, 1.)
        np.testing.assert_allclose(g3.centre.value, 15.)
        np.testing.assert_allclose(g4.centre.value, 0)

        np.testing.assert_allclose(g1.sigma.value, 15.)
        np.testing.assert_allclose(g2.sigma.value, 3.)
        np.testing.assert_allclose(g3.sigma.value, 0.)
        np.testing.assert_allclose(g4.sigma.value, 1)

        np.testing.assert_allclose(p.coefficients.value, (2, 2, 3, 3))


class TestModel2D:

    def setup_method(self, method):
        g = hs.model.components2D.Gaussian2D(
            centre_x=-5.,
            centre_y=-5.,
            sigma_x=1.,
            sigma_y=2.)
        x = np.arange(-10, 10, 0.01)
        y = np.arange(-10, 10, 0.01)
        X, Y = np.meshgrid(x, y)
        im = hs.signals.Signal2D(g.function(X, Y))
        im.axes_manager[0].scale = 0.01
        im.axes_manager[0].offset = -10
        im.axes_manager[1].scale = 0.01
        im.axes_manager[1].offset = -10
        self.im = im

    def test_fitting(self):
        im = self.im
        m = im.create_model()
        gt = hs.model.components2D.Gaussian2D(centre_x=-4.5,
                                              centre_y=-4.5,
                                              sigma_x=0.5,
                                              sigma_y=1.5)
        m.append(gt)
        m.fit()
        np.testing.assert_allclose(gt.centre_x.value, -5.)
        np.testing.assert_allclose(gt.centre_y.value, -5.)
        np.testing.assert_allclose(gt.sigma_x.value, 1.)
        np.testing.assert_allclose(gt.sigma_y.value, 2.)


class TestModelFitBinned:

    def setup_method(self, method):
        np.random.seed(1)
        s = hs.signals.Signal1D(
            np.random.normal(
                scale=2,
                size=10000)).get_histogram()
        s.metadata.Signal.binned = True
        g = hs.model.components1D.Gaussian()
        m = s.create_model()
        m.append(g)
        g.sigma.value = 1
        g.centre.value = 0.5
        g.A.value = 1e3
        self.m = m

    def test_fit_neldermead_leastsq(self):
        self.m.fit(fitter="Nelder-Mead", method="ls")
        np.testing.assert_allclose(self.m[0].A.value, 9976.14519369)
        np.testing.assert_allclose(self.m[0].centre.value, -0.110610743285)
        np.testing.assert_allclose(self.m[0].sigma.value, 1.98380705455)

    def test_fit_neldermead_ml(self):
        self.m.fit(fitter="Nelder-Mead", method="ml")
        np.testing.assert_allclose(self.m[0].A.value, 10001.39613936,
                                   atol=1E-3)
        np.testing.assert_allclose(self.m[0].centre.value, -0.104151206314,
                                   atol=1E-6)
        np.testing.assert_allclose(self.m[0].sigma.value, 2.00053642434)

    def test_fit_leastsq(self):
        self.m.fit(fitter="leastsq")
        np.testing.assert_allclose(self.m[0].A.value, 9976.14526082, 1)
        np.testing.assert_allclose(self.m[0].centre.value, -0.110610727064)
        np.testing.assert_allclose(self.m[0].sigma.value, 1.98380707571, 5)

    def test_fit_mpfit(self):
        self.m.fit(fitter="mpfit")
        np.testing.assert_allclose(self.m[0].A.value, 9976.14526286, 5)
        np.testing.assert_allclose(self.m[0].centre.value, -0.110610718444,
                                   atol=1E-6)
        np.testing.assert_allclose(self.m[0].sigma.value, 1.98380707614,
                                   atol=1E-6)

    def test_fit_odr(self):
        self.m.fit(fitter="odr")
        np.testing.assert_allclose(self.m[0].A.value, 9976.14531979, 3)
        np.testing.assert_allclose(self.m[0].centre.value, -0.110610724054,
                                   atol=1e-7)
        np.testing.assert_allclose(self.m[0].sigma.value, 1.98380709939)

    def test_fit_leastsq_grad(self):
        self.m.fit(fitter="leastsq", grad=True)
        np.testing.assert_allclose(self.m[0].A.value, 9976.14526084)
        np.testing.assert_allclose(self.m[0].centre.value, -0.11061073306)
        np.testing.assert_allclose(self.m[0].sigma.value, 1.98380707552)

    def test_fit_mpfit_grad(self):
        self.m.fit(fitter="mpfit", grad=True)
        np.testing.assert_allclose(self.m[0].A.value, 9976.14526084)
        np.testing.assert_allclose(self.m[0].centre.value, -0.11061073306)
        np.testing.assert_allclose(self.m[0].sigma.value, 1.98380707552)

    def test_fit_odr_grad(self):
        self.m.fit(fitter="odr", grad=True)
        np.testing.assert_allclose(self.m[0].A.value, 9976.14531979, 3)
        np.testing.assert_allclose(self.m[0].centre.value, -0.110610724054,
                                   atol=1e-7)
        np.testing.assert_allclose(self.m[0].sigma.value, 1.98380709939)

    def test_fit_bounded_mpfit(self):
        self.m[0].centre.bmin = 0.5
        # self.m[0].bounded = True
        self.m.fit(fitter="mpfit", bounded=True)
        np.testing.assert_allclose(self.m[0].A.value, 9991.65422046, 4)
        np.testing.assert_allclose(self.m[0].centre.value, 0.5)
        np.testing.assert_allclose(self.m[0].sigma.value, 2.08398236966)

    def test_fit_bounded_leastsq(self):
        pytest.importorskip("scipy", minversion="0.17")
        self.m[0].centre.bmin = 0.5
        # self.m[0].bounded = True
        self.m.fit(fitter="leastsq", bounded=True)
        np.testing.assert_allclose(self.m[0].A.value, 9991.65422046, 3)
        np.testing.assert_allclose(self.m[0].centre.value, 0.5)
        np.testing.assert_allclose(self.m[0].sigma.value, 2.08398236966)

    def test_fit_bounded_lbfgs(self):
        self.m[0].centre.bmin = 0.5
        # self.m[0].bounded = True
        self.m.fit(fitter="L-BFGS-B", bounded=True, grad=True)
        np.testing.assert_allclose(self.m[0].A.value, 9991.65422046, 4)
        np.testing.assert_allclose(self.m[0].centre.value, 0.5)
        np.testing.assert_allclose(self.m[0].sigma.value, 2.08398236966)

    def test_fit_bounded_bad_starting_values_mpfit(self):
        self.m[0].centre.bmin = 0.5
        self.m[0].centre.value = -1
        # self.m[0].bounded = True
        self.m.fit(fitter="mpfit", bounded=True)
        np.testing.assert_allclose(self.m[0].A.value, 9991.65422046, 4)
        np.testing.assert_allclose(self.m[0].centre.value, 0.5)
        np.testing.assert_allclose(self.m[0].sigma.value, 2.08398236966)

    def test_fit_bounded_bad_starting_values_leastsq(self):
        self.m[0].centre.bmin = 0.5
        self.m[0].centre.value = -1
        # self.m[0].bounded = True
        self.m.fit(fitter="leastsq", bounded=True)
        np.testing.assert_allclose(self.m[0].A.value, 9991.65422046, 3)
        np.testing.assert_allclose(self.m[0].centre.value, 0.5)
        np.testing.assert_allclose(self.m[0].sigma.value, 2.08398236966)

    def test_fit_bounded_bad_starting_values_lbfgs(self):
        self.m[0].centre.bmin = 0.5
        self.m[0].centre.value = -1
        # self.m[0].bounded = True
        self.m.fit(fitter="L-BFGS-B", bounded=True, grad=True)
        np.testing.assert_allclose(self.m[0].A.value, 9991.65422046, 4)
        np.testing.assert_allclose(self.m[0].centre.value, 0.5)
        np.testing.assert_allclose(self.m[0].sigma.value, 2.08398236966)

    def test_wrong_method(self):
        with pytest.raises(ValueError):
            self.m.fit(method="dummy")


class TestModelWeighted:

    def setup_method(self, method):
        np.random.seed(1)
        s = hs.signals.Signal1D(np.arange(10, 100, 0.1))
        s.metadata.set_item("Signal.Noise_properties.variance",
                            hs.signals.Signal1D(np.arange(10, 100, 0.01)))
        s.axes_manager[0].scale = 0.1
        s.axes_manager[0].offset = 10
        s.add_poissonian_noise()
        m = s.create_model()
        m.append(hs.model.components1D.Polynomial(1))
        self.m = m

    def test_fit_leastsq_binned(self):
        self.m.signal.metadata.Signal.binned = True
        self.m.fit(fitter="leastsq", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9165596693502778, 1.6628238107916631)):
            np.testing.assert_allclose(result, expected, atol=1E-5)

    def test_fit_odr_binned(self):
        self.m.signal.metadata.Signal.binned = True
        self.m.fit(fitter="odr", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9165596548961972, 1.6628247412317521)):
            np.testing.assert_allclose(result, expected, atol=1E-5)

    def test_fit_mpfit_binned(self):
        self.m.signal.metadata.Signal.binned = True
        self.m.fit(fitter="mpfit", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9165596607108739, 1.6628243846485873)):
            np.testing.assert_allclose(result, expected, atol=1E-5)

    def test_fit_neldermead_binned(self):
        self.m.signal.metadata.Signal.binned = True
        self.m.fit(
            fitter="Nelder-Mead",
            method="ls",
        )
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9137288425667442, 1.8446013472266145)):
            np.testing.assert_allclose(result, expected, atol=1E-5)

    def test_fit_leastsq_unbinned(self):
        self.m.signal.metadata.Signal.binned = False
        self.m.fit(fitter="leastsq", method="ls")
        for result, expected in zip(
                self.m[0].coefficients.value,
                (0.99165596391487121, 0.16628254242532492)):
            np.testing.assert_allclose(result, expected, atol=1E-5)

    def test_fit_odr_unbinned(self):
        self.m.signal.metadata.Signal.binned = False
        self.m.fit(fitter="odr", method="ls")
        for result, expected in zip(
                self.m[0].coefficients.value,
                (0.99165596548961943, 0.16628247412317315)):
            np.testing.assert_allclose(result, expected, atol=1E-5)

    def test_fit_mpfit_unbinned(self):
        self.m.signal.metadata.Signal.binned = False
        self.m.fit(fitter="mpfit", method="ls")
        for result, expected in zip(
                self.m[0].coefficients.value,
                (0.99165596295068958, 0.16628257462820528)):
            np.testing.assert_allclose(result, expected, atol=1E-5)

    def test_fit_neldermead_unbinned(self):
        self.m.signal.metadata.Signal.binned = False
        self.m.fit(
            fitter="Nelder-Mead",
            method="ls",
        )
        for result, expected in zip(
                self.m[0].coefficients.value,
                (0.99136169230026261, 0.18483060534056939)):
            np.testing.assert_allclose(result, expected, atol=1E-5)

    def test_chisq(self):
        self.m.signal.metadata.Signal.binned = True
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.chisq.data, 3029.16949561)

    def test_red_chisq(self):
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.red_chisq.data, 3.37700055)


class TestModelScalarVariance:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.ones(100))
        m = s.create_model()
        m.append(hs.model.components1D.Offset())
        self.s = s
        self.m = m

    def test_std1_chisq(self):
        std = 1
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.chisq.data, 78.35015229)

    def test_std10_chisq(self):
        std = 10
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.chisq.data, 78.35015229)

    def test_std1_red_chisq(self):
        std = 1
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.red_chisq.data, 0.79949135)

    def test_std10_red_chisq(self):
        std = 10
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.red_chisq.data, 0.79949135)

    def test_std1_red_chisq_in_range(self):
        std = 1
        self.m.set_signal_range(10, 50)
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        np.testing.assert_allclose(self.m.red_chisq.data, 0.86206965)


class TestModelSignalVariance:

    def setup_method(self, method):
        variance = hs.signals.Signal1D(np.arange(100, 300).reshape(
            (2, 100)))
        s = variance.deepcopy()
        np.random.seed(1)
        std = 10
        s.add_gaussian_noise(std)
        s.add_poissonian_noise()
        s.metadata.set_item("Signal.Noise_properties.variance",
                            variance + std ** 2)
        m = s.create_model()
        m.append(hs.model.components1D.Polynomial(order=1))
        self.s = s
        self.m = m

    def test_std1_red_chisq(self):
        self.m.multifit(fitter="leastsq", method="ls", show_progressbar=None)
        np.testing.assert_allclose(self.m.red_chisq.data[0],
                                   0.79693355673230915)
        np.testing.assert_allclose(self.m.red_chisq.data[1],
                                   0.91453032901427167)


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
        self.m.multifit(fetch_only_fixed=False, show_progressbar=None)
        np.testing.assert_array_almost_equal(self.m[0].r.map['values'],
                                             [3., 100.])
        np.testing.assert_array_almost_equal(self.m[0].A.map['values'],
                                             [2., 2.])

    def test_fetch_only_fixed_true(self):
        self.m.multifit(fetch_only_fixed=True, show_progressbar=None)
        np.testing.assert_array_almost_equal(self.m[0].r.map['values'],
                                             [3., 3.])
        np.testing.assert_array_almost_equal(self.m[0].A.map['values'],
                                             [2., 2.])

    def test_parameter_as_signal_values(self):
        # There are more as_signal tests in test_parameters.py
        rs = self.m[0].r.as_signal(field="values")
        np.testing.assert_allclose(rs.data, np.array([2., 100.]))
        assert not "Signal.Noise_properties.variance" in rs.metadata
        self.m.multifit(fetch_only_fixed=True, show_progressbar=None)
        rs = self.m[0].r.as_signal(field="values")
        assert "Signal.Noise_properties.variance" in rs.metadata
        assert isinstance(rs.metadata.Signal.Noise_properties.variance,
                          hs.signals.Signal1D)

    def test_bounded_snapping_mpfit(self):
        m = self.m
        m[0].A.free = True
        m.signal.data *= 2.
        m[0].A.value = 2.
        m[0].A.bmin = 3.
        m.multifit(fitter='mpfit', bounded=True, show_progressbar=None)
        np.testing.assert_array_almost_equal(self.m[0].r.map['values'],
                                             [3., 3.])
        np.testing.assert_array_almost_equal(self.m[0].A.map['values'],
                                             [4., 4.])

    def test_bounded_snapping_leastsq(self):
        m = self.m
        m[0].A.free = True
        m.signal.data *= 2.
        m[0].A.value = 2.
        m[0].A.bmin = 3.
        m.multifit(fitter='leastsq', bounded=True, show_progressbar=None)
        np.testing.assert_array_almost_equal(self.m[0].r.map['values'],
                                             [3., 3.])
        np.testing.assert_array_almost_equal(self.m[0].A.map['values'],
                                             [4., 4.])


class TestStoreCurrentValues:

    def setup_method(self, method):
        self.m = hs.signals.Signal1D(np.arange(10)).create_model()
        self.o = hs.model.components1D.Offset()
        self.m.append(self.o)

    def test_active(self):
        self.o.offset.value = 2
        self.o.offset.std = 3
        self.m.store_current_values()
        assert self.o.offset.map["values"][0] == 2
        assert self.o.offset.map["is_set"][0] == True

    def test_not_active(self):
        self.o.active = False
        self.o.offset.value = 2
        self.o.offset.std = 3
        self.m.store_current_values()
        assert self.o.offset.map["values"][0] != 2


class TestSetCurrentValuesTo:

    def setup_method(self, method):
        self.m = hs.signals.Signal1D(
            np.arange(10).reshape(2, 5)).create_model()
        self.comps = [
            hs.model.components1D.Offset(),
            hs.model.components1D.Offset()]
        self.m.extend(self.comps)

    def test_set_all(self):
        for c in self.comps:
            c.offset.value = 2
        self.m.assign_current_values_to_all()
        assert (self.comps[0].offset.map["values"] == 2).all()
        assert (self.comps[1].offset.map["values"] == 2).all()

    def test_set_1(self):
        self.comps[1].offset.value = 2
        self.m.assign_current_values_to_all([self.comps[1]])
        assert (self.comps[0].offset.map["values"] != 2).all()
        assert (self.comps[1].offset.map["values"] == 2).all()


class TestAsSignal:

    def setup_method(self, method):
        self.m = hs.signals.Signal1D(
            np.arange(20).reshape(2, 2, 5)).create_model()
        self.comps = [
            hs.model.components1D.Offset(),
            hs.model.components1D.Offset()]
        self.m.extend(self.comps)
        for c in self.comps:
            c.offset.value = 2
        self.m.assign_current_values_to_all()

    @pytest.mark.parallel
    def test_threaded_identical(self):
        # all components
        s = self.m.as_signal(show_progressbar=False, parallel=True)
        s1 = self.m.as_signal(show_progressbar=False, parallel=False)
        np.testing.assert_allclose(s1.data, s.data)

        # more complicated
        self.m[0].active_is_multidimensional = True
        self.m[0]._active_array[0] = False
        for component in [0, 1]:
            s = self.m.as_signal(component_list=[component],
                                 show_progressbar=False, parallel=True)
            s1 = self.m.as_signal(component_list=[component],
                                  show_progressbar=False, parallel=False)
            np.testing.assert_allclose(s1.data, s.data)

    @pytest.mark.parametrize('parallel', [pytest.mark.parallel(True), False])
    def test_all_components_simple(self, parallel):
        s = self.m.as_signal(show_progressbar=False, parallel=parallel)
        assert np.all(s.data == 4.)

    @pytest.mark.parametrize('parallel', [pytest.mark.parallel(True), False])
    def test_one_component_simple(self, parallel):
        s = self.m.as_signal(component_list=[0], show_progressbar=False,
                             parallel=parallel)
        assert np.all(s.data == 2.)
        assert self.m[1].active

    @pytest.mark.parametrize('parallel', [pytest.mark.parallel(True), False])
    def test_all_components_multidim(self, parallel):
        self.m[0].active_is_multidimensional = True

        s = self.m.as_signal(show_progressbar=False, parallel=parallel)
        assert np.all(s.data == 4.)

        self.m[0]._active_array[0] = False
        s = self.m.as_signal(show_progressbar=False, parallel=parallel)
        np.testing.assert_array_equal(
            s.data, np.array([np.ones((2, 5)) * 2, np.ones((2, 5)) * 4]))
        assert self.m[0].active_is_multidimensional

    @pytest.mark.parametrize('parallel', [pytest.mark.parallel(True), False])
    def test_one_component_multidim(self, parallel):
        self.m[0].active_is_multidimensional = True

        s = self.m.as_signal(component_list=[0], show_progressbar=False,
                             parallel=parallel)
        assert np.all(s.data == 2.)
        assert self.m[1].active
        assert not self.m[1].active_is_multidimensional

        s = self.m.as_signal(component_list=[1], show_progressbar=False,
                             parallel=parallel)
        np.testing.assert_equal(s.data, 2.)
        assert self.m[0].active_is_multidimensional

        self.m[0]._active_array[0] = False
        s = self.m.as_signal(component_list=[1], show_progressbar=False,
                             parallel=parallel)
        assert np.all(s.data == 2.)

        s = self.m.as_signal(component_list=[0], show_progressbar=False,
                             parallel=parallel)
        np.testing.assert_array_equal(s.data, np.array([np.zeros((2, 5)),
                                                        np.ones((2, 5)) * 2]))


class TestCreateModel:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.asarray([0, ]))
        self.im = hs.signals.Signal2D(np.ones([1, 1, ]))

    def test_create_model(self):
        from hyperspy.models.model1d import Model1D
        from hyperspy.models.model2d import Model2D
        assert isinstance(self.s.create_model(), Model1D)
        assert isinstance(self.im.create_model(), Model2D)


class TestAdjustPosition:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.random.rand(10, 10, 20))
        self.m = self.s.create_model()

    def test_enable_adjust_position(self):
        self.m.append(hs.model.components1D.Gaussian())
        self.m.enable_adjust_position()
        assert len(self.m._position_widgets) == 1
        # Check that both line and label was added
        assert len(list(self.m._position_widgets.values())[0]) == 2

    def test_disable_adjust_position(self):
        self.m.append(hs.model.components1D.Gaussian())
        self.m.enable_adjust_position()
        self.m.disable_adjust_position()
        assert len(self.m._position_widgets) == 0

    def test_enable_all(self):
        self.m.append(hs.model.components1D.Gaussian())
        self.m.enable_adjust_position()
        self.m.append(hs.model.components1D.Gaussian())
        assert len(self.m._position_widgets) == 2

    def test_enable_all_zero_start(self):
        self.m.enable_adjust_position()
        self.m.append(hs.model.components1D.Gaussian())
        assert len(self.m._position_widgets) == 1

    def test_manual_close(self):
        self.m.append(hs.model.components1D.Gaussian())
        self.m.append(hs.model.components1D.Gaussian())
        self.m.enable_adjust_position()
        list(self.m._position_widgets.values())[0][0].close()
        assert len(self.m._position_widgets) == 2
        assert len(list(self.m._position_widgets.values())[0]) == 1
        list(self.m._position_widgets.values())[0][0].close()
        assert len(self.m._position_widgets) == 1
        assert len(list(self.m._position_widgets.values())[0]) == 2
        self.m.disable_adjust_position()
        assert len(self.m._position_widgets) == 0
