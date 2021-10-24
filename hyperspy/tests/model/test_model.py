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

from unittest import mock

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.misc.test_utils import ignore_warning
from hyperspy.misc.utils import slugify


class TestModelJacobians:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros(1))
        m = s.create_model()
        self.low_loss = 7.0
        self.weights = 0.3
        m.axis.axis = np.array([1, 0])
        m.channel_switches = np.array([0, 1], dtype=bool)
        m.append(hs.model.components1D.Gaussian())
        m[0].A.value = 1
        m[0].centre.value = 2.0
        m[0].sigma.twin = m[0].centre
        m._low_loss = mock.MagicMock()
        m.low_loss.return_value = self.low_loss
        self.model = m
        m.convolution_axis = np.zeros(2)

    def test_jacobian_not_convolved(self):
        m = self.model
        m.convolved = False
        jac = m._jacobian((1, 2, 3), None, weights=self.weights)
        np.testing.assert_array_almost_equal(
            jac.squeeze(),
            self.weights
            * np.array([m[0].A.grad(0), m[0].sigma.grad(0) + m[0].centre.grad(0)]),
        )
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
        np.testing.assert_array_almost_equal(
            jac.squeeze(),
            self.weights
            * np.array(
                [
                    m[0].A.grad(0),
                    m[0].sigma.grad(0) + m[0].centre.grad(0),
                    m[1].A.grad(0) * self.low_loss,
                    m[1].centre.grad(0) * self.low_loss,
                    m[1].sigma.grad(0) * self.low_loss,
                ]
            ),
        )
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
        m.convolution_axis = np.array([0.0])

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
            res, np.array([np.nan, 0.5, 0.25, np.nan, np.nan])
        )
        assert m.__call__.called
        assert m.__call__.call_args[1] == {"non_convolved": False, "onlyactive": True}
        assert not m.fetch_stored_values.called

    def test_model2plot_other_am(self):
        m = self.model
        res = m._model2plot(m.axes_manager.deepcopy(), out_of_range2nans=False)
        np.testing.assert_array_equal(res, np.array([0.5, 0.25]))
        assert m.__call__.called
        assert m.__call__.call_args[1] == {"non_convolved": False, "onlyactive": True}
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

        with pytest.warns(
            VisibleDeprecationWarning,
            match=r".* has been deprecated and will be made private",
        ):
            m.set_boundaries()

        assert m.free_parameters_boundaries == [(0.1, 0.11), (0.2, 0.21), (0.3, 0.31)]

    def test_setting_mpfit_parameters_info(self):
        m = self.model
        m[0].A.bmax = None
        m[0].centre.bmin = None
        m[0].centre.bmax = 0.31
        m.append(hs.model.components1D.Gaussian())
        m[-1].active = False

        with pytest.warns(
            VisibleDeprecationWarning,
            match=r".* has been deprecated and will be made private",
        ):
            m.set_mpfit_parameters_info()

        assert m.mpfit_parinfo == [
            {"limited": [True, False], "limits": [0.1, 0]},
            {"limited": [False, True], "limits": [0, 0.31]},
            {"limited": [False, True], "limits": [0, 0.31]},
        ]


class TestModel1D:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.empty(1))
        m = s.create_model()
        self.model = m

    def test_errfunc(self):
        m = self.model
        m._model_function = mock.MagicMock()
        m._model_function.return_value = 3.0
        np.testing.assert_equal(m._errfunc(None, 1.0, None), 2.0)
        np.testing.assert_equal(m._errfunc(None, 1.0, 0.3), 0.6)

    def test_errfunc_sq(self):
        m = self.model
        m._model_function = mock.MagicMock()
        m._model_function.return_value = 3.0 * np.ones(2)
        np.testing.assert_equal(m._errfunc_sq(None, np.ones(2), None), 8.0)
        np.testing.assert_equal(m._errfunc_sq(None, np.ones(2), 0.3), 0.72)

    def test_gradient_ls(self):
        m = self.model
        m._errfunc = mock.MagicMock()
        m._errfunc.return_value = 0.1
        m._jacobian = mock.MagicMock()
        m._jacobian.return_value = np.ones((1, 2)) * 7.0
        np.testing.assert_allclose(m._gradient_ls(None, None), 2.8)

    def test_gradient_ml(self):
        m = self.model
        m._model_function = mock.MagicMock()
        m._model_function.return_value = 3.0 * np.ones(2)
        m._jacobian = mock.MagicMock()
        m._jacobian.return_value = np.ones((1, 2)) * 7.0
        np.testing.assert_allclose(m._gradient_ml(None, 1.2), 8.4)

    def test_gradient_huber(self):
        m = self.model
        m._errfunc = mock.MagicMock()
        m._errfunc.return_value = 0.1
        m._jacobian = mock.MagicMock()
        m._jacobian.return_value = np.ones((1, 2)) * 7.0
        np.testing.assert_allclose(m._gradient_huber(None, None), 1.4)

    def test_model_function(self):
        m = self.model
        m.append(hs.model.components1D.Gaussian())
        m[0].A.value = 1.3
        m[0].centre.value = 0.003
        m[0].sigma.value = 0.1
        param = (100, 0.1, 0.2)
        np.testing.assert_array_almost_equal(176.03266338, m._model_function(param))
        assert m[0].A.value == 100
        assert m[0].centre.value == 0.1
        assert m[0].sigma.value == 0.2

    def test_append_existing_component(self):
        g = hs.model.components1D.Gaussian()
        m = self.model
        m.append(g)
        with pytest.raises(ValueError, match="Component already in model"):
            m.append(g)

    def test_append_component(self):
        g = hs.model.components1D.Gaussian()
        m = self.model
        m.append(g)
        assert g in m
        assert g.model is m
        assert g._axes_manager is m.axes_manager
        assert all([hasattr(p, "map") for p in g.parameters])

    def test_calculating_convolution_axis(self):
        m = self.model
        # setup
        m.axis.offset = 10
        m.axis.size = 10
        ll_axis = mock.MagicMock()
        ll_axis.size = 7
        ll_axis.value2index.return_value = 3
        m._low_loss = mock.MagicMock()
        m.low_loss.axes_manager.signal_axes = [
            ll_axis,
        ]

        # calculation
        m.set_convolution_axis()

        # tests
        np.testing.assert_array_equal(m.convolution_axis, np.arange(7, 23))
        np.testing.assert_equal(ll_axis.value2index.call_args[0][0], 0)

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
            hs.model.components1D.Gaussian(),
        ]
        m.extend(gs)
        assert m["Gaussian"] is gs[0]
        assert m["Gaussian_0"] is gs[1]
        assert m["Gaussian_1"] is gs[2]

    def test_several_component_with_same_name(self):
        m = self.model
        gs = [
            hs.model.components1D.Gaussian(),
            hs.model.components1D.Gaussian(),
            hs.model.components1D.Gaussian(),
        ]
        m.extend(gs)
        m[0]._name = "hs.model.components1D.Gaussian"
        m[1]._name = "hs.model.components1D.Gaussian"
        m[2]._name = "hs.model.components1D.Gaussian"

        with pytest.raises(ValueError, match=r"Component name .* not found in model"):
            m["Gaussian"]

    def test_no_component_with_that_name(self):
        m = self.model
        with pytest.raises(ValueError, match=r"Component name .* not found in model"):
            m["Voigt"]

    def test_component_already_in_model(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        with pytest.raises(ValueError, match="Component already in model"):
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
        g3.A.twin = g1.A
        g1.sigma.twin = g2.sigma
        m.extend([g1, g2, g3])
        del m[:2]
        assert g1 not in m
        assert g2 not in m
        assert g3 in m
        assert not g1.sigma.twin
        assert not g1.A._twins

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
        with pytest.raises(ValueError, match="Not a component or component id"):
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

        with pytest.raises(AttributeError, match="object has no attribute 'Gaussian'"):
            getattr(m.components, "Gaussian")

    def test_components_class_change_invalid_name(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        g1.name = "1, Test This!"
        assert getattr(m.components, slugify(g1.name, valid_variable_name=True)) is g1

    def test_components_class_change_name_del_default2(self):
        m = self.model
        g1 = hs.model.components1D.Gaussian()
        m.append(g1)
        invalid_name = "1, Test This!"
        g1.name = invalid_name
        g1.name = "test"
        with pytest.raises(AttributeError, match=r"object has no attribute .*"):
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
        p = hs.model.components1D.Polynomial(3, legacy=False)
        m.append(p)

        g1.A.value = 3.0
        g1.centre.bmin = 300.0
        g1.centre.value = 1.0
        g1.sigma.bmax = 15.0
        g1.sigma.value = 30

        g2.A.value = 1
        g2.A.bmin = 0.0
        g2.A.bmax = 3.0
        g2.centre.value = 0
        g2.centre.bmin = 1
        g2.centre.bmax = 3.0
        g2.sigma.value = 4
        g2.sigma.bmin = 1
        g2.sigma.bmax = 3.0

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
        p.a0.value = 1
        p.a1.value = 2
        p.a2.value = 3
        p.a3.value = 4
        p.a0.bmin = 2
        p.a1.bmin = 2
        p.a2.bmin = 2
        p.a3.bmin = 2
        p.a0.bmax = 3
        p.a1.bmax = 3
        p.a2.bmax = 3
        p.a3.bmax = 3

        m.ensure_parameters_in_bounds()
        np.testing.assert_allclose(g1.A.value, 3.0)
        np.testing.assert_allclose(g2.A.value, 1.0)
        np.testing.assert_allclose(g3.A.value, -3.0)
        np.testing.assert_allclose(g4.A.value, 300.0)

        np.testing.assert_allclose(g1.centre.value, 300.0)
        np.testing.assert_allclose(g2.centre.value, 1.0)
        np.testing.assert_allclose(g3.centre.value, 15.0)
        np.testing.assert_allclose(g4.centre.value, 0)

        np.testing.assert_allclose(g1.sigma.value, 15.0)
        np.testing.assert_allclose(g2.sigma.value, 3.0)
        np.testing.assert_allclose(g3.sigma.value, 0.0)
        np.testing.assert_allclose(g4.sigma.value, 1)

        np.testing.assert_almost_equal(p.a0.value, 2)
        np.testing.assert_almost_equal(p.a1.value, 2)
        np.testing.assert_almost_equal(p.a2.value, 3)
        np.testing.assert_almost_equal(p.a3.value, 3)


class TestModelPrintCurrentValues:
    def setup_method(self, method):
        np.random.seed(1)
        s = hs.signals.Signal1D(np.arange(10, 100, 0.1))
        s.axes_manager[0].scale = 0.1
        s.axes_manager[0].offset = 10
        m = s.create_model()
        with ignore_warning(message="The API of the `Polynomial` component"):
            m.append(hs.model.components1D.Polynomial(1))
        m.append(hs.model.components1D.Offset())
        self.s = s
        self.m = m

    @pytest.mark.parametrize("only_free", [True, False])
    @pytest.mark.parametrize("skip_multi", [True, False])
    def test_print_current_values(self, only_free, skip_multi):
        self.m.print_current_values(only_free, skip_multi)

    def test_print_current_values_component_list(self):
        self.m.print_current_values(component_list=list(self.m))


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
        assert self.o.offset.map["is_set"][0]

    def test_not_active(self):
        self.o.active = False
        self.o.offset.value = 2
        self.o.offset.std = 3
        self.m.store_current_values()
        assert self.o.offset.map["values"][0] != 2


class TestSetCurrentValuesTo:
    def setup_method(self, method):
        self.m = hs.signals.Signal1D(np.arange(10).reshape(2, 5)).create_model()
        self.comps = [hs.model.components1D.Offset(), hs.model.components1D.Offset()]
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


def test_fetch_values_from_arrays():
    m = hs.signals.Signal1D(np.arange(10)).create_model()
    gaus = hs.model.components1D.Gaussian(A=100, sigma=10, centre=3)
    m.append(gaus)
    values = np.array([1.2, 3.4, 5.6])
    stds = values - 1
    m.fetch_values_from_array(values, array_std=stds)
    parameters = sorted(gaus.free_parameters, key=lambda x: x.name)
    for v, s, p in zip(values, stds, parameters):
        assert p.value == v
        assert p.std == s


class TestAsSignal:
    def setup_method(self, method):
        self.m = hs.signals.Signal1D(np.arange(20).reshape(2, 2, 5)).create_model()
        self.comps = [hs.model.components1D.Offset(), hs.model.components1D.Offset()]
        self.m.extend(self.comps)
        for c in self.comps:
            c.offset.value = 2
        self.m.assign_current_values_to_all()

    def test_all_components_simple(self):
        s = self.m.as_signal()
        assert np.all(s.data == 4.0)

    def test_one_component_simple(self):
        s = self.m.as_signal(component_list=[0])
        assert np.all(s.data == 2.0)
        assert self.m[1].active

    def test_all_components_multidim(self):
        self.m[0].active_is_multidimensional = True

        s = self.m.as_signal()
        assert np.all(s.data == 4.0)

        self.m[0]._active_array[0] = False
        s = self.m.as_signal()
        np.testing.assert_array_equal(
            s.data, np.array([np.ones((2, 5)) * 2, np.ones((2, 5)) * 4])
        )
        assert self.m[0].active_is_multidimensional

    def test_one_component_multidim(self):
        self.m[0].active_is_multidimensional = True

        s = self.m.as_signal(component_list=[0])
        assert np.all(s.data == 2.0)
        assert self.m[1].active
        assert not self.m[1].active_is_multidimensional

        s = self.m.as_signal(component_list=[1])
        np.testing.assert_equal(s.data, 2.0)
        assert self.m[0].active_is_multidimensional

        self.m[0]._active_array[0] = False
        s = self.m.as_signal(component_list=[1])
        assert np.all(s.data == 2.0)

        s = self.m.as_signal(component_list=[0])
        np.testing.assert_array_equal(
            s.data, np.array([np.zeros((2, 5)), np.ones((2, 5)) * 2])
        )

    @pytest.mark.parametrize("kw", [{"parallel": True}, {"max_workers": 1}])
    def test_warnings(self, kw):
        with pytest.warns(
            VisibleDeprecationWarning, match=r".* has been deprecated",
        ):
            _ = self.m.as_signal(**kw)

    def test_out_of_range_to_nan(self):
        index = 2
        self.m.channel_switches[:index] = False
        s1 = self.m.as_signal(component_list=[0], out_of_range_to_nan=True)

        s2 = self.m.as_signal(component_list=[0], out_of_range_to_nan=False)

        np.testing.assert_allclose(
            self.m.channel_switches, [False, False, True, True, True]
        )

        np.testing.assert_allclose(s2.data, np.ones_like(s2) * 2)
        np.testing.assert_allclose(s1.isig[index:], s2.isig[index:])
        np.testing.assert_allclose(
            s1.isig[:index], np.ones_like(s1.isig[:index].data) * np.nan
        )
        np.testing.assert_allclose(
            s1.isig[index:], np.ones_like(s1.isig[index:].data) * 2
        )

    def test_out_argument(self):
        out = self.m.as_signal()
        out.data.fill(0)
        s = self.m.as_signal(out=out)
        assert np.all(s.data == 4.0)


@lazifyTestClass
class TestCreateModel:
    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.asarray([0]))
        self.im = hs.signals.Signal2D(np.ones([1, 1]))

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


def test_deprecated_private_functions():
    s = hs.signals.Signal1D(np.zeros(1))
    m = s.create_model()

    with pytest.warns(VisibleDeprecationWarning, match=r".* has been deprecated"):
        m.set_boundaries()

    with pytest.warns(VisibleDeprecationWarning, match=r".* has been deprecated"):
        m.set_mpfit_parameters_info()


class TestSignalRange:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.random.rand(10, 10, 20))
        s.axes_manager[-1].offset = 100
        m = s.create_model()
        self.s = s
        self.m = m

    def test_parse_value(self):
        m = self.m
        assert m._parse_signal_range_values(105, 110) == (5, 10)
        with pytest.raises(ValueError):
            m._parse_signal_range_values(89, 85)

    def test_parse_value_negative_scale(self):
        m = self.m
        s = self.s
        s.axes_manager[-1].scale = -1
        assert m._parse_signal_range_values(89, 85) == (11, 15)
        with pytest.raises(ValueError):
            m._parse_signal_range_values(85, 89)
        assert m._parse_signal_range_values(89, 20) == (11, 19)

    def test_parse_roi(self):
        m = self.m
        roi = hs.roi.SpanROI(105, 110)
        assert m._parse_signal_range_values(roi) == (5, 10)
