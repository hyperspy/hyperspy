# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.


import numpy as np

import nose.tools as nt
from hyperspy._signals.signal1d import Signal1D
from hyperspy.component import Parameter, Component
from hyperspy.components1d import Gaussian, Lorentzian, ScalableFixedPattern


def remove_empty_numpy_strings(dic):
    for k, v in dic.items():
        if isinstance(v, dict):
            remove_empty_numpy_strings(v)
        elif isinstance(v, list):
            for vv in v:
                if isinstance(vv, dict):
                    remove_empty_numpy_strings(vv)
                elif isinstance(vv, np.string_) and len(vv) == 0:
                    vv = ''
        elif isinstance(v, np.string_) and len(v) == 0:
            del dic[k]
            dic[k] = ''


class DummyAxesManager:
    navigation_shape = [1, ]
    navigation_size = 2
    indices = ()

    @property
    def _navigation_shape_in_array(self):
        return self.navigation_shape[::-1]


class TestParameterDictionary:

    def setUp(self):
        self.par = Parameter()
        self.par.name = 'asd'
        self.par._id_name = 'newone'

        def ft(x):
            return x * x

        def fit(x):
            return x * x + 1
        self.par.twin_function = ft
        self.par.twin_inverse_function = fit
        self.par._axes_manager = DummyAxesManager()
        self.par._create_array()
        self.par.value = 1
        self.par.std = 0.1
        self.par.store_current_value_in_array()
        self.par.ext_bounded = False
        self.par.ext_force_positive = False

    def test_to_dictionary(self):
        d = self.par.as_dictionary()

        nt.assert_equal(d['name'], self.par.name)
        nt.assert_equal(d['_id_name'], self.par._id_name)
        np.testing.assert_equal(d['map']['values'][0], 1)
        np.testing.assert_equal(d['map']['std'][0], 0.1)
        nt.assert_true(d['map']['is_set'][0])
        np.testing.assert_equal(d['value'], self.par.value)
        np.testing.assert_equal(d['std'], self.par.std)
        nt.assert_is(d['free'], self.par.free)
        nt.assert_equal(d['self'], id(self.par))
        np.testing.assert_equal(d['_bounds'], self.par._bounds)
        nt.assert_is(d['ext_bounded'], self.par.ext_bounded)
        nt.assert_is(
            d['ext_force_positive'], self.par.ext_force_positive)

    def test_load_dictionary(self):
        d = self.par.as_dictionary()
        p = Parameter()
        p._id_name = 'newone'
        _id = p._load_dictionary(d)

        nt.assert_equal(_id, id(self.par))
        nt.assert_equal(p.name, self.par.name)
        nt.assert_equal(p._id_name, self.par._id_name)
        np.testing.assert_equal(p.map['values'][0], 1)
        np.testing.assert_equal(p.map['std'][0], 0.1)
        nt.assert_true(p.map['is_set'][0])
        np.testing.assert_equal(p.value, self.par.value)
        np.testing.assert_equal(p.std, self.par.std)
        np.testing.assert_equal(p.free, self.par.free)
        np.testing.assert_equal(p._bounds, self.par._bounds)

        rn = np.random.random()
        np.testing.assert_equal(
            p.twin_function(rn),
            self.par.twin_function(rn))
        np.testing.assert_equal(
            p.twin_inverse_function(rn),
            self.par.twin_inverse_function(rn))

    @nt.raises(ValueError)
    def test_invalid_name(self):
        d = self.par.as_dictionary()
        d['_id_name'] = 'otherone'
        p = Parameter()
        p._id_name = 'newone'
        _id = p._load_dictionary(d)


class TestComponentDictionary:

    def setUp(self):
        self.parameter_names = ['par1', 'par2']
        self.comp = Component(self.parameter_names)
        self.comp.name = 'newname!'
        self.comp._id_name = 'dummy names yay!'
        self.comp._axes_manager = DummyAxesManager()
        self.comp._create_arrays()
        self.comp.par1.value = 2.
        self.comp.par2.value = 5.
        self.comp.par1.std = 0.2
        self.comp.par2.std = 0.5
        self.comp.store_current_parameters_in_map()

    def test_to_dictionary(self):
        d = self.comp.as_dictionary()
        c = self.comp

        nt.assert_equal(c.name, d['name'])
        nt.assert_equal(c._id_name, d['_id_name'])
        nt.assert_false(d['active_is_multidimensional'])
        nt.assert_true(d['active'])
        nt.assert_is_none(d['_active_array'])
        for ip, p in enumerate(c.parameters):
            nt.assert_dict_equal(p.as_dictionary(), d['parameters'][ip])

        c.active_is_multidimensional = True
        d1 = c.as_dictionary()
        nt.assert_true(d1['active_is_multidimensional'])
        np.testing.assert_array_equal(d1['_active_array'], c._active_array)

    def test_load_dictionary(self):
        c = self.comp
        d = c.as_dictionary(True)
        n = Component(self.parameter_names)

        n._id_name = 'dummy names yay!'
        _ = n._load_dictionary(d)
        nt.assert_equal(c.name, n.name)
        nt.assert_equal(c.active, n.active)
        nt.assert_equal(
            c.active_is_multidimensional,
            n.active_is_multidimensional)

        for pn, pc in zip(n.parameters, c.parameters):
            rn = np.random.random()
            nt.assert_equal(pn.twin_function(rn), pc.twin_function(rn))
            nt.assert_equal(
                pn.twin_inverse_function(rn),
                pc.twin_inverse_function(rn))
            dn = pn.as_dictionary()
            del dn['self']
            del dn['twin_function']
            del dn['twin_inverse_function']
            dc = pc.as_dictionary()
            del dc['self']
            del dc['twin_function']
            del dc['twin_inverse_function']
            print(list(dn.keys()))
            print(list(dc.keys()))
            nt.assert_dict_equal(dn, dc)

    @nt.raises(ValueError)
    def test_invalid_component_name(self):
        c = self.comp
        d = c.as_dictionary()
        n = Component(self.parameter_names)
        id_dict = n._load_dictionary(d)

    @nt.raises(ValueError)
    def test_invalid_parameter_name(self):
        c = self.comp
        d = c.as_dictionary()
        n = Component([a + 's' for a in self.parameter_names])
        n._id_name = 'dummy names yay!'
        id_dict = n._load_dictionary(d)


class TestModelDictionary:

    def setUp(self):
        s = Signal1D(np.array([1.0, 2, 4, 7, 12, 7, 4, 2, 1]))
        m = s.create_model()
        m.low_loss = (s + 3.0).deepcopy()
        self.model = m
        self.s = s

        m.append(Gaussian())
        m.append(Gaussian())
        m.append(ScalableFixedPattern(s * 0.3))
        m[0].A.twin = m[1].A
        m.fit()

    def test_to_dictionary(self):
        m = self.model
        d = m.as_dictionary()

        print(d['low_loss'])
        np.testing.assert_almost_equal(m.low_loss.data, d['low_loss']['data'])
        np.testing.assert_almost_equal(m.chisq.data, d['chisq.data'])
        np.testing.assert_almost_equal(m.dof.data, d['dof.data'])
        np.testing.assert_equal(
            d['free_parameters_boundaries'],
            m.free_parameters_boundaries)
        nt.assert_is(d['convolved'], m.convolved)

        for num, c in enumerate(m):
            tmp = c.as_dictionary()
            remove_empty_numpy_strings(tmp)
            nt.assert_equal(d['components'][num]['name'], tmp['name'])
            nt.assert_equal(d['components'][num]['_id_name'], tmp['_id_name'])
        np.testing.assert_equal(d['components'][-1]['signal1D'],
                                (m.signal * 0.3)._to_dictionary())

    def test_load_dictionary(self):
        d = self.model.as_dictionary()
        mn = self.s.create_model()
        mn.append(Lorentzian())
        mn._load_dictionary(d)
        mo = self.model

        # nt.assert_true(np.allclose(mo.signal1D.data, mn.signal1D.data))
        np.testing.assert_allclose(mo.chisq.data, mn.chisq.data)
        np.testing.assert_allclose(mo.dof.data, mn.dof.data)

        np.testing.assert_allclose(mn.low_loss.data, mo.low_loss.data)

        np.testing.assert_equal(
            mn.free_parameters_boundaries,
            mo.free_parameters_boundaries)
        nt.assert_is(mn.convolved, mo.convolved)
        for i in range(len(mn)):
            nt.assert_equal(mn[i]._id_name, mo[i]._id_name)
            for po, pn in zip(mo[i].parameters, mn[i].parameters):
                np.testing.assert_allclose(po.map['values'], pn.map['values'])
                np.testing.assert_allclose(po.map['is_set'], pn.map['is_set'])

        nt.assert_is(mn[0].A.twin, mn[1].A)
