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
import pytest

import dill
import copy
import hyperspy.api as hs
from hyperspy.samfire_utils.samfire_kernel import multi_kernel
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.samfire_utils.samfire_worker import create_worker


class Mock_queue(object):

    def __init__(self):
        self.var = []

    def put(self, value):
        self.var.append(value)

np.random.seed(123)


def generate_test_model():

    # import hyperspy.api as hs
    from hyperspy.signals import Signal1D
    from hyperspy.components1d import (Gaussian, Lorentzian)
    import numpy as np
    from scipy.ndimage import gaussian_filter
    total = None
# blurs = [0., 0.5, 1., 2.,5.]
    blurs = [1.5]
    radius = 5
    domain = 15
# do circle/domain
    cent = (domain // 2, domain // 2)
    y, x = np.ogrid[-cent[0]:domain - cent[0], -cent[1]:domain - cent[1]]
    mask = x * x + y * y <= radius * radius
    lor_map = None
    for blur in blurs:

        s = Signal1D(np.ones((domain, domain, 1024)))
        cent = tuple([int(0.5 * i) for i in s.data.shape[:-1]])
        m0 = s.create_model()

        gs01 = Lorentzian()
        m0.append(gs01)
        gs01.gamma.map['values'][:] = 50
        gs01.gamma.map['is_set'][:] = True
        gs01.centre.map['values'][:] = 300
        gs01.centre.map['values'][mask] = 400
        gs01.centre.map['values'] = gaussian_filter(
            gs01.centre.map['values'],
            blur)
        gs01.centre.map['is_set'][:] = True
        gs01.A.map['values'][:] = 100 * \
            np.random.random((domain, domain)) + 300000
        gs01.A.map['values'][mask] *= 0.75
        gs01.A.map['values'] = gaussian_filter(gs01.A.map['values'], blur)
        gs01.A.map['is_set'][:] = True

        gs02 = Gaussian()
        m0.append(gs02)
        gs02.sigma.map['values'][:] = 15
        gs02.sigma.map['is_set'][:] = True
        gs02.centre.map['values'][:] = 400
        gs02.centre.map['values'][mask] = 300
        gs02.centre.map['values'] = gaussian_filter(
            gs02.centre.map['values'],
            blur)
        gs02.centre.map['is_set'][:] = True
        gs02.A.map['values'][:] = 50000
        gs02.A.map['is_set'][:] = True

        gs03 = Lorentzian()
        m0.append(gs03)
        gs03.gamma.map['values'][:] = 20
        gs03.gamma.map['is_set'][:] = True
        gs03.centre.map['values'][:] = 100
        gs03.centre.map['values'][mask] = 900
        gs03.centre.map['is_set'][:] = True
        gs03.A.map['values'][:] = 100 * \
            np.random.random((domain, domain)) + 50000
        gs03.A.map['values'][mask] *= 0.
        gs03.A.map['is_set'][:] = True

        s11 = m0.as_signal(show_progressbar=False, parallel=False)
        if total is None:
            total = s11.data.copy()
            lor_map = gs01.centre.map['values'].copy()
        else:
            total = np.concatenate((total, s11.data), axis=1)
            lor_map = np.concatenate(
                (lor_map, gs01.centre.map['values'].copy()), axis=1)

    s = Signal1D(total)
    s.add_poissonian_noise()
    s.data += 0.1
    s.estimate_poissonian_noise_variance()

    m = s.inav[:, :7].create_model()
    g = Gaussian()
    l1 = Lorentzian()
    l2 = Lorentzian()
    g.sigma.value = 50
    g.centre.value = 400
    g.A.value = 50000
    l1.gamma.value = 40
    l1.centre.value = 300
    l1.A.value = 300000
    l2.gamma.value = 15
    l2.centre.value = 100
    l2.A.value = 50000
    l2.centre.bmin = 0
    l2.centre.bmax = 200
    l2.A.bmin = 30000
    l2.A.bmax = 100000
    l2.gamma.bmin = 0
    l2.gamma.bmax = 60
    m.extend([g, l1, l2])
    m.assign_current_values_to_all()
    l2.active_is_multidimensional = True
    return m, gs01, gs02, gs03

class TestSamfireEmpty:

    def setup_method(self, method):
        self.shape = (7, 15)
        s = hs.signals.Signal1D(np.ones(self.shape + (1024,)) + 3.)
        s.estimate_poissonian_noise_variance()
        m = s.create_model()
        m.append(hs.model.components1D.Gaussian())
        m.append(hs.model.components1D.Lorentzian())
        m.append(hs.model.components1D.Lorentzian())
        self.model = m

    @pytest.mark.parallel
    def test_setup(self):
        m = self.model
        samf = m.create_samfire(workers=1, setup=False)
        assert samf.metadata._gt_dump is None
        assert samf.pool is None
        samf._setup(ipyparallel=False)
        assert samf.metadata._gt_dump is not None
        assert samf.pool is not None

    def test_samfire_init_marker(self):
        m = self.model
        samf = m.create_samfire(workers=1, setup=False)
        np.testing.assert_array_almost_equal(samf.metadata.marker,
                                             np.zeros(self.shape))

    def test_samfire_init_model(self):
        m = self.model
        samf = m.create_samfire(workers=1, setup=False)
        assert samf.model is m

    def test_samfire_init_metadata(self):
        m = self.model
        samf = m.create_samfire(workers=1, setup=False)
        assert isinstance(samf.metadata, DictionaryTreeBrowser)

    def test_samfire_init_strategy_list(self):
        from hyperspy.samfire import StrategyList
        m = self.model
        samf = m.create_samfire(workers=1, setup=False)
        assert isinstance(samf.strategies, StrategyList)

    def test_samfire_init_strategies(self):
        m = self.model
        samf = m.create_samfire(workers=1, setup=False)
        from hyperspy.samfire_utils.local_strategies import ReducedChiSquaredStrategy
        from hyperspy.samfire_utils.global_strategies import HistogramStrategy
        assert isinstance(samf.strategies[0],
                          ReducedChiSquaredStrategy)
        assert isinstance(samf.strategies[1], HistogramStrategy)

    def test_samfire_init_fig(self):
        m = self.model
        samf = m.create_samfire(workers=1, setup=False)
        assert samf._figure is None

    def test_samfire_init_default(self):
        m = self.model
        from multiprocessing import cpu_count
        samf = m.create_samfire(setup=False)
        assert samf._workers == cpu_count() - 1
        assert np.allclose(samf.metadata.marker, np.zeros(self.shape))

    def test_optional_components(self):
        m = self.model
        m[-1].active_is_multidimensional = False

        samf = m.create_samfire(setup=False)
        samf.optional_components = [m[0], 1]
        samf._enable_optional_components()
        assert m[0].active_is_multidimensional
        assert m[1].active_is_multidimensional
        assert np.all([isinstance(a, int)
                       for a in samf.optional_components])
        np.testing.assert_equal(samf.optional_components, [0, 1])

    def test_swap_dict_and_model(self):
        m = self.model
        for i in range(len(m)):
            for ip, p in enumerate(m[i].parameters):
                p.map['values'][0, 0] = 3.0 + i + ip
                p.map['std'][0, 0] = 2.44 + i + ip
                p.map['is_set'][0, 0] = True
        m[1].active_is_multidimensional = True
        m[1]._active_array[0, 0] = False
        assert m[1]._active_array[1, 0]
        m.chisq.data[0, 0] = 1200.
        m.dof.data[0, 0] = 1.

        small_m = m.inav[0, 0]
        d = {'chisq.data': np.array(small_m.chisq.data[0]),
             'dof.data': np.array(small_m.dof.data[0]),
             'components': {component.name: {parameter.name: parameter.map for
                                             parameter in component.parameters}
                            for component in small_m if component.active}
             }

        d = copy.deepcopy(d)
        samf = m.create_samfire(setup=False)
        samf._swap_dict_and_model((1, 0), d)
        assert m.chisq.data[1, 0] == 1200.
        assert m.dof.data[1, 0] == 1.

        assert d['dof.data'] == 0.
        assert np.isnan(d['chisq.data'])

        assert np.all(~m[1]._active_array[:2, 0])
        for c in m:
            if c.active:
                for p in c.parameters:
                    assert (
                        p.map['values'][
                            0, 0] == p.map['values'][
                            1, 0])
                    assert p.map['std'][0, 0] == p.map['std'][1, 0]
                    assert (
                        p.map['is_set'][
                            0, 0] == p.map['is_set'][
                            1, 0])

    def test_next_pixels(self):
        m = self.model
        samf = m.create_samfire(setup=False)
        ans = samf._next_pixels(3)
        assert len(ans) == 0
        ind_list = [(1, 2), (0, 1), (3, 3), (4, 6)]
        for ind in ind_list:
            samf.metadata.marker[ind] += 2.
        ans = samf._next_pixels(10)
        assert len(ans) == 4
        for ind in ans:
            assert ind in ind_list
        for n, ind in enumerate(ind_list):
            samf.metadata.marker[ind] += n
        ans = samf._next_pixels(10)
        assert ans == [(4, 6), ]

    def test_change_strategy(self):
        m = self.model
        samf = m.create_samfire(setup=False)
        from hyperspy.samfire_utils.local_strategies import ReducedChiSquaredStrategy
        from hyperspy.samfire_utils.global_strategies import HistogramStrategy

        ind = (0, 0)
        samf.metadata.marker[ind] = -2
        samf.strategies.append(ReducedChiSquaredStrategy())
        samf.change_strategy(2)
        assert samf.metadata.marker[ind] == -1
        assert samf._active_strategy_ind == 2

        samf.change_strategy(samf.strategies[1])
        assert samf._active_strategy_ind == 1
        assert samf.metadata.marker[ind] == -2

        new_strat = HistogramStrategy()
        samf.strategies.append(new_strat)
        samf.change_strategy(3)
        assert samf._active_strategy_ind == 3
        assert samf.active_strategy is new_strat
        assert samf.metadata.marker[ind] == -2


class TestSamfireMain:

    def setup_method(self, method):
        np.random.seed(1)
        self.model, self.lor1, self.g, self.lor2 = generate_test_model()
        self.shape = (7, 15)

    def test_multiprocessed(self):
        self.model.fit()
        samf = self.model.create_samfire(ipyparallel=False)
        samf.plot_every = np.nan
        samf.strategies[0].radii = 1.
        samf.strategies.remove(1)
        samf.optional_components = [self.model[2]]
        samf.start(fitter='mpfit', bounded=True)
        # let at most 3 pixels to fail randomly.
        fitmask = samf.metadata.marker == -np.ones(self.shape)
        print('number of pixels failed: {}'.format(np.sum(fitmask) -
                                                   np.prod(self.shape)))
        assert np.sum(fitmask) >= np.prod(self.shape) - 3
        for o_c, n_c in zip([self.g, self.lor1, self.lor2], self.model):
            for p, p1 in zip(o_c.parameters, n_c.parameters):
                if n_c._active_array is not None:
                    mask = np.logical_and(n_c._active_array, fitmask)
                else:
                    mask = fitmask
                print(o_c._id_name, n_c._id_name, p1._id_name, p._id_name)
                print(p.map['values'][:4, :4])
                print('----------------------------')
                print(p1.map['values'][:4, :4])
                print('ooooooooooooooooooooooooooooooooooooooooooo')
                np.testing.assert_allclose(
                    p1.map['values'][mask],
                    p.map['values'][:7, :15][mask],
                    rtol=0.3)


def test_create_worker_defaults():
    worker = create_worker('worker')
    assert worker.identity == 'worker'
    assert worker.shared_queue is None
    assert worker.result_queue is None
    assert worker.individual_queue is None
    np.testing.assert_equal(worker.best_AICc, np.inf)
    np.testing.assert_equal(worker.best_values, [])
    np.testing.assert_equal(worker.best_dof, np.inf)
    np.testing.assert_equal(worker.last_time, 1)


class TestSamfireWorker:

    def setup_method(self, method):
        np.random.seed(17)
        ax = np.arange(250)

        self.widths = [5, 10, 15]
        self.centres = [50, 105, 180]
        self.areas = [5000, 10000, 20000]

        g = hs.model.components1D.Gaussian()
        g.sigma.value = self.widths[0]
        g.A.value = self.areas[0]

        l = hs.model.components1D.Lorentzian()
        l.gamma.value = self.widths[1]
        l.A.value = self.areas[1]

        l1 = hs.model.components1D.Lorentzian()
        l1.gamma.value = self.widths[2]
        l1.A.value = self.areas[2]

        d = g.function(ax - self.centres[0]) + \
            l.function(ax - self.centres[1]) + \
            l1.function(ax - self.centres[2])
        s = hs.signals.Signal1D(np.array([d, d]))
        s.add_poissonian_noise()
        s.metadata.Signal.set_item("Noise_properties.variance",
                                   s.deepcopy() + 1.)
        m = s.create_model()
        m.append(hs.model.components1D.Gaussian())
        m[-1].name = 'g1'
        m.append(hs.model.components1D.Lorentzian())
        m[-1].name = 'l1'
        m.append(hs.model.components1D.Lorentzian())
        m[-1].name = 'l2'
        m.append(hs.model.components1D.Gaussian())
        m[-1].name = 'g2'
        m.append(hs.model.components1D.Gaussian())
        m[-1].name = 'g3'
        m.append(hs.model.components1D.Lorentzian())
        m[-1].name = 'l3'

        for c in m:
            c.active_is_multidimensional = True

        vals = {'g1': {},
                'g2': {},
                'g3': {},
                'l1': {},
                'l2': {},
                'l3': {},
                }

        vals['g1']['centre'] = [50, 150]
        vals['g1']['sigma'] = [5]
        vals['g1']['A'] = [10000]

        vals['l1']['centre'] = [43]
        vals['l1']['gamma'] = [25]
        vals['l1']['A'] = [10000]

        vals['l2']['centre'] = [125]
        vals['l2']['gamma'] = [8]
        vals['l2']['A'] = [10000]

        vals['g2']['centre'] = [105]
        vals['g2']['sigma'] = [20]
        vals['g2']['A'] = [10000]

        vals['l3']['centre'] = [185]
        vals['l3']['gamma'] = [11]
        vals['l3']['A'] = [10000]

        vals['g3']['centre'] = [175]
        vals['g3']['sigma'] = [12]
        vals['g3']['A'] = [10000]

        self.vals = vals
        self.model = m
        self.q = Mock_queue()
        self.ind = (1,)
        self.args = {}
        self.model_letter = 'sldkfjg'
        from hyperspy.samfire_utils.fit_tests import red_chisq_test as rct
        self._gt_dump = dill.dumps(rct(tolerance=1.0))
        m_slice = m.inav[self.ind[::-1]]
        m_slice.store(self.model_letter)
        m_dict = m_slice.signal._to_dictionary(False)
        m_dict['models'] = m_slice.signal.models._models.as_dictionary()
        self.model_dictionary = m_dict
        self.optional_comps = [1, 2, 3, 4, 5]

    def test_add_model(self):
        worker = create_worker('worker')
        worker.create_model(self.model_dictionary, self.model_letter)
        from hyperspy.model import BaseModel
        assert isinstance(worker.model, BaseModel)
        for component in worker.model:
            assert not component.active_is_multidimensional
            assert component.active

    def test_main_result(self):
        worker = create_worker('worker')
        worker.create_model(self.model_dictionary, self.model_letter)
        worker.setup_test(self._gt_dump)
        worker.set_optional_names({self.model[comp].name for comp in
                                   self.optional_comps})
        self.vals.update({
            'signal.data': self.model.signal(),
            'fitting_kwargs': {},
            'variance.data':
            self.model.signal.metadata.Signal.Noise_properties.variance()
        })
        keyword, (_id, _ind, result, found_solution) = \
            worker.run_pixel(self.ind, self.vals)
        assert _id == 'worker'
        assert _ind == self.ind
        assert found_solution

        assert result['dof.data'][()] == 9

        lor_components = [key for key in result['components'].keys() if
                          key.find('l') == 0]
        assert len(result['components']) == 3
        assert len(lor_components) == 2

        gauss_name = list(set(result['components'].keys()) -
                          set(lor_components))[0]

        gauss = result['components'][gauss_name]
        np.testing.assert_allclose(gauss['A'][0]['values'], self.areas[0],
                                   rtol=0.05)
        np.testing.assert_allclose(gauss['sigma'][0]['values'], self.widths[0],
                                   rtol=0.05)
        np.testing.assert_allclose(gauss['centre'][0]['values'],
                                   self.centres[0], rtol=0.05)

        lor1 = result['components'][lor_components[0]]
        lor1_values = tuple(lor1[par][0]['values'] for par in ['A', 'gamma',
                                                               'centre'])
        lor2 = result['components'][lor_components[1]]
        lor2_values = tuple(lor2[par][0]['values'] for par in ['A', 'gamma',
                                                               'centre'])

        possible_values1 = (self.areas[1], self.widths[1], self.centres[1])
        possible_values2 = (self.areas[2], self.widths[2], self.centres[2])

        assert (np.allclose(lor1_values, possible_values1, rtol=0.05)
                or
                np.allclose(lor1_values, possible_values2, rtol=0.05))

        assert (np.allclose(lor2_values, possible_values1, rtol=0.05)
                or
                np.allclose(lor2_values, possible_values2, rtol=0.05))
