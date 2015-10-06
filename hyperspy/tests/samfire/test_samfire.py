# Copyright 2007-2014 The HyperSpy developers
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
import dill
import copy
import hyperspy.hspy as hs
from hyperspy.model import Model
from hyperspy._samfire_utils.samfire_kernel import multi_kernel
from hyperspy.misc.utils import DictionaryTreeBrowser


class Mock_queue(object):

    def __init__(self):
        self.var = []

    def put(self, value):
        self.var.append(value)


def generate_test_model():

    import hyperspy.hspy as hs
    import numpy as np
    from scipy.ndimage import gaussian_filter
    total = None
# blurs = [0., 0.5, 1., 2.,5.]
    blurs = [2.5]
    radius = 5
    domain = 15
# do circle/domain
    cent = (domain // 2, domain // 2)
    y, x = np.ogrid[-cent[0]:domain - cent[0], -cent[1]:domain - cent[1]]
    mask = x * x + y * y <= radius * radius
    lor_map = None
    for blur in blurs:

        s = hs.signals.SpectrumSimulation(np.ones((domain, domain, 1024)))
        cent = tuple([int(0.5 * i) for i in s.data.shape[:-1]])
        m0 = s.create_model()

        gs01 = hs.components.Lorentzian()
        m0.append(gs01)
        gs01.gamma.map['values'][:] = 40
        gs01.gamma.map['is_set'][:] = True
        gs01.centre.map['values'][:] = 300
        gs01.centre.map['values'][mask] = 450
        gs01.centre.map['values'] = gaussian_filter(
            gs01.centre.map['values'],
            blur)
        gs01.centre.map['is_set'][:] = True
        gs01.A.map['values'][:] = 100 * \
            np.random.random((domain, domain)) + 300000
        gs01.A.map['values'][mask] *= 0.75
        gs01.A.map['values'] = gaussian_filter(gs01.A.map['values'], blur)
        gs01.A.map['is_set'][:] = True

        gs02 = hs.components.Gaussian()
        m0.append(gs02)
        gs02.sigma.map['values'][:] = 30
        gs02.sigma.map['is_set'][:] = True
        gs02.centre.map['values'][:] = 400
        gs02.centre.map['values'][mask] = 300
        gs02.centre.map['values'] = gaussian_filter(
            gs02.centre.map['values'],
            blur)
        gs02.centre.map['is_set'][:] = True
        gs02.A.map['values'][:] = 50000
        gs02.A.map['is_set'][:] = True

        gs03 = hs.components.Lorentzian()
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

        s11 = m0.as_signal(show_progressbar=False)
        if total is None:
            total = s11.data.copy()
            lor_map = gs01.centre.map['values'].copy()
        else:
            total = np.concatenate((total, s11.data), axis=1)
            lor_map = np.concatenate(
                (lor_map, gs01.centre.map['values'].copy()), axis=1)

    s = hs.signals.SpectrumSimulation(total)
    s.add_poissonian_noise()
    s.data += 0.1
    s.estimate_poissonian_noise_variance()

    m = s.inav[:, :7].create_model()
    g = hs.components.Gaussian()
    l1 = hs.components.Lorentzian()
    l2 = hs.components.Lorentzian()
    g.sigma.value = 30
    g.centre.value = 400
    g.A.value = 50000
    l1.gamma.value = 40
    l1.centre.value = 300
    l1.A.value = 300000
    l2.gamma.value = 20
    l2.centre.value = 100
    l2.A.value = 50000
    l2.centre.bmin = 0
    l2.centre.bmax = 1000
    l2.A.bmin = 30000
    l2.A.bmax = 100000
    m.extend([g, l1, l2])
    m.assign_current_values_to_all()
    l2.active_is_multidimensional = True
    return m, gs01, gs02, gs03


class TestSamfireEmpty:

    def setUp(self):
        self.shape = (7, 15)
        s = hs.signals.SpectrumSimulation(np.empty(self.shape + (1024,)))
        s.estimate_poissonian_noise_variance()
        m = s.create_model()
        m.append(hs.components.Gaussian())
        m.append(hs.components.Lorentzian())
        m.append(hs.components.Lorentzian())
        self.model = m

    def test_setup(self):
        m = self.model
        samf = m.create_samfire(workers=1)
        nt.assert_is_none(samf._gt_dump)
        nt.assert_is_none(samf._result_q)
        nt.assert_is_none(samf.pool)
        samf._setup()
        nt.assert_is_not_none(samf._gt_dump)
        nt.assert_is_not_none(samf._result_q)
        nt.assert_is_not_none(samf.pool)
        nt.assert_equal(samf.pool._state, 0)

        samf.pool.close()
        samf._setup()
        nt.assert_equal(samf.pool._state, 0)

        samf.pool.terminate()
        samf._setup()
        nt.assert_equal(samf.pool._state, 0)

    def test_samfire_init_marker(self):
        m = self.model
        samf = m.create_samfire(marker=np.zeros(self.shape), workers=1)
        nt.assert_true(np.allclose(samf.metadata.marker, np.zeros(self.shape)))

    def test_samfire_init_workers(self):
        m = self.model
        samf = m.create_samfire(marker=np.zeros(self.shape), workers=1)
        nt.assert_equal(samf.workers, 1)
        samf = m.create_samfire(marker=np.zeros(self.shape), workers=-1.333)
        nt.assert_equal(samf.workers, 1)
        nt.assert_is_instance(samf.workers, int)

    def test_samfire_init_model(self):
        m = self.model
        samf = m.create_samfire(marker=np.zeros(self.shape), workers=1)
        nt.assert_true(samf.model is m)

    def test_samfire_init_metadata(self):
        m = self.model
        samf = m.create_samfire(marker=np.zeros(self.shape), workers=1)
        nt.assert_true(isinstance(samf.metadata, DictionaryTreeBrowser))

    def test_samfire_init_strategy_list(self):
        m = self.model
        samf = m.create_samfire(marker=np.zeros(self.shape), workers=1)
        nt.assert_true(isinstance(samf.strategies, samf._strategy_list))

    def test_samfire_init_strategies(self):
        m = self.model
        samf = m.create_samfire(marker=np.zeros(self.shape), workers=1)
        from hyperspy._samfire_utils._strategies.diffusion.red_chisq import reduced_chi_squared_strategy
        from hyperspy._samfire_utils._strategies.segmenter.histogram import histogram_strategy
        nt.assert_true(
            isinstance(
                samf.strategies[0],
                reduced_chi_squared_strategy))
        nt.assert_true(isinstance(samf.strategies[1], histogram_strategy))

    def test_samfire_init_fig(self):
        m = self.model
        samf = m.create_samfire(marker=np.zeros(self.shape), workers=1)
        nt.assert_true(samf._figure is None)

    def test_samfire_init_default(self):
        m = self.model
        from multiprocessing import cpu_count
        samf = m.create_samfire()
        nt.assert_equal(samf.workers, cpu_count() - 1)
        nt.assert_true(np.allclose(samf.metadata.marker, np.zeros(self.shape)))

    def test_optional_components(self):
        m = self.model
        m[-1].active_is_multidimensional = False

        samf = m.create_samfire()
        samf.optional_components = [m[0], 1]
        samf._enable_optional_components()
        nt.assert_true(m[0].active_is_multidimensional)
        nt.assert_true(m[1].active_is_multidimensional)
        nt.assert_true(np.all([isinstance(a, int)
                       for a in samf.optional_components]))
        nt.assert_true(np.allclose(samf.optional_components, [0, 1]))

    def test_swap_dict_and_model(self):
        m = self.model
        for i in range(len(m)):
            for ip, p in enumerate(m[i].parameters):
                p.map['values'][0, 0] = 3.0 + i + ip
                p.map['std'][0, 0] = 2.44 + i + ip
                p.map['is_set'][0, 0] = True
        m[1].active_is_multidimensional = True
        m[1]._active_array[0, 0] = False
        nt.assert_true(m[1]._active_array[1, 0])
        m.chisq.data[0, 0] = 1200.
        m.dof.data[0, 0] = 1.
        d = copy.deepcopy(m.inav[0, 0].as_dictionary())
        samf = m.create_samfire()
        samf._swap_dict_and_model((1, 0), d)
        nt.assert_equal(m.chisq.data[1, 0], 1200.)
        nt.assert_equal(m.dof.data[1, 0], 1.)

        nt.assert_equal(d['dof.data'], 0.)
        nt.assert_true(np.isnan(d['chisq.data']))

        # nt.assert_false(m[1]._active_array[1, 0])
        for c in m:
            for p in c.parameters:
                nt.assert_equal(p.map['values'][0, 0], p.map['values'][1, 0])
                nt.assert_equal(p.map['std'][0, 0], p.map['std'][1, 0])
                nt.assert_equal(p.map['is_set'][0, 0], p.map['is_set'][1, 0])

    def test_next_pixels(self):
        m = self.model
        samf = m.create_samfire()
        ans = samf._next_pixels(3)
        nt.assert_equal(len(ans), 0)
        ind_list = [(1, 2), (0, 1), (3, 3), (4, 6)]
        for n, ind in enumerate(ind_list):
            samf.metadata.marker[ind] += 2.
        ans = samf._next_pixels(10)
        nt.assert_equal(len(ans), 4)
        for ind in ans:
            nt.assert_in(ind, ind_list)

        for n, ind in enumerate(ind_list):
            samf.metadata.marker[ind] += n
        ans = samf._next_pixels(10)
        nt.assert_equal(ans, [(4, 6), ])

    def test_change_strategy(self):
        m = self.model
        samf = m.create_samfire()
        from hyperspy._samfire_utils._strategies.diffusion.red_chisq import reduced_chi_squared_strategy
        from hyperspy._samfire_utils._strategies.segmenter.histogram import histogram_strategy

        ind = (0, 0)
        samf.metadata.marker[ind] = -2
        samf.strategies.append(reduced_chi_squared_strategy())
        samf.change_strategy(2)
        nt.assert_equal(samf.metadata.marker[ind], -1)
        nt.assert_equal(samf.active_strategy, 2)

        samf.change_strategy(1)
        nt.assert_equal(samf.active_strategy, 1)
        nt.assert_equal(samf.metadata.marker[ind], -2)

        samf.strategies.append(histogram_strategy())
        samf.change_strategy(3)
        nt.assert_equal(samf.active_strategy, 3)
        nt.assert_equal(samf.metadata.marker[ind], -2)


class TestSamfireMain:

    def setUp(self):
        np.random.seed(1)
        self.model, self.lor1, self.g, self.lor2 = generate_test_model()
        self.shape = (7, 15)

    def test_all(self):
        self.model.fit()
        samf = self.model.create_samfire()
        samf.plot_every = np.nan
        samf.strategies[0].radii = 1.
        samf.strategies.remove(1)
        samf.optional_components = [self.model[2]]
        samf.start(fitter='mpfit', bounded=True)
        nt.assert_true(np.all(samf.metadata.marker == -np.ones(self.shape)))
        for o_c, n_c in zip([self.g, self.lor1, self.lor2], self.model):
            for p, p1 in zip(o_c.parameters, n_c.parameters):

                print o_c._id_name, n_c._id_name, p1._id_name, p._id_name
                print p.map['values'][:4, :4]
                print '----------------------------'
                print p1.map['values'][:4, :4]
                print 'ooooooooooooooooooooooooooooooooooooooooooo'

                test = np.allclose(
                    p.map['values'][
                        :7, :15][
                        n_c._active_array], p1.map['values'][
                        n_c._active_array], rtol=0.2)
                nt.assert_true(test)


class TestSamfireFitKernel:

    def setUp(self):
        np.random.seed(17)
        ax = np.arange(200)

        self.widths = [10, 20, 30]
        self.centres = [50, 105, 180]
        self.areas = [5000, 10000, 20000]

        g = hs.components.Gaussian()
        g.sigma.value = self.widths[0]
        g.A.value = self.areas[0]

        l = hs.components.Lorentzian()
        l.gamma.value = self.widths[1]
        l.A.value = self.areas[1]

        l1 = hs.components.Lorentzian()
        l1.gamma.value = self.widths[2]
        l1.A.value = self.areas[2]

        d = g.function(ax - self.centres[0]) + \
            l.function(ax - self.centres[1]) + \
            l1.function(ax - self.centres[2])
        s = hs.signals.SpectrumSimulation(np.array([d, d]))
        s.add_poissonian_noise()
        s.metadata.Signal.set_item(
            "Noise_properties.variance",
            hs.signals.Signal(
                s.data))

        m = s.create_model()
        m.append(hs.components.Gaussian())
        m[-1].name = 'g1'
        m.append(hs.components.Lorentzian())
        m[-1].name = 'l1'
        m.append(hs.components.Lorentzian())
        m[-1].name = 'l2'
        m.append(hs.components.Gaussian())
        m[-1].name = 'g2'
        m.append(hs.components.Gaussian())
        m[-1].name = 'g3'
        m.append(hs.components.Lorentzian())
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
        vals['g1']['sigma'] = [13]
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
        from hyperspy._samfire_utils.fit_tests import red_chisq_test as rct
        self.gt_dump = dill.dumps(rct(tolerance=1.0))

    def test_main_result(self):
        m = self.model
        result_q = self.q
        m_dict = m.inav[self.ind[::-1]]
        m_dict.stash.save('z')
        m_dict = m_dict.spectrum._to_dictionary()
        optional_comps = [1, 2, 3, 4, 5]
        run_args = (self.ind,
                    m_dict,
                    self.vals,
                    optional_comps,
                    self.args,
                    result_q,
                    self.gt_dump)
        multi_kernel(*run_args)
        _, result, _ = result_q.var[0]

        nt.assert_equal(result['dof.data'][0], 9)

        for c in range(3):
            nt.assert_true(result['components'][c]['active'])
        for c in range(3, 6):
            nt.assert_false(result['components'][c]['active'])

        possible_pars = [
            (A, w, c) for c, A, w in zip(
                self.centres, self.areas, self.widths)]

        for ic in range(3):
            comp = result['components'][ic]
            this_val = [comp['parameters'][ip]['value'] for ip in range(3)]
            # to allow for swapping of the components
            nt.assert_true(
                np.any([np.allclose(pp, this_val, rtol=0.05) for pp in possible_pars]))
