# Copyright 2007-2015 The HyperSpy developers
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


def single_kernel(m, ind, values, optional_components, _args, test):
    from itertools import combinations, product
    import numpy as np
    from hyperspy.utils.model_selection import AICc

    def generate_values_iterator(compnames, vals, turned_on_component_inds):
        turned_on_names = [compnames[i] for i in turned_on_component_inds]
        tmp = []
        name_list = []
        # TODO: put changing _position parameter of each component at the
        # beginning
        for _c_n, _c in vals.iteritems():
            if _c_n in turned_on_names:
                for p_n, p in _c.iteritems():
                    if not isinstance(p, list):
                        p = [p]
                    tmp.append(p)
                    name_list.append((_c_n, p_n))
        return name_list, product(*tmp)

    comb = []
    AICc_fraction = 0.99
    m.axes_manager.indices = ind[::-1]
    for c in optional_components:
        m[c].active = False

    for num in xrange(len(optional_components) + 1):
        for c in combinations(optional_components, num):
            comb.append(c)

    best_AICc, best_dof = np.inf, np.inf
    best_comb, best_values, best_names = None, None, None

    component_names = [c.name for c in m]

    for c in comb:
        # iterate all component combinations
        for component in c:
            m[component].active = True

        on_comps = [i for i, _c in enumerate(m) if _c.active]
        name_list, iterator = generate_values_iterator(
            component_names, values, on_comps)

        ifgood = False
        for it in iterator:
            # iterate all parameter value combinations
            for (c_n, p_n), v in zip(name_list, it):
                try:
                    getattr(m[c_n], p_n).value = v
                except:
                    pass
            m.fit(**_args)
            # only perform iterations until we find a solution that we think is
            # good enough
            ifgood = test.test(m, ind)
            if ifgood:
                break
        if ifgood:
            # shortcut when no optional components:
            if len(comb) == 1:
                return True

            # get model with best chisq here, and test model validation
            new_AICc = AICc(m.inav[ind[::-1]])

            if new_AICc < AICc_fraction * best_AICc or \
                    (np.abs(new_AICc - best_AICc) <= np.abs(AICc_fraction * best_AICc)
                     and len(m.p0) < best_dof):
                best_values = [
                    getattr(
                        m[c_n],
                        p_n).value for c_n,
                    p_n in name_list]
                best_names = name_list
                best_comb = c
                best_AICc = new_AICc
                best_dof = len(m.p0)
        for component in c:
            m[component].active = False

    # take the best overall combination of components and parameters:
    if best_comb is None:
        m.chisq.data[ind] = np.nan
        return False
    else:
        for component in best_comb:
            m[component].active = True
        for (c_n, p_n), v in zip(best_names, best_values):
            try:
                getattr(m[c_n], p_n).value = v
            except:
                pass
        m.fit(**_args)
        return True


def multi_kernel(
        ind, m_dic, values, optional_components, _args, result_q, test_dict):
    import hyperspy.api as hs
    from hyperspy.signal import Signal
    from multiprocessing import current_process
    from itertools import combinations, product
    # from collections import Iterable
    import numpy as np
    import copy
    from hyperspy.utils.model_selection import AICc
    import dill

    def generate_values_iterator(compnames, vals, turned_on_component_inds):
        turned_on_names = [compnames[i] for i in turned_on_component_inds]
        tmp = []
        name_list = []
        # TODO: put changing _position parameter of each component at the
        # beginning
        for _c_n, _c in vals.iteritems():
            if _c_n in turned_on_names:
                for p_n, p in _c.iteritems():
                    if not isinstance(p, list):
                        p = [p]
                    tmp.append(p)
                    name_list.append((_c_n, p_n))
        return name_list, product(*tmp)

    def send_good_results(m, previous_switching, cur_p, result_q, ind):
        result = copy.deepcopy(m.as_dictionary(picklable=True))
        for num, c in enumerate(previous_switching):
            result['components'][num]['active_is_multidimensional'] = c
        result['current'] = cur_p._identity
        result_q.put((ind, result, True))

    test = dill.loads(test_dict)
    cur_p = current_process()
    previous_switching = []
    comb = []
    AICc_fraction = 0.99

    comp_dict = m_dic['metadata']['Analysis'][
        'models']['z']['_dict']['components']
    for num, c in enumerate(comp_dict):
        previous_switching.append(c['active_is_multidimensional'])
        c['active_is_multidimensional'] = False
    for c in optional_components:
        comp_dict[c]['active'] = False

    for num in xrange(len(optional_components) + 1):
        for c in combinations(optional_components, num):
            comb.append(c)

    best_AICc, best_dof = np.inf, np.inf
    best_comb, best_values, best_names = None, None, None

    component_names = [c['name'] for c in comp_dict]

    s = Signal(**m_dic)
    s._assign_subclass()
    if s.metadata.Signal.signal_type == 'EELS':
        additional_kwds = {'low_loss': None,
                           'auto_background': False,
                           'auto_add_edges': False,
                           'GOS': None,
                           }
    else:
        additional_kwds = {}
    m = s.create_model(**additional_kwds)
    m.stash.pop('z')

    for c in comb:
        # iterate all component combinations
        for component in c:
            m[component].active = True

        on_comps = [i for i, _c in enumerate(m) if _c.active]
        name_list, iterator = generate_values_iterator(
            component_names, values, on_comps)

        ifgood = False
        for it in iterator:
            # iterate all parameter value combinations
            for (c_n, p_n), v in zip(name_list, it):
                try:
                    getattr(m[c_n], p_n).value = v
                except:
                    pass
            m.fit(**_args)
            # only perform iterations until we find a solution that we think is
            # good enough
            ifgood = test.test(m, (0,))
            if ifgood:
                break
        if ifgood:
            # get model with best chisq here, and test model validation
            if len(comb) == 1:
                send_good_results(m, previous_switching, cur_p, result_q, ind)
            new_AICc = AICc(m)

            if new_AICc < AICc_fraction * best_AICc or \
                    (np.abs(new_AICc - best_AICc) <= np.abs(AICc_fraction * best_AICc)
                     and len(m.p0) < best_dof):
                best_values = [
                    getattr(
                        m[c_n],
                        p_n).value for c_n,
                    p_n in name_list]
                best_names = name_list
                best_comb = c
                best_AICc = new_AICc
                best_dof = len(m.p0)
        for component in c:
            m[component].active = False

    # take the best overall combination of components and parameters:
    if best_comb is None:
        result_q.put((ind, None, False))
    else:
        for component in best_comb:
            m[component].active = True
        for (c_n, p_n), v in zip(best_names, best_values):
            try:
                getattr(m[c_n], p_n).value = v
            except:
                pass
        m.fit(**_args)
        send_good_results(m, previous_switching, cur_p, result_q, ind)
        # result = copy.deepcopy(m.as_dictionary(picklable=True))
        # for num, c in enumerate(previous_switching):
        #     result['components'][num]['active_is_multidimensional'] = c
        # result['current'] = cur_p._identity
        # result_q.put((ind, result, True))
