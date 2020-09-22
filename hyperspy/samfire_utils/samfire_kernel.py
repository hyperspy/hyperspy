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


def single_kernel(model, ind, values, optional_components, _args, test):
    from itertools import combinations, product
    import numpy as np
    from hyperspy.utils.model_selection import AICc

    def generate_values_iterator(compnames, vals, turned_on_component_inds):
        turned_on_names = [compnames[i] for i in turned_on_component_inds]
        tmp = []
        name_list = []
        # TODO: put changing _position parameter of each component at the
        # beginning
        for _comp_n, _comp in vals.items():
            if _comp_n in turned_on_names:
                for par_n, par in _comp.items():
                    if not isinstance(par, list):
                        par = [par]
                    tmp.append(par)
                    name_list.append((_comp_n, par_n))
        return name_list, product(*tmp)

    comb = []
    AICc_fraction = 0.99
    model.axes_manager.indices = ind[::-1]
    for comp in optional_components:
        model[comp].active = False

    for num in range(len(optional_components) + 1):
        for c in combinations(optional_components, num):
            comb.append(c)

    best_AICc, best_dof = np.inf, np.inf
    best_comb, best_values, best_names = None, None, None

    component_names = [c.name for c in model]

    for combination in comb:
        # iterate all component combinations
        for component in combination:
            model[component].active = True

        on_comps = [i for i, _c in enumerate(model) if _c.active]
        name_list, iterator = generate_values_iterator(
            component_names, values, on_comps)

        ifgood = False
        for it in iterator:
            # iterate all parameter value combinations
            for (comp_n, par_n), val in zip(name_list, it):
                try:
                    getattr(model[comp_n], par_n).value = val
                except:
                    pass
            model.fit(**_args)
            # only perform iterations until we find a solution that we think is
            # good enough
            ifgood = test.test(model, ind)
            if ifgood:
                break
        if ifgood:
            # shortcut when no optional components:
            if len(comb) == 1:
                return True

            # get model with best chisq here, and test model validation
            new_AICc = AICc(model.inav[ind[::-1]])

            if new_AICc < AICc_fraction * best_AICc or \
                    (np.abs(new_AICc - best_AICc) <= np.abs(AICc_fraction * best_AICc)
                     and len(model.p0) < best_dof):
                best_values = [
                    getattr(
                        model[comp_n],
                        par_n).value for comp_n,
                    par_n in name_list]
                best_names = name_list
                best_comb = combination
                best_AICc = new_AICc
                best_dof = len(model.p0)
        for component in combination:
            model[component].active = False

    # take the best overall combination of components and parameters:
    if best_comb is None:
        model.chisq.data[ind] = np.nan
        return False
    else:
        for component in best_comb:
            model[component].active = True
        for (comp_n, par_n), val in zip(best_names, best_values):
            try:
                getattr(model[comp_n], par_n).value = val
            except:
                pass
        model.fit(**_args)
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
        for _comp_n, _comp in vals.items():
            if _comp_n in turned_on_names:
                for par_n, par in _comp.items():
                    if not isinstance(par, list):
                        par = [par, ]
                    tmp.append(par)
                    name_list.append((_comp_n, par_n))
        return name_list, product(*tmp)

    def send_good_results(model, previous_switching, cur_p, result_q, ind):
        result = copy.deepcopy(model.as_dictionary())
        for num, a_i_m in enumerate(previous_switching):
            result['components'][num]['active_is_multidimensional'] = a_i_m
        result['current'] = cur_p._identity
        result_q.put((ind, result, True))

    test = dill.loads(test_dict)
    cur_p = current_process()
    previous_switching = []
    comb = []
    AICc_fraction = 0.99

    comp_dict = m_dic['models']['z']['_dict']['components']
    for num, comp in enumerate(comp_dict):
        previous_switching.append(comp['active_is_multidimensional'])
        comp['active_is_multidimensional'] = False
    for comp in optional_components:
        comp_dict[comp]['active'] = False

    for num in range(len(optional_components) + 1):
        for comp in combinations(optional_components, num):
            comb.append(comp)

    best_AICc, best_dof = np.inf, np.inf
    best_comb, best_values, best_names = None, None, None

    component_names = [c['name'] for c in comp_dict]

    sig = Signal(**m_dic)
    sig._assign_subclass()
    model = sig.models.z.restore()
    for combination in comb:
        # iterate all component combinations
        for component in combination:
            model[component].active = True

        on_comps = [i for i, _c in enumerate(model) if _c.active]
        name_list, iterator = generate_values_iterator(
            component_names, values, on_comps)

        ifgood = False
        for it in iterator:
            # iterate all parameter value combinations
            for (comp_n, par_n), val in zip(name_list, it):
                try:
                    getattr(model[comp_n], par_n).value = val
                except:
                    pass
            model.fit(**_args)
            # only perform iterations until we find a solution that we think is
            # good enough
            ifgood = test.test(model, (0,))
            if ifgood:
                break
        if ifgood:
            # get model with best chisq here, and test model validation
            if len(comb) == 1:
                send_good_results(
                    model,
                    previous_switching,
                    cur_p,
                    result_q,
                    ind)
            new_AICc = AICc(model)

            if new_AICc < AICc_fraction * best_AICc or \
                    (np.abs(new_AICc - best_AICc) <= np.abs(AICc_fraction * best_AICc)
                     and len(model.p0) < best_dof):
                best_values = [
                    getattr(
                        model[comp_n],
                        par_n).value for comp_n,
                    par_n in name_list]
                best_names = name_list
                best_comb = combination
                best_AICc = new_AICc
                best_dof = len(model.p0)
        for component in combination:
            model[component].active = False

    # take the best overall combination of components and parameters:
    if best_comb is None:
        result_q.put((ind, None, False))
    else:
        for component in best_comb:
            model[component].active = True
        for (comp_n, par_n), val in zip(best_names, best_values):
            try:
                getattr(model[comp_n], par_n).value = val
            except:
                pass
        model.fit(**_args)
        send_good_results(model, previous_switching, cur_p, result_q, ind)
        # result = copy.deepcopy(m.as_dictionary(picklable=True))
        # for num, c in enumerate(previous_switching):
        #     result['components'][num]['active_is_multidimensional'] = c
        # result['current'] = cur_p._identity
        # result_q.put((ind, result, True))
