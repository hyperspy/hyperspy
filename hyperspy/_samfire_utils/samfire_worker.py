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
import sys
from itertools import combinations, product
from hyperspy.signal import Signal
import dill
import time
from queue import Empty


class worker:

    def __init__(self, identity, individual_queue):
        self.identity = identity
        self.individual_queue = individual_queue
        self.shared_queue = None
        self.result_queue = None
        self.timestep = 0.05
        self.max_get_timeout = 3

        self._AICc_fraction = 0.99

        self.reset()

        self.last_time = 1
        self.start_listening()

    def setup_queues(self, shared_queue, result_queue):
        self.shared_queue = shared_queue
        self.result_queue = result_queue

    def create_model(self, signal_dict, model_letter):
        sig = Signal(**signal_dict)
        sig._assign_subclass()
        self.model = getattr(sig.models, model_letter).restore()
        for component in self.model:
            component.active_is_multidimensional = False
            component.active = True

    def set_optional_names(self, optional_names):
        self.optional_names = optional_names

    def generate_values_iterator(turned_on_names):
        tmp = []
        name_list = []
        for _comp_n, _comp in self.value_dict.items():
            for par_n, par in _comp.items():
                if _comp_n not in turned_on_names:
                    par = [None, ]
                if not isinstance(par, list):
                    par = [par, ]
                tmp.append(par)
                name_list.append((_comp_n, par_n))
        return name_list, product(*tmp)

    def set_values(self, name_list, iterator):
        for value_combination in iterator:
            for (comp_name, parameter_name), value in zip(name_list,
                                                          value_combination):
                if value is None:
                    self.model[comp_name].active = False
                else:
                    self.model[comp_name].active = True
                    try:
                        getattr(self.model[comp_name],
                                parameter_name).value = value
                    except:
                        e = sys.exc_info()[0]
                        self.result_queue.put(('Error',
                                               'Setting {}.{} value to {}.'
                                               'Caught:\n{}'.format(comp_name,
                                                                    parameter_name,
                                                                    value, e)))
            yield

    def fit(self, component_comb):
        name_list, iterator = self.generate_values_iterator(component_comb)
        good_fit = False
        for _ in self.set_values(name_list, iterator):
            self.model.fit(**self.fitting_kwargs)
            good_fit = self.fit_test.test(self.model, (0,))
            if good_fit:
                break
        return good_fit

    def generate_component_combinations(self):
        all_names = {component.name for component in self.model}
        names_to_skip_generators = [combinations(self.optional_names, howmany)
                                    for howmany in
                                    range(len(self.optional_names) + 1)]
        names_to_skip = []
        for _gen in reversed(names_to_skip_generators):
            names_to_skip.extend(list(_gen))
        for name_comb in names_to_skip:
            yield all_names - set(name_comb)

    def reset(self):
        self.best_AICc = np.inf
        self.best_values = []
        self.best_dof = np.inf

    def test(self, ind, value_dict):
        self.reset()
        self.ind = ind
        self.value_dict = value_dict

        self.fitting_kwargs = self.value_dict.pop('fitting_kwargs', {})
        for component_comb in generate_component_combinations():
            good_fit = self.fit(component_comb)

            if good_fit:
                if len(self.optional_names) == 0:
                    self.send_results(current=True)
                    return
                else:
                    self.compare_models()
        self.send_results()

    def _collect_values(self):
        result = {component.name: {parameter.name: parameter.map[0] for
                                   parameter in component.parameters} for
                  component in self.model if component.active}
        return result

    def compare_model(self):
        new_AICc = AICc(self.model)

        if (new_AICc < self._AICc_fraction * self.best_AICc) or \
            (np.abs(new_AICc - self.best_AICc) <= np.abs(self._AICc_fraction *
                                                         self.best_AICc) and
             len(self.model.p0) < self.best_dof:

             self.best_values=self._collect_values()
             self.best_AICc=new_AICc
             self.best_dof=len(self.model.p0)
             self.best_chisq=self.model.chisq.data[0]

    def send_results(self, current=False):
        if current:
             self.best_chisq=self.model.chisq.data[0]
             self.best_dof=len(self.model.p0)
             self.best_values=self._collect_values()
        if len(self.best_values):  # i.e. we have a good result
             result={'chisq': self.best_chisq
                      'dof': self.best_dof,
                      'components': self.best_values
                     }
             found_solution=True
        else:
             result=None
             found_solution=False
        self.result_queue.put((self.ind, result, found_solution))

    def setup_test(self, test_string):
        self.fit_test=dill.loads(test_string)

    def start_listening(self):
        self._listening=True
        self.listen()

    def stop_listening(self):
        self._listening=False

    def parse(self, result):
        function=result
        arguments=[]
        if isinstance(result, tuple):
            function, arguments=result
        getattr(self, function)(*arguments)

    def ping(self, message=None):
        self.result_queue.put(('pong', self.identity, time.time(), message))

    def sleep(self, time=None):
        if time is None:
            time=self.timestep
        self.last_time=time.time()
        time.sleep(time)

    def listen(self):
        while self._listening:
            queue=None
            found_what_to_do=False
            time_diff=time.time() - self.last_time
            if time_diff >= self.timestep:
                if not self.individual_queue.empty():
                    queue=self.individual_queue
                elif (self.shared_queue is not None and not
                      self.shared_queue.empty()):
                    queue=self.shared_queue
                if queue is not None:
                    try:
                        result=queue.get(block=True,
                                           timeout=self.max_get_timeout)
                        found_what_to_do=True
                        self.parse(result)
                    except Empty:
                        pass
                if not found_what_to_do:
                    self.sleep()
            else:
                self.sleep()
