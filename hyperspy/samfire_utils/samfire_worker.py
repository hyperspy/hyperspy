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

import logging
import os
import time
import sys
from itertools import combinations, product
from queue import Empty
import dill
import numpy as np
import matplotlib

matplotlib.rcParams['backend'] = 'Agg'

from hyperspy.signal import BaseSignal
from hyperspy.utils.model_selection import AICc

_logger = logging.getLogger(__name__)


class Worker:

    def __init__(self, identity, individual_queue=None, shared_queue=None,
                 result_queue=None):
        self.identity = identity
        self.individual_queue = individual_queue
        self.shared_queue = shared_queue
        self.result_queue = result_queue
        self.timestep = 0.001
        self.max_get_timeout = 3
        self._AICc_fraction = 0.99
        self.reset()
        self.last_time = 1
        self.optional_names = set()
        self.model = None
        self.parameters = {}

    def create_model(self, signal_dict, model_letter):
        _logger.debug('Creating model in worker {}'.format(self.identity))
        sig = BaseSignal(**signal_dict)
        sig._assign_subclass()
        self.model = sig.models[model_letter].restore()
        for component in self.model:
            component.active_is_multidimensional = False
            component.active = True
            for par in component.parameters:
                par.map = par.map.copy()

        if self.model.signal.metadata.has_item(
                'Signal.Noise_properties.variance'):
            var = self.model.signal.metadata.Signal.Noise_properties.variance
            if isinstance(var, BaseSignal):
                var.data = var.data.copy()
        self._array_views_to_copies()

    def _array_views_to_copies(self):
        dct = self.model.__dict__
        self.parameters = {}
        for k, v in dct.items():
            if isinstance(v, BaseSignal):
                v.data = v.data.copy()
                if k not in ['signal', 'image', 'spectrum'] and not \
                   k.startswith('_'):
                    self.parameters[k] = None
            if isinstance(v, np.ndarray):
                dct[k] = v.copy()

    def set_optional_names(self, optional_names):
        self.optional_names = optional_names
        _logger.debug('Setting optional names in worker {} to '
                      '{}'.format(self.identity, self.optional_names))

    def set_parameter_boundaries(self, received):
        for rec, comp in zip(received, self.model):
            for (bmin, bmax), par in zip(rec, comp.parameters):
                par.bmin = bmin
                par.bmax = bmax

    def generate_values_iterator(self, turned_on_names):
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
                    except BaseException:
                        e = sys.exc_info()[0]
                        to_send = ('Error',
                                   (self.identity,
                                    'Setting {}.{} value to {}. '
                                    'Caught:\n{}'.format(comp_name,
                                                         parameter_name,
                                                         value,
                                                         e)
                                    )
                                   )
                        if self.result_queue is None:
                            return to_send
                        else:
                            self.result_queue.put()
                            return
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
        for _gen in names_to_skip_generators:
            names_to_skip.extend(list(_gen))
        for name_comb in names_to_skip:
            yield all_names - set(name_comb)

    def reset(self):
        self.best_AICc = np.inf
        self.best_values = []
        self.best_dof = np.inf

    def run_pixel(self, ind, value_dict):
        self.reset()
        self.ind = ind
        self.value_dict = value_dict

        self.fitting_kwargs = self.value_dict.pop('fitting_kwargs', {})
        if 'min_function' in self.fitting_kwargs:
            self.fitting_kwargs['min_function'] = dill.loads(
                self.fitting_kwargs['min_function'])
        if 'min_function_grad' in self.fitting_kwargs and isinstance(
                self.fitting_kwargs['min_function_grad'], bytes):
            self.fitting_kwargs['min_function_grad'] = dill.loads(
                self.fitting_kwargs['min_function_grad'])
        self.model.signal.data[:] = self.value_dict.pop('signal.data')

        if self.model.signal.metadata.has_item(
                'Signal.Noise_properties.variance'):
            var = self.model.signal.metadata.Signal.Noise_properties.variance
            if isinstance(var, BaseSignal):
                var.data[:] = self.value_dict.pop('variance.data')

        if 'low_loss.data' in self.value_dict:
            self.model.low_loss.data[:] = self.value_dict.pop('low_loss.data')

        for component_comb in self.generate_component_combinations():
            good_fit = self.fit(component_comb)

            if good_fit:
                if len(self.optional_names) == 0:
                    return self.send_results(current=True)
                else:
                    self.compare_models()
        return self.send_results()

    def _collect_values(self):
        result = {component.name: {parameter.name: parameter.map.copy() for
                                   parameter in component.parameters} for
                  component in self.model if component.active}
        return result

    def compare_models(self):
        new_AICc = AICc(self.model)

        AICc_test = new_AICc < (self._AICc_fraction * self.best_AICc)
        AICc_absolute_test = np.abs(new_AICc - self.best_AICc) <= \
            np.abs(self._AICc_fraction * self.best_AICc)
        dof_test = len(self.model.p0) < self.best_dof

        if AICc_test or AICc_absolute_test and dof_test:

            self.best_values = self._collect_values()
            self.best_AICc = new_AICc
            self.best_dof = len(self.model.p0)
            for k in self.parameters.keys():
                self.parameters[k] = getattr(self.model, k).data[0]

    def send_results(self, current=False):
        if current:
            self.best_values = self._collect_values()
            for k in self.parameters.keys():
                self.parameters[k] = getattr(self.model, k).data[0]
        if len(self.best_values):  # i.e. we have a good result
            _logger.debug('we have a good result in worker '
                          '{}'.format(self.identity))
            result = {k + '.data': np.array(v) for k, v in
                      self.parameters.items()}
            result['components'] = self.best_values
            found_solution = True
        else:
            _logger.debug("we don't have a good result in worker "
                          "{}".format(self.identity))
            result = None
            found_solution = False
        to_send = ('result', (self.identity, self.ind, result, found_solution))
        if self.individual_queue is None:
            return to_send
        self.result_queue.put(to_send)

    def setup_test(self, test_string):
        self.fit_test = dill.loads(test_string)

    def start_listening(self):
        self._listening = True
        self.listen()

    def stop_listening(self):
        self._listening = False

    def parse(self, result):
        function = result
        arguments = []
        if isinstance(result, tuple):
            function, arguments = result
        getattr(self, function)(*arguments)

    def ping(self, message=None):
        to_send = ('pong', (self.identity, os.getpid(), time.time(), message))
        if self.result_queue is None:
            return to_send
        self.result_queue.put(to_send)

    def sleep(self, howlong=None):
        if howlong is None:
            howlong = self.timestep
        self.last_time = time.time()
        time.sleep(howlong)

    def change_timestep(self, value):
        self.timestep = value

    def listen(self):
        while self._listening:
            queue = None
            found_what_to_do = False
            time_diff = time.time() - self.last_time
            if time_diff >= self.timestep:
                if not self.individual_queue.empty():
                    queue = self.individual_queue
                elif (self.shared_queue is not None and not
                      self.shared_queue.empty()):
                    queue = self.shared_queue
                if queue is not None:
                    try:
                        result = queue.get(block=True,
                                           timeout=self.max_get_timeout)
                        found_what_to_do = True
                        self.parse(result)
                    except Empty:
                        pass
                if not found_what_to_do:
                    self.sleep()
            else:
                self.sleep()


def create_worker(identity, individual_queue=None,
                  shared_queue=None, result_queue=None):
    w = Worker(identity, individual_queue, shared_queue, result_queue)
    if individual_queue is None:
        return w
    w.start_listening()
    return 1
