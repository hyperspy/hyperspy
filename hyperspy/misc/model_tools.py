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


def _is_iter(val):
    "Checks if value is a list or tuple"
    return isinstance(val, tuple) or isinstance(val, list)


def _iter_join(val):
    "Joins values of iterable parameters for the fancy view, unless it equals None, then blank"
    return "(" + ", ".join(["{:6g}".format(v) for v in val]) + ")" if val is not None else ""


def _non_iter(val):
    "Returns formatted string for a value unless it equals None, then blank"
    return "{:6g}".format(val) if val is not None else ""


class current_component_values():
    """Convenience class that makes use of __repr__ methods for nice printing in the notebook
    of the properties of parameters of a component
     
    Parameters
    ----------
    component : hyperspy component instance
    only_free : bool, default False
        If True: Only include the free parameters in the view
    only_active : bool, default False
        If True: Helper for current_model_values. Only include active components in the view.
        Always shows values if used on an individual component.
     """

    def __init__(self, component, only_free=False, only_active=False):
        self.name = component.name
        self.active = component.active
        self.parameters = component.parameters
        self._id_name = component._id_name
        self.only_free = only_free
        self.only_active = only_active

    def __repr__(self):
        # Number of digits for each label for the terminal-style view.
        size = {
            'name': 14,
            'free': 5,
            'value': 10,
            'std': 10,
            'bmin': 10,
            'bmax': 10,
        }
        # Using nested string formatting for flexibility in future updates
        signature = "{{:>{name}}} | {{:>{free}}} | {{:>{value}}} | {{:>{std}}} | {{:>{bmin}}} | {{:>{bmax}}}".format(
            **size)

        if self.only_active:
            text = "{0}: {1}".format(self.__class__.__name__, self.name)
        else:
            text = "{0}: {1}\nActive: {2}".format(
                self.__class__.__name__, self.name, self.active)
        text += "\n"
        text += signature.format("Parameter Name",
                                 "Free", "Value", "Std", "Min", "Max")
        text += "\n"
        text += signature.format("=" * size['name'], "=" * size['free'], "=" *
                                 size['value'], "=" * size['std'], "=" * size['bmin'], "=" * size['bmax'],)
        text += "\n"
        for para in self.parameters:
            if not self.only_free or self.only_free and para.free:
                if _is_iter(para.value):
                    # iterables (polynomial.value) must be handled separately
                    # `blank` results in a column of spaces
                    blank = len(para.value) * ['']
                    std = para.std if _is_iter(para.std) else blank
                    bmin = para.bmin if _is_iter(para.bmin) else blank
                    bmax = para.bmax if _is_iter(para.bmax) else blank
                    for i, (v, s, bn, bx) in enumerate(
                            zip(para.value, std, bmin, bmax)):
                        if i == 0:
                            text += signature.format(para.name[:size['name']], str(para.free)[:size['free']], str(
                                v)[:size['value']], str(s)[:size['std']], str(bn)[:size['bmin']], str(bx)[:size['bmax']])
                        else:
                            text += signature.format("", "", str(v)[:size['value']], str(
                                s)[:size['std']], str(bn)[:size['bmin']], str(bx)[:size['bmax']])
                        text += "\n"
                else:
                    text += signature.format(para.name[:size['name']], str(para.free)[:size['free']], str(para.value)[
                                             :size['value']], str(para.std)[:size['std']], str(para.bmin)[:size['bmin']], str(para.bmax)[:size['bmax']])
                    text += "\n"
        return text

    def _repr_html_(self):
        if self.only_active:
            text = "<p><b>{0}: {1}</b></p>".format(self.__class__.__name__, self.name)
        else:
            text = "<p><b>{0}: {1}</b><br />Active: {2}</p>".format(
                self.__class__.__name__, self.name, self.active)

        para_head = """<table style="width:100%"><tr><th>Parameter Name</th><th>Free</th>
            <th>Value</th><th>Std</th><th>Min</th><th>Max</th></tr>"""
        text += para_head
        for para in self.parameters:
            if not self.only_free or self.only_free and para.free:
                if _is_iter(para.value):
                    # iterables (polynomial.value) must be handled separately
                    # This should be removed with hyperspy 2.0 as Polynomial 
                    # has been replaced.
                    value = _iter_join(para.value)
                    std = _iter_join(para.std)
                    bmin = _iter_join(para.bmin)
                    bmax = _iter_join(para.bmax)
                else:
                    value = _non_iter(para.value)
                    std = _non_iter(para.std)
                    bmin = _non_iter(para.bmin)
                    bmax = _non_iter(para.bmax)

                text += """<tr><td>{0}</td><td>{1}</td><td>{2}</td>
                    <td>{3}</td><td>{4}</td><td>{5}</td></tr>""".format(
                        para.name, para.free, value, std, bmin, bmax)
        text += "</table>"
        return text


class current_model_values():
    """Convenience class that makes use of __repr__ methods for nice printing in the notebook
    of the properties of parameters in components in a model 
     
    Parameters
    ----------
    component : hyperspy component instance
    only_free : bool, default False
        If True: Only include the free parameters in the view
    only_active : bool, default False
        If True: Only include active parameters in the view
    """

    def __init__(self, model, only_free=False, only_active=False, component_list=None):
        self.model = model
        self.only_free = only_free
        self.only_active = only_active
        self.component_list = model if component_list == None else component_list
        self.model_type = str(self.model.__class__).split("'")[1].split('.')[-1]

    def __repr__(self):
        text = "{}: {}\n".format(
            self.model_type, self.model.signal.metadata.General.title)
        for comp in self.component_list:
            if not self.only_active or self.only_active and comp.active:
                if not self.only_free or comp.free_parameters and self.only_free:
                    text += current_component_values(
                        component=comp, 
                        only_free=self.only_free, 
                        only_active=self.only_active
                        ).__repr__() + "\n"
        return text

    def _repr_html_(self):

        html = "<h4>{}: {}</h4>".format(self.model_type,
                                        self.model.signal.metadata.General.title)
        for comp in self.component_list:
            if not self.only_active or self.only_active and comp.active:
                if not self.only_free or comp.free_parameters and self.only_free:
                    html += current_component_values(
                        component=comp, 
                        only_free=self.only_free, 
                        only_active=self.only_active
                        )._repr_html_()
        return html
