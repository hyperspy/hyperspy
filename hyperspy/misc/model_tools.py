# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

class current_component_values():
    """Convenience class that makes use of __repr__ methods for nice printing in
     the notebook"""

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
            text = "{0}: {1}".format(self._id_name, self.name)
        else:
            text = "{0}: {1}\nActive: {2}".format(
                self._id_name, self.name, self.active)
        text += "\n"
        text += signature.format("Parameter Name",
                                 "Free", "Value", "Std", "Min", "Max")
        text += "\n"
        text += signature.format("=" * size['name'], "=" * size['free'], "=" *
                                 size['value'], "=" * size['std'], "=" * size['bmin'], "=" * size['bmax'],)
        text += "\n"
        for para in self.parameters:
            if not self.only_free or self.only_free and para.free:
                if type(para.value) == tuple or type(para.value) == list:
                    std = para.std if type(para.std) == tuple or type(
                        para.std) == list else len(para.value) * ['']
                    bmin = para.bmin if type(para.bmin) == tuple or type(
                        para.bmin) == list else len(para.value) * ['']
                    bmax = para.bmax if type(para.bmax) == tuple or type(
                        para.bmax) == list else len(para.value) * ['']
                    for i, (v, s, bn, bx) in enumerate(zip(para.value, std, bmin, bmax)):
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
            text = "<p><b>{0}: {1}</b></p>".format(self._id_name, self.name)
        else:
            text = "<p><b>{0}: {1}</b><br />Active: {2}</p>".format(
                self._id_name, self.name, self.active)

        para_head = """<table style="width:100%"><tr><th>Parameter Name</th><th>Free</th>
            <th>Value</th><th>Std</th><th>Min</th><th>Max</th></tr>"""
        text += para_head
        for para in self.parameters:
            if not self.only_free or self.only_free and para.free:
                text += """<tr><td>{0}</td><td>{1}</td><td>{2}</td>
                    <td>{3}</td><td>{4}</td><td>{5}</td></tr>""".format(
                        para.name, para.free, para.value, para.std, para.bmin, para.bmax
                )
        text += "</table>"
        return text


class current_model_values():
    """Convenience class that makes use of __repr__ methods for nice printing in
     the notebook"""

    def __init__(self, model, only_free, only_active):
        self.model = model
        self.only_free = only_free
        self.only_active = only_active
        self.model_type = str(self.model.__class__).split("'")[
            1].split('.')[-1]

    def __repr__(self):
        text = "{}: {}\n".format(
            self.model_type, self.model.signal.metadata.General.title)
        for comp in self.model:
            if not self.only_active or self.only_active and comp.active:
                if not self.only_free or comp.free_parameters and self.only_free:
                    text += current_component_values(
                        component=comp, only_free=self.only_free, only_active=self.only_active).__repr__() + "\n"
        return text

    def _repr_html_(self):

        html = "<h4>{}: {}</h4>".format(self.model_type,
                                        self.model.signal.metadata.General.title)
        for comp in self.model:
            if not self.only_active or self.only_active and comp.active:
                if not self.only_free or comp.free_parameters and self.only_free:
                    html += current_component_values(
                        component=comp, only_free=self.only_free, only_active=self.only_active)._repr_html_()
        return html