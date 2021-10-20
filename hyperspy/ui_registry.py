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

"""Registry of user interface widgets.

Format {"tool_key" : {"toolkit" : <function(obj, display, \*\*kwargs)>}}

The ``tool_key`` is defined by the "model function" to which the widget provides
and user interface. That function gets the widget function from this registry
and executes it passing the ``obj``, ``display`` and any extra keyword
arguments. When ``display`` is true, ``function`` displays the widget. If
``False`` it returns a dictionary with whatever is needed to display the
widgets externally (usually for testing or customisation purposes).

"""

import importlib

from hyperspy.misc.utils import isiterable
from hyperspy.extensions import ALL_EXTENSIONS


UI_REGISTRY = {toolkey: {} for toolkey in ALL_EXTENSIONS["GUI"]["toolkeys"]}

TOOLKIT_REGISTRY = set()
KNOWN_TOOLKITS = set(("ipywidgets", "traitsui"))


if "widgets" in ALL_EXTENSIONS["GUI"] and ALL_EXTENSIONS["GUI"]["widgets"]:
    for toolkit, widgets in ALL_EXTENSIONS["GUI"]["widgets"].items():
        TOOLKIT_REGISTRY.add(toolkit)
        for toolkey, specs in widgets.items():
            if not toolkey in UI_REGISTRY:
                raise NameError("%s is not a registered toolkey" % toolkey)
            UI_REGISTRY[toolkey][toolkit] = specs


def _toolkits_to_string(toolkits):
    if isinstance(toolkits, str):
        return f"{toolkits} toolkit"
    else:
        toolkits = tuple(toolkits)
        if len(toolkits) == 1:
            return f"{toolkits[0]} toolkit"

        elif len(toolkits) == 2:
            return " and ".join(toolkits) + " toolkits"
        else:  # > 2
            txt = ", ".join(toolkits[:-1])
            return f"{txt} and {toolkits[-1]} toolkits"


def get_gui(self, toolkey, display=True, toolkit=None, **kwargs):
    if not TOOLKIT_REGISTRY:
        raise ImportError(
            "No toolkit registered. Install hyperspy_gui_ipywidgets or "
            "hyperspy_gui_traitsui GUI elements."
        )
    from hyperspy.defaults_parser import preferences
    if isinstance(toolkit, str):
        toolkit = (toolkit,)
    if isiterable(toolkit):
        toolkits = set()
        for tk in toolkit:
            if tk in TOOLKIT_REGISTRY:
                toolkits.add(tk)
            else:
                raise ValueError(f"{tk} is not a registered toolkit.")
    elif toolkit is None:
        toolkits = set()
        available_disabled_toolkits = set()
        if "ipywidgets" in TOOLKIT_REGISTRY:
            if preferences.GUIs.enable_ipywidgets_gui:
                toolkits.add("ipywidgets")
            else:
                available_disabled_toolkits.add("ipywidgets")
        if "traitsui" in TOOLKIT_REGISTRY:
            if preferences.GUIs.enable_traitsui_gui:
                toolkits.add("traitsui")
            else:
                available_disabled_toolkits.add("traitsui")
        if not toolkits and available_disabled_toolkits:
            is_or_are = "is" if len(
                available_disabled_toolkits) == 1 else "are"
            them_or_it = ("it" if len(available_disabled_toolkits) == 1
                          else "them")
            raise ValueError(
                "No toolkit available. The "
                f"{_toolkits_to_string(available_disabled_toolkits)} "
                f"{is_or_are} installed but disabled in `preferences`. "
                f"Enable {them_or_it} in `preferences` or "
                "manually select a toolkit with the `toolkit` argument."
            )

    else:
        raise ValueError(
            "`toolkit` must be a string, an iterable of strings or None.")
    if toolkey not in UI_REGISTRY or not UI_REGISTRY[toolkey]:
        propose = KNOWN_TOOLKITS - TOOLKIT_REGISTRY
        if propose:
            propose = [f"hyperspy_gui_{tk}" for tk in propose]
            if len(propose) > 1:
                propose_ = ", ".join(propose[:-1])
                propose = f"{propose_} and/or {propose[-1]}"
            else:
                propose = propose.pop()
        raise NotImplementedError(
            "There is no user interface registered for this feature."
            f"Try installing {propose}."
        )
    if not display:
        widgets = {}
    available_toolkits = set()
    used_toolkits = set()
    for toolkit, specs in UI_REGISTRY[toolkey].items():
        f = getattr(
            importlib.import_module(
                specs["module"]),
            specs["function"])
        if toolkit in toolkits:
            used_toolkits.add(toolkit)
            try:
                thisw = f(obj=self, display=display, **kwargs)
            except NotImplementedError as e:
                # traitsui raises this exception when the backend is
                # not supported
                if toolkit == "traitsui":
                    pass
                else:
                    raise e
            if not display:
                widgets[toolkit] = thisw
        else:
            available_toolkits.add(toolkit)
    if not used_toolkits and available_toolkits:
        is_or_are = "is" if len(toolkits) == 1 else "are"
        raise NotImplementedError(
            f"The {_toolkits_to_string(toolkits)} {is_or_are} not available "
            "for this functionality, try with the "
            f"{_toolkits_to_string(available_toolkits)}."
        )
    if not display:
        return widgets


def get_partial_gui(toolkey):
    def pg(self, display=True, toolkit=None, **kwargs):
        return get_gui(self, toolkey=toolkey, display=display,
                       toolkit=toolkit, **kwargs)
    return pg


DISPLAY_DT = """display : bool
            If True, display the user interface widgets. If False, return the
            widgets container in a dictionary, usually for customisation or
            testing."""

TOOLKIT_DT = """toolkit : str, iterable of strings or None
            If None (default), all available widgets are displayed or returned.
            If string, only the widgets of the selected toolkit are displayed
            if available. If an interable of toolkit strings, the widgets of
            all listed toolkits are displayed or returned."""
GUI_DT = """Display or return interactive GUI element if available.

Parameters
----------
%s
%s

""" % (DISPLAY_DT, TOOLKIT_DT)


def add_gui_method(toolkey):
    def decorator(cls):
        # Not using functools.partialmethod because it is not possible to set
        # the docstring that way.
        setattr(cls, "gui", get_partial_gui(toolkey))
        setattr(cls.gui, "__doc__", GUI_DT)
        return cls
    return decorator
