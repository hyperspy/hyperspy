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

import traits.api as t
import traitsui.api as tu
from traitsui.menu import OKButton, CancelButton


class MessageHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        if is_ok is True:
            info.object.is_ok = True
        else:
            info.object.is_ok = False
        return True

information_view = tu.View(tu.Group(
    tu.Item('text',
            show_label=False,
            style='readonly',
            springy=True,
            width=300,
            padding=15),),
    kind='modal',
    buttons=[OKButton, CancelButton],
    handler=MessageHandler,
    title='Message')


class Message(t.HasTraits):
    text = t.Str
    is_ok = t.Bool(False)

    def __init__(self, text):
        self.text = text
    traits_view = information_view


class Options(t.HasTraits):
    options = t.Enum(('a'))

    def __init__(self, options=None):
        if not options:
            options = ['a', 'b', 'c']
        self.options = options


class MessageWithOptions(Message, Options):

    def __init__(self, text, options):
        Message.__init__(self, text)
        Options.__init__(self, options)


def information(text):
    message = Message(text)
    message.text = text
    message.edit_traits()
    return message.is_ok


def options(options_):
    class Options(t.HasTraits):
        options = t.Enum(options_)
    return Options
