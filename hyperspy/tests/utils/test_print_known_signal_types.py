# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

try:
    from prettytable import TableStyle

    MARKDOWN = TableStyle.MARKDOWN
except ImportError:
    # Deprecated in prettytable 3.12.0
    from prettytable import MARKDOWN

from hyperspy.utils import print_known_signal_types


def test_text_output(capsys):
    print_known_signal_types()
    captured = capsys.readouterr()
    assert "signal_type" in captured.out
    # the output will be str, not html
    assert "<p>" not in captured.out


def test_style(capsys):
    print_known_signal_types(style=MARKDOWN)
    captured = capsys.readouterr()

    assert "signal_type" in captured.out
    # the output will be markdown, not ascii
    assert ":--" in captured.out  # markdown
    assert "<p>" not in captured.out  # not html
    assert "+--" not in captured.out  # not ascii
