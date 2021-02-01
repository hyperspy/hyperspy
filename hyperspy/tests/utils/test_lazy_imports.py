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

class TestLazyImports:

    def test_lazy_imports(self):
        import sys
        lazy_modules=[
            'pint', 'dask', 'numba', 'sklearn', 'skimage', 'sympy'
            'hyperspy-gui-ipywidgets', 'hyperspy-gui-traitsui']
        for module in list(sys.modules):
            for lazy_module in lazy_modules:
                if lazy_module in module:
                    sys.modules.pop(module)

        import hyperspy.api as hs

        modules = set(sys.modules.keys())
        for lazy_module in lazy_modules:
            assert lazy_module not in modules
