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

"""HyperSpy-specific additions to anyplotlib's rosettasciio shim.

Spliced into the rsciio block of anywidget_bridge.js by
hyperspy_anywidget._patch_bridge and executed inside Pyodide, where sys,
_types, _ru (the rsciio.utils module object) and _rsciio are already bound.

Upstream stubs rsciio.utils.path and rsciio.utils.rgb.  HyperSpy's lazy
signals also do "from rsciio.utils import file" at module import time, which
is reached as soon as an example calls map() with lazy output -- so without
this, any such example dies with ImportError before it draws anything.

Only get_file_handle is ever called (LazySignal.close_file), and only to close
a handle belonging to an h5py-backed dask array.  Nothing in Pyodide has one,
and close_file already catches AttributeError, so returning None is both
correct and the documented no-op path.

Style constraint: no backticks and no backslashes anywhere in this file, and
no JavaScript template-literal placeholders (a dollar sign followed by a
brace).  hyperspy_anywidget enforces this at build time.
"""


def _hs_get_file_handle(data, warn=True):
    """No file-backed dask arrays exist in Pyodide, so there is no handle."""
    return None


def _hs_memmap_distributed(*args, **kwargs):
    raise NotImplementedError("memmap_distributed is unavailable in Pyodide")


def _hs_inspect_npy_bytes(*args, **kwargs):
    raise NotImplementedError("inspect_npy_bytes is unavailable in Pyodide")


_rf = _types.ModuleType("rsciio.utils.file")  # noqa: F821
_rf.get_file_handle = _hs_get_file_handle
_rf.memmap_distributed = _hs_memmap_distributed
_rf.inspect_npy_bytes = _hs_inspect_npy_bytes

_ru.file = _rf  # noqa: F821
sys.modules["rsciio.utils.file"] = _rf  # noqa: F821


# --- Let the real prettytable through ---------------------------------------
#
# Upstream pre-empts prettytable with a stub whose __str__ returns a list repr
# and which has no .align.  HyperSpy formats model parameters through it
# (misc/_utils.py -> print), so every model example died with
# "'_PrettyTable' object has no attribute 'align'".  prettytable is a small
# pure-Python wheel and a genuine HyperSpy dependency, so dropping the stub
# here -- before the wheel install -- lets micropip fetch the real one.
# It must correspondingly be absent from DEFAULT_MOCK_PACKAGES.
sys.modules.pop("prettytable", None)  # noqa: F821

print("[hyperspy] rsciio.utils.file shim installed; real prettytable allowed")
