# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

from packaging.version import Version
from tqdm import __version__ as tqdm_version

if Version(tqdm_version) >= Version("4.36.0"):
    # API change for 5.0 https://github.com/tqdm/tqdm/pull/800
    from tqdm import tqdm
    from tqdm.notebook import tqdm as tqdm_notebook
else:
    from tqdm import tqdm, tqdm_notebook

from hyperspy.defaults_parser import preferences


def progressbar(*args, **kwargs):
    """Uses tqdm progressbar. This function exists for wrapping purposes only.

    Original docstring follows:
    ---------------------------
    %s
    %s
    """
    if preferences.General.nb_progressbar:
        try:
            return tqdm_notebook(*args, **kwargs)
        except:
            pass
    return tqdm(*args, **kwargs)
progressbar.__doc__ %= (tqdm.__doc__, tqdm.__init__.__doc__)
