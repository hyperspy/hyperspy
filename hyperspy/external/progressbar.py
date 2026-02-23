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

import tqdm

from hyperspy.defaults_parser import preferences


def progressbar(*args, **kwargs):
    """Uses tqdm progressbar. This function exists for wrapping purposes only.

    Original docstring follows:
    ---------------------------
    %s
    %s
    """
    if preferences.General.nb_progressbar:
        # use tqdm.auto to use tqdm.std in terminal and
        # tqdm.notebook in a jupyter environment.
        from tqdm.auto import tqdm

        return tqdm(*args, **kwargs)

    # use tqdm.std all the time, even in a jupyter environment.
    return tqdm.tqdm(*args, **kwargs)

progressbar.__doc__ %= (tqdm.__doc__, tqdm.__init__.__doc__)
