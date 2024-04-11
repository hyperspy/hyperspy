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

"""Common docstring snippets for signal1d."""

CROP_PARAMETER_DOC = """crop : bool
            If True automatically crop the signal axis at both ends if needed."""

SPIKES_DIAGNOSIS_DOCSTRING = """Plots a histogram to help in choosing the threshold for
        spikes removal.

        Parameters
        ----------
        signal_mask: boolean array
            Restricts the operation to the signal locations not marked
            as True (masked)
        navigation_mask: boolean array
            Restricts the operation to the navigation locations not
            marked as True (masked).
        %s
        **kwargs : dict
            Keyword arguments pass to
            :meth:`~hyperspy.api.signals.BaseSignal.get_histogram`

        See Also
        --------
        spikes_removal_tool

        """

SPIKES_REMOVAL_TOOL_DOCSTRING = """Graphical interface to remove spikes from EELS spectra or
        luminescence data.
        If non-interactive, it removes all spikes.

        Parameters
        ----------
        %s
        %s
        threshold : ``'auto'`` or int
            if ``int`` set the threshold value use for the detecting the spikes.
            If ``"auto"``, determine the threshold value as being the first zero
            value in the histogram obtained from the
            :meth:`~hyperspy.api.signals.Signal1D.spikes_diagnosis`
            method.
        %s
        interactive : bool
            If True, remove the spikes using the graphical user interface.
            If False, remove all the spikes automatically, which can
            introduce artefacts if used with signal containing peak-like
            features. However, this can be mitigated by using the
            ``signal_mask`` argument to mask the signal of interest.
        %s
        %s
        **kwargs : dict
            Keyword arguments pass to ``SpikesRemoval``.

        See Also
        --------
        :meth:`~hyperspy.api.signals.Signal1D.spikes_diagnosis`

        """

MASK_ZERO_LOSS_PEAK_WIDTH = """zero_loss_peak_mask_width : None or float
            If None, the zero loss peak is not masked, otherwise, use the
            provided value as width of the zero loss peak mask.
            Default is None."""
