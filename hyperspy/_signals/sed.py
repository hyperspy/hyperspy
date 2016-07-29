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
from __future__ import division

import traits.api as t
import math
import numpy as np
import scipy.ndimage as ndi

from hyperspy._signals.signal2d import Signal2D
from hyperspy.decorators import only_interactive
from hyperspy.gui.sed import SEDParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui


class SEDPattern(Signal2D):
    _signal_type = "SED_Pattern"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        # Attributes defaults
        if 'Acquisition_instrument.TEM' not in self.metadata:
            if 'Acquisition_instrument.SEM' in self.metadata:
                self.metadata.set_item(
                    "Acquisition_instrument.TEM",
                    self.metadata.Acquisition_instrument.SEM)
                del self.metadata.Acquisition_instrument.SEM
        self._set_default_param()

    def _set_default_param(self):
        """Set microscope parameters to default values (defined in preferences)
        """

        md = self.metadata
        md.Signal.signal_type = 'SED_Pattern'

        if "Acquisition_instrument.TEM.beam_energy" not in md:
            md.set_item("Acquisition_instrument.TEM.beam_energy",
                        preferences.SED.sed_beam_energy)
        if "Acquisition_instrument.TEM.camera_length" not in md:
            md.set_item("Acquisition_instrument.TEM.camera_length",
                        preferences.SED.sed_camera_length)
        if "Acquisition_instrument.TEM.scan_rotation" not in md:
            md.set_item("Acquisition_instrument.TEM.scan_rotation",
                        preferences.SED.sed_scan_rotation)
        if "Acquisition_instrument.TEM.convergence_angle" not in md:
            md.set_item("Acquisition_instrument.TEM.convergence_angle",
                        preferences.SED.sed_convergence_angle)
        if "Acquisition_instrument.TEM.precession_angle" not in md:
            md.set_item("Acquisition_instrument.TEM.precession_angle",
                        preferences.SED.sed_precession_angle)
        if "Acquisition_instrument.TEM.precession_frequency" not in md:
            md.set_item("Acquisition_instrument.TEM.precession_frequency",
                        preferences.SED.sed_precession_frequency)
        if "Acquisition_instrument.TEM.Detector.SED.exposure_time" not in md:
            md.set_item("Acquisition_instrument.TEM.Detector.SED.exposure_time",
                        preferences.SED.sed_exposure_time)

    def set_microscope_parameters(self,
                                  beam_energy=None,
                                  camera_length=None,
                                  scan_rotation=None,
                                  convergence_angle=None,
                                  precession_angle=None,
                                  precession_frequency=None,
                                  exposure_time=None):
        """Set the microscope parameters.

        If no arguments are given, raises an interactive mode to enter values.

        Parameters
        ----------
        beam_energy: float
            The energy of the electron beam in keV
        camera_length: float
            Camera length in m
        scan_rotation: float
            Scan rotation in degrees
        convergence_angle : float
            Convergence angle in mrad
        precession_angle : float
            Precession angle in mrad
        precession_frequency : float
            Precession frequency in Hz
        exposure_time : float
            Exposure time in ms.

        Examples
        --------
        >>> dp = hs.datasets.example_signals.SED_Pattern()
        >>> print(dp.metadata.Acquisition_instrument.TEM.precession_angle)
        >>> dp.set_microscope_parameters(precession_angle=36.)
        >>> print(dp.metadata.Acquisition_instrument.TEM.precession_angle)
        18.0
        36.0
        """
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.TEM.beam_energy",
                        beam_energy)
        if camera_length is not None:
            md.set_item("Acquisition_instrument.TEM.camera_length",
                        camera_length)
        if scan_rotation is not None:
            md.set_item("Acquisition_instrument.TEM.scan_rotation",
                        scan_rotation)
        if convergence_angle is not None:
            md.set_item("Acquisition_instrument.TEM.convergence_angle",
                        convergence_angle)
        if precession_angle is not None:
            md.set_item("Acquisition_instrument.TEM.precession_angle",
                        precession_angle)
        if precession_frequency is not None:
            md.set_item("Acquisition_instrument.TEM.precession_frequency",
                        precession_frequency)
        if exposure_time is not None:
            md.set_item("Acquisition_instrument.TEM.Detector.SED.exposure_time",
                        exposure_time)

        if set([beam_energy, camera_length, scan_rotation,
                convergence_angle, precession_angle, precession_frequency,
                exposure_time]) == {None}:
            self._are_microscope_parameters_missing()

    @only_interactive
    def _set_microscope_parameters(self):
        sed_par = SEDParametersUI()
        mapping = {
            'Acquisition_instrument.TEM.beam_energy':
            'sed_par.beam_energy',
            'Acquisition_instrument.TEM.camera_length':
            'sed_par.camera_length',
            'Acquisition_instrument.TEM.scan_rotation':
            'sed_par.scan_rotation',
            'Acquisition_instrument.TEM.beam_energy':
            'sed_par.beam_energy',
            'Acquisition_instrument.TEM.convergence_angle':
            'sed_par.convergence_angle',
            'Acquisition_instrument.TEM.precession_angle':
            'sed_par.precession_angle',
            'Acquisition_instrument.TEM.precession_frequency':
            'sed_par.precession_frequency',
            'Acquisition_instrument.TEM.Detector.SED.exposure_time':
            'sed_par.exposure_time', }
        for key, value in mapping.items():
            if self.metadata.has_item(key):
                exec('%s = self.metadata.%s' % (value, key))
        sed_par.edit_traits()

        mapping = {
            'Acquisition_instrument.TEM.beam_energy':
            sed_par.beam_energy,
            'Acquisition_instrument.TEM.camera_length':
            sed_par.camera_length,
            'Acquisition_instrument.TEM.scan_rotation':
            sed_par.scan_rotation,
            'Acquisition_instrument.TEM.convergence_angle':
            sed_par.convergence_angle,
            'Acquisition_instrument.TEM.precession_angle':
            sed_par.precession_angle,
            'Acquisition_instrument.TEM.precession_frequency':
            sed_par.precession_frequency,
            'Acquisition_instrument.TEM.Detector.SED.exposure_time':
            sed_par.exposure_time, }

        for key, value in mapping.iteritems():
            if value != t.Undefined:
                self.metadata.set_item(key, value)
        self._are_microscope_parameters_missing()

    def _are_microscope_parameters_missing(self):
        """Check that the SED parameters necessary for pattern calibration are
        defined in metadata and raise a UI if not to add them."""
        must_exist = ('Acquisition_instrument.TEM.beam_energy',
                      'Acquisition_instrument.TEM.camera_length',
                      'Acquisition_instrument.TEM.scan_rotation')

        missing_parameters = []
        for item in must_exist:
            exists = self.metadata.has_item(item)
            if exists is False:
                missing_parameters.append(item)
        if missing_parameters:
            if preferences.General.interactive is True:
                par_str = "The following parameters are missing:\n"
                for par in missing_parameters:
                    par_str += '%s\n' % par
                par_str += 'Please set them in the following wizard'
                is_ok = messagesui.information(par_str)
                if is_ok:
                    self._set_microscope_parameters()
                else:
                    return True
            else:
                return True
        else:
            return False

    def _get_direct_beam_position(self, z, center=None, radius=None,
                                  subpixel=None):
        """Refine the position of the direct beam and hence an estimate for the
        position of the pattern center in each SED pattern.

        Parameters
        ----------
        radius : int
            Defines the size of the circular region within which the direct beam
            position is refined.
        subpixel : bool
            If True the direct beam position is refined to sub-pixel precision
            via calculation of the intensity center of mass.

        Return
        ------
        center: array
            Refined position (x, y) of the direct beam.

        Notes
        -----
        This method is based on work presented by Thomas White in his PhD (2009)
        which itself built on Zaefferer (2000).
        """
        # initialise problem with initial center estimate
        c_int = z[center[0], center[1]]
        ny = z.shape[1]
        nx = z.shape[0]
        y, x = np.ogrid[-center[0]:ny-center[0], -center[1]:nx-center[1]]
        mask = x * x + y * y <= radius * radius
        z_tmp = z * mask
        # refine center position to pixel level precision via optimisation of
        # ROI
        while c_int < z_tmp.max():
            maxes = np.asarray(np.where(z_tmp == z_tmp.max()))
            center = np.rint([np.average(maxes[0]), np.average(maxes[1])])
            center = center.astype(int)
            c_int = z[center[0], center[1]]
            y, x = np.ogrid[-center[0]:ny-center[0], -center[1]:nx-center[1]]
            mask = x * x + y * y <= radius * radius
            z_tmp = z * mask
        # refine center value to sub-pixel precision by evaluating intensity
        # centre of mass.
        if subpixel is True:
            center = np.asarray(ndi.measurements.center_of_mass(z_tmp))

        return center

    def direct_beam_shifts(self, radius=10, subpixel=False):
        """Determine rigid shifts in the SED patterns based on the position of
        the direct beam and return the shifts required to center all patterns.

        Parameters
        ----------
        radius : int
            Defines the size of the circular region within which the direct beam
            position is refined.
        subpixel : bool
            If True the direct beam position is refined to sub-pixel precision
            via calculation of the intensity center_of_mass.

        Returns
        -------
        shifts : array
            Array containing the shift to be applied to each SED pattern to
            center it.

        See also
        --------
        _get_direct_beam_position
        """
        # sum images to produce image in which direct beam reinforced and take
        # the position of maximum intensity as the initial estimate of center.
        dp_sum = self.sum()
        max_ref = np.asarray(np.where(dp_sum.data == dp_sum.data.max()))
        c_ref = np.rint([np.average(max_ref[0]), np.average(max_ref[1])])
        c_ref = c_ref.astype(int)
        # specify array of dims (nav_size, 2) in which to store centers and find
        # the center of each pattern by determining the direct beam position.
        arr_shape = (self.axes_manager.navigation_size, 2)
        centers = np.zeros(arr_shape, dtype=int)
        for z, index in zip(self._iterate_signal(),
                            np.arange(0,
                                      self.axes_manager.navigation_size,
                                      1)):
            centers[index] = self._get_direct_beam_position(z, center=c_ref,
                                                            radius=radius,
                                                            subpixel=subpixel)
        # calculate shifts to align all patterns to the reference position
        shifts = centers - [self.axes_manager.signal_shape[0] / 2,
                            self.axes_manager.signal_shape[1] / 2]

        return shifts

    def direct_beam_mask(self, radius, center=None):
        """Generate a signal mask for the direct beam.

        Parameters
        ----------
        radius : int
            User specified radius for the circular mask.
        center : tuple, None
            User specified (x, y) position of the diffraction pattern center.
            i.e. the direct beam position. If None it is assumed that the direct
            beam is at the center of the diffraction pattern.

        Return
        ------
        mask : array
            The mask of the direct beam
        """
        r = radius

        if center = None:
            ny = self.axes_manager.signal_shape[1] / 2
            nx = self.axes_manager.signal_shape[0] / 2
        else:
            nx, ny = center

        y, x = np.ogrid[-ny:ny, -nx:nx]
        mask = x*x + y*y <= r*r
        return mask

    def vacuum_mask(self, radius, center=None,
                    threshold, closing=True, opening=False):
        """Generate a navigation mask to exlude SED patterns acquired in vacuum.

        Parameters
        ----------
        radius: int
            Radius of circular mask to exclude direct beam.
        center : tuple, None
            User specified position of the diffraction pattern center. If None
            it is assumed that the pattern center is the center of the image.
        threshold : float
            Minimum intensity required to consider a diffracted beam to be
            present.

        Returns
        -------
        mask : signal
            The mask of the region of interest.
        """
        db = np.invert(self.direct_beam_mask(radius=radius, center=center))
        diff_only = self * db
        mask = (diff_only.max((-1, -2)) <= threshold)
        if closing:
            mask.data = ndi.morphology.binary_dilation(mask.data,
                                                       border_value=0)
            mask.data = ndi.morphology.binary_erosion(mask.data,
                                                      border_value=1)
        if opening:
            mask.data = ndi.morphology.binary_erosion(mask.data,
                                                      border_value=1)
            mask.data = ndi.morphology.binary_dilation(mask.data,
                                                       border_value=0)
        return mask

    def decomposition(self,
                      normalize_poissonian_noise=True,
                      signal_mask=None,
                      center=None,
                      navigation_mask=None,
                      threshold=None,
                      closing=True,
                      *args,
                      **kwargs):
        """Decomposition with a choice of algorithms.

        The results are stored in self.learning_results

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
        direct_beam_mask : None or float or boolean numpy array
            The navigation locations marked as True are not used in the
            decompostion. If float is given the direct_beam_mask method is used
            to generate a mask with the float value as radius.
        closing: bool
            If true, applied a morphologic closing to the maks obtained by
            vacuum_mask.
        algorithm : 'svd' | 'fast_svd' | 'mlpca' | 'fast_mlpca' | 'nmf' |
            'sparse_pca' | 'mini_batch_sparse_pca'
        output_dimension : None or int
            number of components to keep/calculate
        centre : None | 'variables' | 'trials'
            If None no centring is applied. If 'variable' the centring will be
            performed in the variable axis. If 'trials', the centring will be
            performed in the 'trials' axis. It only has effect when using the
            svd or fast_svd algorithms
        auto_transpose : bool
            If True, automatically transposes the data to boost performance.
            Only has effect when using the svd of fast_svd algorithms.
        signal_mask : boolean numpy array
            The signal locations marked as True are not used in the
            decomposition.
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm
        var_func : function or numpy array
            If function, it will apply it to the dataset to obtain the
            var_array. Alternatively, it can a an array with the coefficients
            of a polynomial.
        polyfit :
        reproject : None | signal | navigation | both
            If not None, the results of the decomposition will be projected in
            the selected masked area.

        Examples
        --------
        >>> dp = hs.datasets.example_signals.SED_Pattern()
        >>> dps = hs.stack([s]*3)
        >>> dps.change_dtype(float)
        >>> dps.decomposition()

        See also
        --------
        direct_beam_mask
        vacuum_mask
        """
        if isinstance(signal_mask, float):
            signal_mask = self.direct_beam_mask(signal_mask, center)
        if isinstance(navigation_mask, float):
            navigation_mask = self.vacuum_mask(navigation_mask,
                                               center, threshold).data
        super(Signal2D, self).decomposition(
            normalize_poissonian_noise=normalize_poissonian_noise,
            signal_mask=signal_mask, navigation_mask=navigation_mask,
            *args, **kwargs)
        self.learning_results.loadings = np.nan_to_num(
            self.learning_results.loadings)
