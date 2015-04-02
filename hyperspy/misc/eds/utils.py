from __future__ import division

import numpy as np
import math

from hyperspy.misc.elements import elements as elements_db


def _get_element_and_line(Xray_line):
    lim = Xray_line.find('_')
    return Xray_line[:lim], Xray_line[lim + 1:]


def _get_energy_xray_line(xray_line):
    energy, line = _get_element_and_line(xray_line)
    return elements_db[energy]['Atomic_properties']['Xray_lines'][
        line]['energy (keV)']


def _get_xray_lines_family(xray_line):
    return xray_line[:xray_line.find('_') + 2]


def get_FWHM_at_Energy(energy_resolution_MnKa, E):
    """Calculates the FWHM of a peak at energy E.

    Parameters
    ----------
    energy_resolution_MnKa : float
        Energy resolution of Mn Ka in eV
    E : float
        Energy of the peak in keV

    Returns
    -------
    float : FWHM of the peak in keV

    Notes
    -----
    From the textbook of Goldstein et al., Plenum publisher,
    third edition p 315

    """
    FWHM_ref = energy_resolution_MnKa
    E_ref = _get_energy_xray_line('Mn_Ka')

    FWHM_e = 2.5 * (E - E_ref) * 1000 + FWHM_ref * FWHM_ref

    return math.sqrt(FWHM_e) / 1000  # In mrad


def xray_range(xray_line, beam_energy, density='auto'):
    '''Return the Anderson-Hasler X-ray range.

    Return the maximum range of X-ray generation in a pure bulk material.

    Parameters
    ----------
    xray_line: str
        The X-ray line, e.g. 'Al_Ka'
    beam_energy: float
        The energy of the beam in kV.
    density: {float, 'auto'}
        The density of the material in g/cm3. If 'auto', the density
        of the pure element is used.

    Returns
    -------
    X-ray range in micrometer.

    Notes
    -----
    From Anderson, C.A. and M.F. Hasler (1966). In proceedings of the
    4th international conference on X-ray optics and microanalysis.

    See also the textbook of Goldstein et al., Plenum publisher,
    third edition p 286

    '''

    element, line = _get_element_and_line(xray_line)
    if density == 'auto':
        density = elements_db[
            element][
            'Physical_properties'][
            'density (g/cm^3)']
    Xray_energy = _get_energy_xray_line(xray_line)

    return 0.064 / density * (np.power(beam_energy, 1.68) -
                              np.power(Xray_energy, 1.68))


def electron_range(element, beam_energy, density='auto', tilt=0):
    '''Return the Kanaya-Okayama electron range.

    Return the maximum electron range in a pure bulk material.

    Parameters
    ----------
    element: str
        The element symbol, e.g. 'Al'.
    beam_energy: float
        The energy of the beam in keV.
    density: {float, 'auto'}
        The density of the material in g/cm3. If 'auto', the density of
        the pure element is used.
    tilt: float.
        The tilt of the sample in degrees.

    Returns
    -------
    Electron range in micrometers.

    Notes
    -----
    From Kanaya, K. and S. Okayama (1972). J. Phys. D. Appl. Phys. 5, p43

    See also the textbook of Goldstein et al., Plenum publisher,
    third edition p 72.

    '''

    if density == 'auto':
        density = elements_db[
            element]['Physical_properties']['density (g/cm^3)']
    Z = elements_db[element]['General_properties']['Z']
    A = elements_db[element]['General_properties']['atomic_weight']

    return (0.0276 * A / np.power(Z, 0.89) / density *
            np.power(beam_energy, 1.67) * math.cos(math.radians(tilt)))


def take_off_angle(tilt_stage,
                   azimuth_angle,
                   elevation_angle):
    """Calculate the take-off-angle (TOA).

    TOA is the angle with which the X-rays leave the surface towards
    the detector.

    Parameters
    ----------
    tilt_stage: float
        The tilt of the stage in degrees. The sample is facing the detector when
        positively tilted.
    azimuth_angle: float
        The azimuth of the detector in degrees. 0 is perpendicular to the tilt
        axis.
    elevation_angle: float
        The elevation of the detector in degrees.

    Returns
    -------
    take_off_angle: float.
        In degrees.

    Notes
    -----
    Defined by M. Schaffer et al., Ultramicroscopy 107(8), pp 587-597 (2007)

    """

    a = math.radians(90 + tilt_stage)
    b = math.radians(azimuth_angle)
    c = math.radians(elevation_angle)

    return math.degrees(np.arcsin(-math.cos(a) * math.cos(b) * math.cos(c)
                                  + math.sin(a) * math.sin(c)))


def xray_lines_model(elements=['Al', 'Zn'],
                     beam_energy=200,
                     weight_percents=[50, 50],
                     energy_resolution_MnKa=130,
                     energy_axis={'name': 'E', 'scale': 0.01, 'units': 'keV',
                                  'offset': -0.1, 'size': 1024}
                     ):
    """
    Generate a model of X-ray lines using a Gaussian epr x-ray lines.

    The area under a main peak (alpha) is equal to 1 and weighted by the
    composition.

    Parameters
    ----------
    elements : list of strings
        A list of chemical element symbols.
    beam_energy: float
        The energy of the beam in keV.
    weight_percents: list of float
        The composition in weight percent.
    energy_resolution_MnKa: float
        The energy resolution of the detector in eV
    energy_axis: dic
        The dictionary for the energy axis. It must contains 'size' and the
        units must be 'eV' of 'keV'.

    Example
    -------
    >>> s = utils_eds.simulate_model(['Cu', 'Fe'], beam_energy=30)
    >>> s.plot()
    """
    from hyperspy._signals.eds_tem import EDSTEMSpectrum
    from hyperspy.model import Model
    from hyperspy import components
    s = EDSTEMSpectrum(np.zeros(energy_axis['size']), axes=[energy_axis])
    s.set_microscope_parameters(
        beam_energy=beam_energy,
        energy_resolution_MnKa=energy_resolution_MnKa)
    s.add_elements(elements)
    counts_rate = 1.
    live_time = 1.
    if weight_percents is None:
        weight_percents = [100] * len(elements)
    m = Model(s)
    for i, (element, weight_percent) in enumerate(zip(
            elements, weight_percents)):
        for line in elements_db[
                element]['Atomic_properties']['Xray_lines'].keys():
            line_energy = elements_db[element]['Atomic_properties'][
                'Xray_lines'][line]['energy (keV)']
            ratio_line = elements_db[element]['Atomic_properties'][
                'Xray_lines'][line]['weight']
            if s._get_xray_lines_in_spectral_range(
                    [element+'_'+line])[1] == []:
                g = components.Gaussian()
                g.centre.value = line_energy
                g.sigma.value = get_FWHM_at_Energy(
                    energy_resolution_MnKa, line_energy) / 2.355
                g.A.value = live_time * counts_rate * \
                    weight_percent / 100 * ratio_line
                m.append(g)
    s.data = m.as_signal().data
    # s.add_poissonian_noise()
    return s


def quantification_cliff_lorimer(intensities,
                                 kfactors,
                                 mask=None):
    """
    Quantification using Cliff-Lorimer

    Parameters
    ----------
    intensities: numpy.array
        the intensities for each X-ray lines. The first axis should be the
        elements axis.
    kfactors: list of float
        The list of kfactor in same order as intensities eg. kfactors =
        [1, 1.47, 1.72] for ['Al_Ka','Cr_Ka', 'Ni_Ka']
    mask: array of bool
        The mask with the dimension of intensities[0]. If a pixel is True,
        the composition is set to zero.

    Return
    ------
    numpy.array containing the weight fraction with the same
    shape as intensities.
    """
    # Value used as an threshold to prevent using zeros as denominator
    min_intensity = 0.1
    dim = intensities.shape
    if len(dim) > 1:
        dim2 = reduce(lambda x, y: x*y, dim[1:])
        intens = intensities.reshape(dim[0], dim2)
        intens = intens.astype('float')
        for i in range(dim2):
            index = np.where(intens[:, i] > min_intensity)[0]
            if len(index) > 1:
                ref_index, ref_index2 = index[:2]
                intens[:, i] = _quantification_cliff_lorimer(
                    intens[:, i], kfactors, ref_index, ref_index2)
            else:
                intens[:, i] = np.zeros_like(intens[:, i])
                if len(index) == 1:
                    intens[index[0], i] = 1.
        intens = intens.reshape(dim)
        if mask is not None:
            for i in range(dim[0]):
                intens[i][mask] = 0
        return intens
    else:
        # intens = intensities.copy()
        # intens = intens.astype('float')
        index = np.where(intensities > min_intensity)[0]
        if len(index) > 1:
            ref_index, ref_index2 = index[:2]
            intens = _quantification_cliff_lorimer(
                intensities, kfactors, ref_index, ref_index2)
        else:
            intens = np.zeros_like(intensities)
            if len(index) == 1:
                intens[index[0]] = 1.
        return intens


def _quantification_cliff_lorimer(intensities,
                                  kfactors,
                                  ref_index=0,
                                  ref_index2=1):
    """
    Quantification using Cliff-Lorimer

    Parameters
    ----------
    intensities: numpy.array
        the intensities for each X-ray lines. The first axis should be the
        elements axis.
    kfactors: list of float
        The list of kfactor in same order as  intensities eg. kfactors =
        [1, 1.47, 1.72] for ['Al_Ka','Cr_Ka', 'Ni_Ka']
    ref_index, ref_index2: int
        index of the elements that will be in the denominator. Should be non
        zeros if possible.

    Return
    ------
    numpy.array containing the weight fraction with the same
    shape as intensities.
    """
    if len(intensities) != len(kfactors):
        raise ValueError('The number of kfactors must match the size of the '
                         'first axis of intensities.')
    ab = np.zeros_like(intensities, dtype='float')
    composition = np.ones_like(intensities, dtype='float')
    # ab = Ia/Ib / kab

    other_index = range(len(kfactors))
    other_index.pop(ref_index)
    for i in other_index:
        ab[i] = intensities[ref_index] * kfactors[ref_index]  \
            / intensities[i] / kfactors[i]
    # Ca = ab /(1 + ab + ab/ac + ab/ad + ...)
    for i in other_index:
        if i == ref_index2:
            composition[ref_index] += ab[ref_index2]
        else:
            composition[ref_index] += (ab[ref_index2] / ab[i])
    composition[ref_index] = ab[ref_index2] / composition[ref_index]
    # Cb = Ca / ab
    for i in other_index:
        composition[i] = composition[ref_index] / ab[i]
    return composition
