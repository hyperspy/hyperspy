
import numpy as np
import math
from scipy import constants

from hyperspy.misc.elements import elements as elements_db
from functools import reduce

eV2keV = 1000.
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))


def _get_element_and_line(xray_line):
    """
    Returns the element name and line character for a particular X-ray line as
    a tuple.

    By example, if xray_line = 'Mn_Ka' this function returns ('Mn', 'Ka')
    """
    lim = xray_line.find('_')
    if lim == -1:
        raise ValueError("Invalid xray-line: %" % xray_line)
    return xray_line[:lim], xray_line[lim + 1:]


def _get_energy_xray_line(xray_line):
    """
    Returns the energy (in keV) associated with a given X-ray line.

    By example, if xray_line = 'Mn_Ka' this function returns 5.8987
    """
    element, line = _get_element_and_line(xray_line)
    return elements_db[element]['Atomic_properties']['Xray_lines'][
        line]['energy (keV)']


def _get_xray_lines_family(xray_line):
    """
    Returns the family to which a particular X-ray line belongs.

    By example, if xray_line = 'Mn_Ka' this function returns 'Mn_K'
    """
    return xray_line[:xray_line.find('_') + 2]


def _parse_only_lines(only_lines):
    if isinstance(only_lines, str):
        pass
    elif hasattr(only_lines, '__iter__'):
        if any(isinstance(line, str) is False for line in only_lines):
            return only_lines
    else:
        return only_lines
    only_lines = list(only_lines)
    for only_line in only_lines:
        if only_line == 'a':
            only_lines.extend(['Ka', 'La', 'Ma'])
        elif only_line == 'b':
            only_lines.extend(['Kb', 'Lb1', 'Mb'])
    return only_lines


def get_xray_lines_near_energy(energy, width=0.2, only_lines=None):
    """Find xray lines near a specific energy, more specifically all xray lines
    that satisfy only_lines and are within the given energy window width around
    the passed energy.

    Parameters
    ----------
    energy : float
        Energy to search near in keV
    width : float
        Window width in keV around energy in which to find nearby energies,
        i.e. a value of 0.2 keV (the default) means to search +/- 0.1 keV.
    only_lines :
        If not None, only the given lines will be added (eg. ('a','Kb')).

    Returns
    -------
    List of xray-lines sorted by energy difference to given energy.
    """
    only_lines = _parse_only_lines(only_lines)
    valid_lines = []
    E_min, E_max = energy - width / 2., energy + width / 2.
    for element, el_props in elements_db.items():
        # Not all elements in the DB have the keys, so catch KeyErrors
        try:
            lines = el_props['Atomic_properties']['Xray_lines']
        except KeyError:
            continue
        for line, l_props in lines.items():
            if only_lines and line not in only_lines:
                continue
            line_energy = l_props['energy (keV)']
            if E_min <= line_energy <= E_max:
                # Store line in Element_Line format, and energy difference
                valid_lines.append((element + "_" + line,
                                    np.abs(line_energy - energy)))
    # Sort by energy difference, but return only the line names
    return [line for line, _ in sorted(valid_lines, key=lambda x: x[1])]


def get_FWHM_at_Energy(energy_resolution_MnKa, E):
    """Calculates an approximate FWHM, accounting for peak broadening due to the
    detector, for a peak at energy E given a known width at a reference energy.

    The factor 2.5 is a constant derived by Fiori & Newbury as references
    below.

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
    This method implements the equation derived by Fiori and Newbury as is
    documented in the following:

        Fiori, C. E., and Newbury, D. E. (1978). In SEM/1978/I, SEM, Inc.,
        AMF O'Hare, Illinois, p. 401.

        Goldstein et al. (2003). "Scanning Electron Microscopy & X-ray
        Microanalysis", Plenum, third edition, p 315.

    """
    FWHM_ref = energy_resolution_MnKa
    E_ref = _get_energy_xray_line('Mn_Ka')

    FWHM_e = 2.5 * (E - E_ref) * eV2keV + FWHM_ref * FWHM_ref

    return math.sqrt(FWHM_e) / 1000.  # In mrad


def xray_range(xray_line, beam_energy, density='auto'):
    """Return the maximum range of X-ray generation according to the
    Anderson-Hasler parameterization.

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

    Examples
    --------
    >>> # X-ray range of Cu Ka in pure Copper at 30 kV in micron
    >>> hs.eds.xray_range('Cu_Ka', 30.)
    1.9361716759499248

    >>> # X-ray range of Cu Ka in pure Carbon at 30kV in micron
    >>> hs.eds.xray_range('Cu_Ka', 30., hs.material.elements.C.
    >>>                      Physical_properties.density_gcm3)
    7.6418811280855454

    Notes
    -----
    From Anderson, C.A. and M.F. Hasler (1966). In proceedings of the
    4th international conference on X-ray optics and microanalysis.

    See also the textbook of Goldstein et al., Plenum publisher,
    third edition p 286

    """

    element, line = _get_element_and_line(xray_line)
    if density == 'auto':
        density = elements_db[
            element][
            'Physical_properties'][
            'density (g/cm^3)']
    Xray_energy = _get_energy_xray_line(xray_line)
    # Note: magic numbers here are from Andersen-Hasler parameterization. See
    # docstring for associated references.
    return 0.064 / density * (np.power(beam_energy, 1.68) -
                              np.power(Xray_energy, 1.68))


def electron_range(element, beam_energy, density='auto', tilt=0):
    """Returns the maximum electron range for a pure bulk material according to
    the Kanaya-Okayama parameterziation.

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

    Examples
    --------
    >>> # Electron range in pure Copper at 30 kV in micron
    >>> hs.eds.electron_range('Cu', 30.)
    2.8766744984001607

    Notes
    -----
    From Kanaya, K. and S. Okayama (1972). J. Phys. D. Appl. Phys. 5, p43

    See also the textbook of Goldstein et al., Plenum publisher,
    third edition p 72.

    """

    if density == 'auto':
        density = elements_db[
            element]['Physical_properties']['density (g/cm^3)']
    Z = elements_db[element]['General_properties']['Z']
    A = elements_db[element]['General_properties']['atomic_weight']
    # Note: magic numbers here are from Kanaya-Okayama parameterization. See
    # docstring for associated references.
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
        The tilt of the stage in degrees. The sample is facing the detector
        when positively tilted.
    azimuth_angle: float
        The azimuth of the detector in degrees. 0 is perpendicular to the tilt
        axis.
    elevation_angle: float
        The elevation of the detector in degrees.

    Returns
    -------
    take_off_angle: float.
        In degrees.

    Examples
    --------
    >>> hs.eds.take_off_angle(tilt_stage=10.,
    >>>                          azimuth_angle=45., elevation_angle=22.)
    28.865971201155283

    Notes
    -----
    Defined by M. Schaffer et al., Ultramicroscopy 107(8), pp 587-597 (2007)

    """

    a = math.radians(90 + tilt_stage)
    b = math.radians(azimuth_angle)
    c = math.radians(elevation_angle)

    return math.degrees(np.arcsin(-math.cos(a) * math.cos(b) * math.cos(c) +
                                  math.sin(a) * math.sin(c)))


def xray_lines_model(elements,
                     beam_energy=200,
                     weight_percents=None,
                     energy_resolution_MnKa=130,
                     energy_axis=None):
    """
    Generate a model of X-ray lines using a Gaussian distribution for each
    peak.

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
    from hyperspy import components1d
    if energy_axis is None:
        energy_axis = {'name': 'E', 'scale': 0.01, 'units': 'keV',
                       'offset': -0.1, 'size': 1024}
    s = EDSTEMSpectrum(np.zeros(energy_axis['size']), axes=[energy_axis])
    s.set_microscope_parameters(
        beam_energy=beam_energy,
        energy_resolution_MnKa=energy_resolution_MnKa)
    s.add_elements(elements)
    counts_rate = 1.
    live_time = 1.
    if weight_percents is None:
        weight_percents = [100. / len(elements)] * len(elements)
    m = s.create_model()
    if len(elements) == len(weight_percents):
        for (element, weight_percent) in zip(elements, weight_percents):
            for line, properties in elements_db[
                    element]['Atomic_properties']['Xray_lines'].items():
                line_energy = properties['energy (keV)']
                ratio_line = properties['weight']
                if s._get_xray_lines_in_spectral_range(
                        [element + '_' + line])[1] == []:
                    g = components1d.Gaussian()
                    g.centre.value = line_energy
                    g.sigma.value = get_FWHM_at_Energy(
                        energy_resolution_MnKa, line_energy) / sigma2fwhm
                    g.A.value = live_time * counts_rate * \
                        weight_percent / 100 * ratio_line
                    m.append(g)
    else:
        raise ValueError("The number of elements specified is not the same \
                         as the number of weight_percents")

    s.data = m.as_signal().data
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
        dim2 = reduce(lambda x, y: x * y, dim[1:])
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

    other_index = list(range(len(kfactors)))
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


def quantification_zeta_factor(intensities,
                               zfactors,
                               dose):
    """
    Quantification using the zeta-factor method

    Parameters
    ----------
    intensities: numpy.array
        The intensities for each X-ray line. The first axis should be the
        elements axis.
    zfactors: list of float
        The list of zeta-factors in the same order as intensities
        e.g. zfactors = [628.10, 539.89] for ['As_Ka', 'Ga_Ka'].
    dose: float
        The total electron dose given by i*t*N, i the current,
        t the acquisition time and
        N the number of electrons per unit electric charge (1/e).

    Returns
    ------
    A numpy.array containing the weight fraction with the same
    shape as intensities and mass thickness in kg/m^2.
    """

    sumzi = np.zeros_like(intensities[0], dtype='float')
    composition = np.zeros_like(intensities, dtype='float')
    for intensity, zfactor in zip(intensities, zfactors):
        sumzi = sumzi + intensity * zfactor
    for i, (intensity, zfactor) in enumerate(zip(intensities, zfactors)):
        composition[i] = intensity * zfactor / sumzi
    mass_thickness = sumzi / dose
    return composition, mass_thickness


def quantification_cross_section(intensities,
                                 cross_sections,
                                 dose):
    """
    Quantification using EDX cross sections
    Calculate the atomic compostion and the number of atoms per pixel
    from the raw X-ray intensity
    Parameters
    ----------
    intensity : numpy.array
        The integrated intensity for each X-ray line, where the first axis
        is the element axis.
    cross_sections : list of floats
        List of X-ray scattering cross-sections in the same order as the
        intensities.
    dose: float
        the dose per unit area given by i*t*N/A, i the current,
        t the acquisition time, and
        N the number of electron by unit electric charge.

    Returns
    -------
    numpy.array containing the atomic fraction of each element, with
    the same shape as the intensity input.
    numpy.array of the number of atoms counts for each element, with the same
    shape as the intensity input.
    """
    shp = len(intensities.shape)-1
    slices = (slice(None),) + (None,) * shp
    x_sections = np.array(cross_sections, dtype='float')[slices]
    number_of_atoms = intensities / (x_sections * dose * 1e-10)
    total_atoms = np.cumsum(number_of_atoms, axis=0)[-1]
    composition = number_of_atoms / total_atoms

    return composition, number_of_atoms


def edx_cross_section_to_zeta(cross_sections, elements):
    """Convert a list of cross_sections in barns (b) to zeta-factors (kg/m^2).

    Parameters
    ----------
    cross_section: list of float
        A list of cross sections in barns.
    elements: list of str
        A list of element chemical symbols in the same order as the
        cross sections e.g. ['Al','Zn']

    Returns
    -------
    zeta_factors : list of float
        zeta_factors with units kg/m^2.

    """
    if len(elements) != len(cross_sections):
        raise ValueError(
            'The number of elements must match the number of cross sections.')
    zeta_factors = []
    for i, element in enumerate(elements):
        atomic_weight = elements_db[element]['General_properties'][
            'atomic_weight']
        zeta = atomic_weight / (cross_sections[i] * constants.Avogadro * 1E-25)
        zeta_factors.append(zeta)
    return zeta_factors


def zeta_to_edx_cross_section(zfactors, elements):
    """Convert a list of zeta-factors (kg/m^2) to cross_sections in barns (b).

    Parameters
    ----------
    zfactors: list of float
        A list of zeta-factors.
    elements: list of str
        A list of element chemical symbols in the same order as the
        cross sections e.g. ['Al','Zn']

    Returns
    -------
    cross_sections : list of float
        cross_sections with units in barns.

    """
    if len(elements) != len(zfactors):
        raise ValueError(
            'The number of elements must match the number of cross sections.')
    cross_sections = []
    for i, element in enumerate(elements):
        atomic_weight = elements_db[element]['General_properties'][
            'atomic_weight']
        xsec = atomic_weight / (zfactors[i] * constants.Avogadro * 1E-25)
        cross_sections.append(xsec)
    return cross_sections
