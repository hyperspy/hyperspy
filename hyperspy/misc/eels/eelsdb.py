
"""
This module provides tools to interact with The EELS Database.

"""
import requests

from hyperspy.io_plugins.msa import parse_msa_string
from hyperspy.io import dict2signal

def eelsdb(type=None, title=None, author=None, element=None, edge=None,
           min_energy=None, max_energy=None, resolution=None,
           resolution_compare=None, monochromated=None, **kwargs):
    """Download spectra from the EELS Data Base.

    Parameters
    ----------
    type: {'coreloss', 'lowloss', 'zeroloss', 'xrayabs'}, optional
    title: string
        Search spectra titles for a text string.
    author: string, optional
        Search authors for a text string.
    element: string or list of strings, optional
        Filter for the presence of one or more element. Each string must
        correspond with a valid element symbol.
    edge: {'K', 'L1', 'L2,3', 'M2,3', 'M4,5', 'N2,3', 'N45' 'O23', 'O45'}, optional
        Filter for spectra with a specific class of edge.
    min_energy, max_energy: float, optional
        Minimum and maximum energy in eV.
    resolution: float, optional
        Energy resolution in eV.
    resolution_compare: string, optional
        lt / eq / gt (less than / equal to / greater than)
    monochromated: bool

    Returns
    -------
    spectra: list
        A list containing all the spectra matching the given criteria if
        any.

    """

    if type is not None and type not in {
        'coreloss', 'lowloss', 'zeroloss', 'xrayabs'}:
        raise ValueError("type must be one of \'coreloss\', \'lowloss\', "
                         "\'zeroloss\', \'xrayabs\'.")
    params = {
        "type": type,
        "title": title,
        "author": author,
        "edge": edge,
        "min_energy": min_energy,
        "max_energy": max_energy,
        "resolution": resolution,
        "resolution_compare": resolution_compare,
        "monochromated": monochromated,
    }

    if isinstance(element, basestring):
        params["element"] = element
    else:
        params["element[]"] = element
    params.update(kwargs)

    request = requests.get('http://api.eelsdb.eu/spectra',
                           params=params)
    spectra = []
    for json_spectrum in request.json():
        download_link = json_spectrum['download_link']
        msa_string = requests.get(download_link).text
        spectra.append(dict2signal(parse_msa_string(msa_string)[0]))

    return spectra
