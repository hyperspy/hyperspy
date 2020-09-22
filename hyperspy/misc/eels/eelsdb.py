"""This module provides tools to interact with The EELS Database."""
import requests
import logging

from hyperspy.io_plugins.msa import parse_msa_string
from hyperspy.io import dict2signal

_logger = logging.getLogger(__name__)


def eelsdb(spectrum_type=None, title=None, author=None, element=None, formula=None,
           edge=None, min_energy=None, max_energy=None, resolution=None,
           min_energy_compare="gt", max_energy_compare="lt",
           resolution_compare="lt", max_n=-1, monochromated=None, order=None,
           order_direction="ASC"):
    r"""Download spectra from the EELS Data Base.

    Parameters
    ----------
    spectrum_type: {'coreloss', 'lowloss', 'zeroloss', 'xrayabs'}, optional
    title: string
        Search spectra titles for a text string.
    author: string, optional
        Search authors for a text string.
    element: string or list of strings, optional
        Filter for the presence of one or more element. Each string must
        correspond with a valid element symbol.
    formula: string
        Chemical formula of the sample.
    edge: {'K', 'L1', 'L2,3', 'M2,3', 'M4,5', 'N2,3', 'N4,5' 'O2,3', 'O4,5'}, optional
        Filter for spectra with a specific class of edge.
    min_energy, max_energy: float, optional
        Minimum and maximum energy in eV.
    resolution: float, optional
        Energy resolution in eV.
    resolution_compare: {"lt", "eq", "gt"}, optional, default "lt"
        "lt" to search for all spectra with resolution less than `resolution`.
        "eq" for equal, "gt" for greater than.
    min_energy_compare, max_energy_compare: {"lt", "eq", "gt"}, optional
        "lt" to search for all spectra with min/max energy less than
        `min_energy`\`max_energy`. "eq" for equal, "gt" for greater than.
        Deafault values are "gt"/"lt" for `min_energy`\`max_energy`
        respectively.
    monochromated: bool or None (default)
    max_n: int, default -1
        Maximum number of spectra to return. -1 to return all.
    order: string
        Key to sort results by. Valid keys are:
        * "spectrumType",
        * "spectrumMin",
        * "spectrumMax",
        * "stepSize",
        * "spectrumFormula",
        * "spectrumElement",
        * "spectrumUpload",
        * "source_purity",
        * "spectrumEdges",
        * "microscope",
        * "guntype",
        * "beamenergy",
        * "resolution",
        * "monochromated",
        * "acquisition_mode",
        * "convergence",
        * "collection",
        * "probesize",
        * "beamcurrent",
        * "integratetime",
        * "readouts",
        * "detector",
        * "darkcurrent",
        * "gainvariation",
        * "calibration",
        * "zeroloss_deconv",
        * "thickness",
        * "deconv_fourier_log",
        * "deconv_fourier_ratio",
        * "deconv_stephens_deconvolution",
        * "deconv_richardson_lucy",
        * "deconv_maximum_entropy",
        * "deconv_other",
        * "assoc_spectra",
        * "ref_freetext",
        * "ref_doi",
        * "ref_url",
        * "ref_authors",
        * "ref_journal",
        * "ref_volume",
        * "ref_issue",
        * "ref_page",
        * "ref_year",
        * "ref_title",
        * "otherURLs"
    order_direction : {"ASC", "DESC"}
        Sorting `order` direction.


    Returns
    -------
    spectra: list
        A list containing all the spectra matching the given criteria if
        any.

    """
    # Verify arguments
    if spectrum_type is not None and spectrum_type not in {
            'coreloss', 'lowloss', 'zeroloss', 'xrayabs'}:
        raise ValueError("spectrum_type must be one of \'coreloss\', \'lowloss\', "
                         "\'zeroloss\', \'xrayabs\'.")
    valid_edges = [
        'K', 'L1', 'L2,3', 'M2,3', 'M4,5', 'N2,3', 'N4,5', 'O2,3', 'O4,5']
    valid_order_keys = [
        "spectrumType",
        "spectrumMin",
        "spectrumMax",
        "stepSize",
        "spectrumFormula",
        "spectrumElement",
        "spectrumUpload",
        "source_purity",
        "spectrumEdges",
        "microscope",
        "guntype",
        "beamenergy",
        "resolution",
        "monochromated",
        "acquisition_mode",
        "convergence",
        "collection",
        "probesize",
        "beamcurrent",
        "integratetime",
        "readouts",
        "detector",
        "darkcurrent",
        "gainvariation",
        "calibration",
        "zeroloss_deconv",
        "thickness",
        "deconv_fourier_log",
        "deconv_fourier_ratio",
        "deconv_stephens_deconvolution",
        "deconv_richardson_lucy",
        "deconv_maximum_entropy",
        "deconv_other",
        "assoc_spectra",
        "ref_freetext",
        "ref_doi",
        "ref_url",
        "ref_authors",
        "ref_journal",
        "ref_volume",
        "ref_issue",
        "ref_page",
        "ref_year",
        "ref_title",
        "otherURLs"]
    if edge is not None and edge not in valid_edges:
        raise ValueError("`edge` must be one of %s." % ", ".join(valid_edges))

    if order is not None and order not in valid_order_keys:
        raise ValueError("`order` must be one of %s." % ", ".join(
            valid_order_keys))
    if order_direction is not None and order_direction not in ["ASC", "DESC"]:
        raise ValueError("`order_direction` must be \"ASC\" or \"DESC\".")
    for kwarg, label in (
            (resolution_compare, "resolution_compare"),
            (min_energy_compare, "min_energy_compare"),
            (max_energy_compare, "max_energy_compare")):
        if kwarg not in ("lt", "gt", "eq"):
            raise ValueError("`%s` must be \"lt\", \"eq\" or \"gt\"." %
                             label)
    if monochromated is not None:
        monochromated = 1 if monochromated else 0
    params = {
        "type": spectrum_type,
        "title": title,
        "author": author,
        "edge": edge,
        "min_energy": min_energy,
        "max_energy": max_energy,
        "resolution": resolution,
        "resolution_compare": resolution_compare,
        "monochromated": monochromated,
        "formula": formula,
        "min_energy_compare": min_energy_compare,
        "max_energy_compare": max_energy_compare,
        "per_page": max_n,
        "order": order,
        "order_direction": order_direction,
    }

    if isinstance(element, str):
        params["element"] = element
    else:
        params["element[]"] = element

    request = requests.get('http://api.eelsdb.eu/spectra', params=params, )
    spectra = []
    jsons = request.json()
    if "message" in jsons:
        # Invalid query, EELSdb raises error.
        raise IOError(
            "Please report the following error to the HyperSpy developers: "
            "%s" % jsons["message"])
    for json_spectrum in jsons:
        download_link = json_spectrum['download_link']
        msa_string = requests.get(download_link).text
        try:
            s = dict2signal(parse_msa_string(msa_string)[0])
            emsa = s.original_metadata
            s.original_metadata = s.original_metadata.__class__(
                {'json': json_spectrum})
            s.original_metadata.emsa = emsa
            spectra.append(s)

        except:
            # parse_msa_string or dict2signal may fail if the EMSA file is not
            # a valid one.
            _logger.exception(
                "Failed to load the spectrum.\n"
                "Title: %s id: %s.\n"
                "Please report this error to http://eelsdb.eu/about \n" %
                (json_spectrum["title"], json_spectrum["id"]))
    if not spectra:
        _logger.info(
            "The EELS database does not contain any spectra matching your query"
            ". If you have some, why not submitting them "
            "https://eelsdb.eu/submit-data/ ?\n")
    else:
        # Add some info from json to metadata
        # Values with units are not yet supported by HyperSpy (v0.8) so
        # we can't get map those fields.
        for s in spectra:
            if spectrum_type == "xrayabs":
                s.set_signal_type("XAS")
            json_md = s.original_metadata.json
            s.metadata.General.title = json_md.title
            if s.metadata.Signal.signal_type == "EELS":
                if json_md.elements:
                    try:
                        s.add_elements(json_md.elements)
                    except ValueError:
                        _logger.exception(
                            "The following spectrum contains invalid chemical "
                            "element information:\n"
                            "Title: %s id: %s. Elements: %s.\n"
                            "Please report this error in "
                            "http://eelsdb.eu/about \n" %
                            (json_md.title, json_md.id, json_md.elements))
                if "collection" in json_md and " mrad" in json_md.collection:
                    beta = float(json_md.collection.replace(" mrad", ""))
                    s.metadata.set_item(
                        "Acquisition_instrument.TEM.Detector.EELS.collection_angle",
                        beta)
                if "convergence" in json_md and " mrad" in json_md.convergence:
                    alpha = float(json_md.convergence.replace(" mrad", ""))
                    s.metadata.set_item(
                        "Acquisition_instrument.TEM.convergence_angle", alpha)
                if "beamenergy" in json_md and " kV" in json_md.beamenergy:
                    beam_energy = float(json_md.beamenergy.replace(" kV", ""))
                    s.metadata.set_item(
                        "Acquisition_instrument.TEM.beam_energy", beam_energy)
            # We don't yet support units, so we cannot map the thickness
            # s.metadata.set_item("Sample.thickness", json_md.thickness)
            s.metadata.set_item("Sample.description", json_md.description)
            s.metadata.set_item("Sample.chemical_formula", json_md.formula)
            s.metadata.set_item("General.author", json_md.author.name)
            s.metadata.set_item("Acquisition_instrument.TEM.microscope",
                                json_md.microscope)

    return spectra
