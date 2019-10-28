"""

Functions that operate on Signal instances and other goodies.

    stack
        Stack Signal instances.

Subpackages:

    material
        Tools related to the material under study.
    plot
        Tools for plotting.
    eds
        Tools for energy-dispersive X-ray data analysis.
    example_signals
        A few example of signal


"""
import hyperspy.utils.material
import hyperspy.utils.eds
import hyperspy.utils.plot
import hyperspy.datasets.example_signals
import hyperspy.utils.model
from hyperspy.misc.utils import (stack, transpose)
from hyperspy.interactive import interactive
import hyperspy.utils.roi
import hyperspy.utils.samfire


def print_known_signal_types():
    """Print all known `signal_type`s

    This includes `signal_type`s from all installed packages that
    extend HyperSpy.

    Examples
    --------
    >>> hs.print_known_signal_types()
    +--------------------+---------------------+--------------------+----------+
    |    signal_type     |       aliases       |     class name     | package  |
    +--------------------+---------------------+--------------------+----------+
    | DielectricFunction | dielectric function | DielectricFunction | hyperspy |
    |      EDS_SEM       |                     |   EDSSEMSpectrum   | hyperspy |
    |      EDS_TEM       |                     |   EDSTEMSpectrum   | hyperspy |
    |        EELS        |       TEM EELS      |    EELSSpectrum    | hyperspy |
    |      hologram      |                     |   HologramImage    | hyperspy |
    |      MySignal      |                     |      MySignal      | hspy_ext |
    +--------------------+---------------------+--------------------+----------+

    """
    from hyperspy.ui_registry import ALL_EXTENSIONS
    from prettytable import PrettyTable
    from hyperspy.misc.utils import print_html
    table = PrettyTable()
    table.field_names = [
        "signal_type",
        "aliases",
        "class name",
        "package"]
    for sclass, sdict in ALL_EXTENSIONS["signals"].items():
        # skip lazy signals and non-data-type specific signals
        if sdict["lazy"] or not sdict["signal_type"]:
            continue
        aliases = (", ".join(sdict["signal_type_aliases"])
                   if "signal_type_aliases" in sdict
                   else "")
        package = sdict["module"].split(".")[0]
        table.add_row([sdict["signal_type"], aliases, sclass, package])
        table.sortby = "class name"
    return print_html(f_text=table.get_string,
                      f_html=table.get_html_string)
