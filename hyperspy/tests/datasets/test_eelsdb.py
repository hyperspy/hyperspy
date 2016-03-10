import nose.tools as nt
from nose.plugins.skip import SkipTest
import requests
from hyperspy.misc.eels.eelsdb import eelsdb


def test_eelsdb_eels():
    try:
        request = requests.get('http://api.eelsdb.eu',)
    except requests.exceptions.ConnectionError:
        raise SkipTest
    ss = eelsdb(
        title="Boron Nitride Multiwall Nanotube",
        formula="BN",
        spectrum_type="coreloss",
        edge="K",
        min_energy=370,
        max_energy=1000,
        min_energy_compare="gt",
        max_energy_compare="lt",
        resolution="0.7 eV",
        resolution_compare="lt",
        max_n=2,
        order="spectrumMin",
        order_direction='DESC',
        monochromated=False, )
    nt.assert_equal(len(ss), 2)
    md = ss[0].metadata
    nt.assert_equal(md.General.author, "Odile Stephan")
    nt.assert_equal(
        md.Acquisition_instrument.TEM.Detector.EELS.collection_angle, 24)
    nt.assert_equal(md.Acquisition_instrument.TEM.convergence_angle, 15)
    nt.assert_equal(md.Acquisition_instrument.TEM.beam_energy, 100)
    nt.assert_equal(md.Signal.signal_type, "EELS")
    nt.assert_true("perpendicular" in md.Sample.description)
    nt.assert_true("parallel" in ss[1].metadata.Sample.description)
    nt.assert_equal(md.Sample.chemical_formula, "BN")
    nt.assert_equal(md.Acquisition_instrument.TEM.microscope, "STEM-VG")


def test_eelsdb_xas():
    try:
        request = requests.get('http://api.eelsdb.eu',)
    except requests.exceptions.ConnectionError:
        raise SkipTest
    ss = eelsdb(
        spectrum_type="xrayabs", max_n=1,)
    nt.assert_equal(len(ss), 1)
    md = ss[0].metadata
    nt.assert_equal(md.Signal.signal_type, "XAS")
