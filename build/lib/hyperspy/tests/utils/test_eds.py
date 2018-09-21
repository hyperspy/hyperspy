

from hyperspy.misc.eds.utils import get_xray_lines_near_energy


def test_xray_lines_near_energy():
    E = 1.36
    lines = get_xray_lines_near_energy(E)
    assert (
        lines ==
        ['Pm_M2N4', 'Ho_Ma', 'Eu_Mg', 'Se_La', 'Br_Ln', 'W_Mz', 'As_Lb3',
         'Kr_Ll', 'Ho_Mb', 'Ta_Mz', 'Dy_Mb', 'As_Lb1', 'Gd_Mg', 'Er_Ma',
         'Sm_M2N4', 'Mg_Kb', 'Se_Lb1', 'Ge_Lb3', 'Br_Ll', 'Sm_Mg', 'Dy_Ma',
         'Nd_M2N4', 'As_La', 'Re_Mz', 'Hf_Mz', 'Kr_Ln', 'Er_Mb', 'Tb_Mb'])
    lines = get_xray_lines_near_energy(E, 0.02)
    assert lines == ['Pm_M2N4']
    E = 5.4
    lines = get_xray_lines_near_energy(E)
    assert (
        lines ==
        ['Cr_Ka', 'La_Lb2', 'V_Kb', 'Pm_La', 'Pm_Ln', 'Ce_Lb3', 'Gd_Ll',
         'Pr_Lb1', 'Xe_Lg3', 'Pr_Lb4'])
    lines = get_xray_lines_near_energy(E, only_lines=('a', 'b'))
    assert (
        lines ==
        ['Cr_Ka', 'V_Kb', 'Pm_La', 'Pr_Lb1'])
    lines = get_xray_lines_near_energy(E, only_lines=('a'))
    assert (
        lines ==
        ['Cr_Ka', 'Pm_La'])
