import traitsui.api as tu
from traitsui.menu import OKButton, CancelButton

traits_view = tu.View(
    tu.Group('beam_energy',
             'tilt_stage',
             label='SEM', show_border=True),
    tu.Group('live_time', 'azimuth_angle',
             'elevation_angle', 'energy_resolution_MnKa',
             label='EDS', show_border=True),
    kind='modal', buttons=[OKButton, CancelButton],
    title='SEM parameters definition wizard')

traits_view = tu.View(
    tu.Group('beam_energy',
             'tilt_stage',
             'probe_area',
             'beam_current',
             label='TEM', show_border=True),
    tu.Group('real_time', 'live_time', 'azimuth_angle',
             'elevation_angle', 'energy_resolution_MnKa',
             label='EDS', show_border=True),
    kind='modal', buttons=[OKButton, CancelButton],
    title='TEM parameters definition wizard')

traits_view = tu.View(
    tu.Group('beam_energy',
             'convergence_angle',
             label='TEM', show_border=True),
    tu.Group('collection_angle',
             label='EELS', show_border=True),
    kind='modal', buttons=[OKButton, CancelButton],
    title='TEM parameters definition wizard')
