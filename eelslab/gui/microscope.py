from eelslab import microscope

import enthought.traits.api as t
import enthought.traits.ui.api as tui

class Microscope(microscope.Microscope, t.HasTraits):   
    microscopes = {}
    name = t.String
    E0 = t.Float
    alpha = t.Float
    beta = t.Float
    pppc = t.Float
    correlation_factor = t.Float

