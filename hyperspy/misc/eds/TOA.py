import math
import numpy as np

def TOA(self):
    """Calculates the take-off-angle, the angle with which the X-rays
    leave the surface towards the detector.
    
    'SEM.tilt_stage', 'SEM.EDS.azimuth_angle' and 
    'SEM.EDS.elevation_angle' need to be defined (in Degree) in
    'mapped_parameters'.
                
    Returns
    -------
    float : TOA in Degree
    
    Notes
    -----
    Defined by M. Schaffer et al., Ultramicroscopy 107(8), pp 587-597 (2007)
    
    """
    mp = self.mapped_parameters
    a = math.radians(90-mp.SEM.tilt_stage)
    b = math.radians(mp.SEM.EDS.azimuth_angle)
    c = math.radians(mp.SEM.EDS.elevation_angle)
    
    return math.degrees( np.arcsin (-math.cos(a)*math.cos(b)*math.cos(c) \
    + math.sin(a)*math.sin(c)))
