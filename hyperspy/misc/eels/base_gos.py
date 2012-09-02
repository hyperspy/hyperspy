from __future__ import division

import numpy as np

from hyperspy.misc.utils import get_linear_interpolation
from hyperspy.misc.eels.edges_dictionary import edges_dict


class GOSBase(object):
    def read_edges_dict(self):
        element = self.element
        # Convert to the "GATAN" nomenclature
        if self.subshell == "K" :
            subshell = "K1"
        else:
            subshell = self.subshell
        if edges_dict.has_key(element) is not True:
            raise ValueError("The given element " + element + 
                              " is not in the database.")
        elif not edges_dict[element]['subshells'].has_key(subshell):
            raise ValueError(
                "The given subshell " + subshell + 
                " is not in the database.\n" + 
                "The available subshells are:\n" + 
            str(edges_dict[element]['subshells'].keys()))
            
        self.onset_energy = \
            edges_dict[element]['subshells'][subshell]['onset_energy']
        self.subshell_factor = \
            edges_dict[element]['subshells'][subshell]['factor']
        self._subshell = subshell
        self.Z = edges_dict[element]['Z']
        self.element_dict = edges_dict[element]
        
    def get_parametrized_qaxis(self, k1, k2, n):
        return k1*(np.exp(np.arange(n)*k2) - 1)*1e10

    def get_parametrized_energy_axis(self, k1, k2, n):
        return k1*(np.exp(np.arange(n)*k2/k1) - 1)
    
    def get_qaxis_and_gos(self, ienergy, qmin, qmax):
        qgosi = self.gos_array[ienergy, :]
        if qmax > self.qaxis[-1]:
            # Linear extrapolation
            g1, g2 = qgosi[-2:]
            q1, q2 = self.qaxis[-2:]
            gosqmax = get_linear_interpolation((q1, g1), (q2, g2), qmax)
            qaxis = np.hstack((self.qaxis, qmax))
            qgosi = np.hstack((qgosi, gosqmax))
        else:
            index = self.qaxis.searchsorted(qmax)
            g1, g2 = qgosi[index - 1:index+1]
            q1, q2 = self.qaxis[index - 1 : index+1]
            gosqmax = get_linear_interpolation((q1, g1), (q2, g2), qmax)
            qaxis = np.hstack((self.qaxis[:index], qmax))
            qgosi = np.hstack((qgosi[:index,], gosqmax))
            
        if qmin > 0:
            index = self.qaxis.searchsorted(qmin)
            g1, g2 = qgosi[index - 1:index + 1]
            q1, q2 = qaxis[index - 1:index + 1]
            gosqmin = get_linear_interpolation((q1, g1), (q2, g2), qmin)
            qaxis = np.hstack((qmin, qaxis[index:]))
            qgosi = np.hstack((gosqmin, qgosi[index:],))
        return qaxis, qgosi.clip(0)
