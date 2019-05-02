import numpy as np
import scipy.linalg
from pymcr.mcr import McrAR
from hyperspy.misc.machine_learning.orthomax import orthomax
import logging

_logger = logging.getLogger(__name__)


def mcr(data, concentrations, purespectra, poisson_scale, im_weight_vec,
        spec_weight_vec, simplicity):
    data = data.reshape([data.shape[0]*data.shape[1], data.shape[2]])

    if simplicity == 'spatial':
        rot_conc, rotation = orthomax(concentrations.T, gamma=1)
        rot_conc = np.array(rot_conc)
        rotation = np.array(rotation)
        rot_spec = np.dot(purespectra, rotation)
        rot_spec = np.sign(rot_spec.sum(0)) * rot_spec

        rot_spec[rot_spec < 0] = 0
        if poisson_scale:
            rot_spec = (rot_spec.T / spec_weight_vec).T
            rot_spec = np.nan_to_num(rot_spec)
            data = (data.T / im_weight_vec).T / spec_weight_vec
            data = np.nan_to_num(data)

        fitmcr = McrAR(max_iter=50, tol_err_change=1e-6)
        fitmcr.fit(data.T, C=rot_spec, verbose=False)

        if poisson_scale:
            conc_out = (fitmcr.ST_opt_ * im_weight_vec).T
            spec_out = (fitmcr.C_opt_.T * spec_weight_vec).T
            data = (data.T * im_weight_vec).T / spec_weight_vec
        else:
            conc_out = fitmcr.ST_opt_.T
            spec_out = fitmcr.C_opt_

    elif simplicity == 'spectral':
        rot_spec, rotation = orthomax(purespectra, gamma=1)
        rot_spec = np.array(rot_spec)
        rotation = np.array(rotation)
        rot_spec = np.sign(rot_spec.sum(0)) * rot_spec
        rot_spec[rot_spec < 0] = 0
        if poisson_scale:
            rot_spec = (rot_spec.T / spec_weight_vec).T
            rot_spec = np.nan_to_num(rot_spec)
            data = (data.T / im_weight_vec).T / spec_weight_vec
            data = np.nan_to_num(data)

        fitmcr = McrAR(max_iter=50, tol_err_change=1e-6)
        fitmcr.fit(data, ST=rot_spec.T, verbose=False)

        if poisson_scale:
            conc_out = (fitmcr.C_opt_.T * im_weight_vec).T
            spec_out = (fitmcr.ST_opt_ * spec_weight_vec).T
            data = (data.T * im_weight_vec).T / spec_weight_vec
        else:
            conc_out = fitmcr.C_opt_
            spec_out = fitmcr.ST_opt_.T

    else:
        raise ValueError("'simplicity' must be either 'spatial' or"
                         "'spectral'."
                         "{} was provided.".format(str(simplicity)))

    spec_out = spec_out/spec_out.sum(0)
    conc_out = conc_out/spec_out.sum(0)
    return spec_out, conc_out
