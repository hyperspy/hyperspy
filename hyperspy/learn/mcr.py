import numpy as np
import scipy.linalg
from pymcr.mcr import McrAls
import logging

_logger = logging.getLogger(__name__)


def sum_to_one(specs, concs):
    specs = (specs.T/specs.sum(1)).T
    concs = concs/specs.sum(1)
    return specs, concs


def flipfactors(factors, maps):
    sign = np.sign(factors.sum(0))
    factors_flipped = sign * factors
    maps_flipped = sign * maps
    return factors_flipped, maps_flipped


def varimax(a, reltol=None, maxit=250):
    # ORTHOMAX Orthogonal rotation of FA or PCA loadings.
    pix, channels = a.shape

    if not reltol:
        reltol = np.sqrt(np.spacing(np.array(1).astype('float32')))
    # De facto, the intial rotation matrix is identity.
    t = np.eye(channels)
    b = np.dot(a, t)

    converged = False
    # Use Lawley and Maxwell's fast version

    # Choose a random rotation matrix if identity rotation
    # makes an obviously bad start.
    l, _, m = scipy.linalg.svd(np.dot(a.T, (pix * b**3 - np.dot(b, np.diag(np.sum(b**2, 0))))))
    t = np.dot(l, m.T)
    if scipy.linalg.norm(t - np.eye(channels)) < reltol:
        # Using identity as the initial rotation matrix, the first
        # iteration does not move the loadings enough to escape the
        # the convergence criteria.  Therefore, pick an initial rotation
        # matrix at random.
        [t, _] = scipy.linalg.qr(np.random.randn(channels, channels))
        b = np.dot(a, t)

    d = 0
    for k in range(0, maxit):
        d_old = d
        l, d, m = scipy.linalg.svd(np.dot(a.T, (pix * b**3 - np.dot(b, np.diag(np.sum(b**2, 0))))))
        t = np.dot(l, m.T)
        d = np.sum(np.diag(d))
        b = np.dot(a, t)
        if np.abs(d - d_old)/d < reltol:
            converged = True
            _logger.info("Varimax rotation completed after %s iterations" % str(k))
            break
    if not converged:
        _logger.warning("Rotation did not converge in maximum number of iterataions: %s" % str(maxit))
    return b, t


def mcr(data, concentrations, purespectra, poisson_scale, im_weight_vec,
        spec_weight_vec, simplicity):
    data = data.reshape([data.shape[0]*data.shape[1], data.shape[2]])

    if simplicity == 'spatial':
        rot_conc, rotation = varimax(concentrations.T)
        rot_spec = np.dot(purespectra, rotation)
        rot_spec, rot_conc = flipfactors(rot_spec, rot_conc)
        rot_spec[rot_spec < 0] = 0
        if poisson_scale:
            rot_spec = (rot_spec / spec_weight_vec)
            rot_conc = (rot_conc / im_weight_vec)
            data = (data.T / im_weight_vec[:, 0]).T / spec_weight_vec[:, 0]

        fitmcr = McrAls(max_iter=50, tol_err_change=1e-6)
        fitmcr.fit(data.T, C=rot_spec, verbose=False)

        if poisson_scale:
            conc_out = (fitmcr.ST_opt_.T * im_weight_vec)
            spec_out = (fitmcr.C_opt_ * spec_weight_vec).T
            # data = (data.T * im_weight_vec[:, 0]).T * spec_weight_vec[:, 0]
        else:
            conc_out = fitmcr.ST_opt_
            spec_out = fitmcr.C_opt_

    elif simplicity == 'spectral':
        rot_spec, rotation = varimax(purespectra)
        rot_conc = np.dot(concentrations.T, rotation)
        rot_spec, rot_conc = flipfactors(rot_spec, rot_conc)
        rot_spec[rot_spec < 0] = 0
        if poisson_scale:
            rot_spec = (rot_spec / spec_weight_vec)
            rot_conc = (rot_conc / im_weight_vec)

        fitmcr = McrAls(max_iter=50, tol_err_change=1e-6)
        fitmcr.fit(data, ST=rot_spec.T, verbose=False)

        if poisson_scale:
            conc_out = (fitmcr.C_opt_ * im_weight_vec)
            spec_out = (fitmcr.ST_opt_.T * spec_weight_vec).T
            # data = (data.T * im_weight_vec[:, 0]).T * spec_weight_vec[:, 0]
        else:
            conc_out = fitmcr.C_opt_
            spec_out = fitmcr.ST_opt_

    else:
        raise ValueError(""
                         "'simplicity' must be either 'spatial' or"
                         "'spectral'."
                         "%s was provided." % str(simplicity))

    spec_out, conc_out = sum_to_one(spec_out, conc_out)
    return spec_out, conc_out
