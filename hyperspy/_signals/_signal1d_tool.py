try:
    from pybaselines import Baseline
except ImportError:
    pass


def _remove_baseline(data, method, x, **kwargs):
    baseline_fitter = getattr(
        Baseline(x, check_finite=False, assume_sorted=True),
        method,
    )
    return data - baseline_fitter(data, **kwargs)[0]
